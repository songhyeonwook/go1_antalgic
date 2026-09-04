# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""채널 그룹별 규칙이 다른 관측 정규화 모듈.

rsl_rl 의 EmpiricalNormalization 은 전 차원에 동일한 (mean, std) 규칙을 적용하지만,
이 관측 레이아웃은 채널마다 필요한 처리가 다르다:

  * projected_gravity 는 이미 단위벡터라 나눌 필요가 없고, 평지에서 gz≈-1 이라
    std 가 0 에 가까워 오히려 노이즈만 증폭된다.
  * velocity_commands 는 샘플링 범위를 yaml 이 이미 알고 있으므로 running 통계보다
    고정 스케일이 안정적이다 (커리큘럼으로 범위가 흔들려도 스케일이 고정된다).
  * previous action / peg-leg one-hot 은 각각 정책 출력과 0/1 이라 정규화 대상이 아니다.
  * joint_vel 은 타입 간 RMS 가 2.2 배 차이나므로 타입별로 나눌 실익이 있다.

또한 mean 을 빼지 않는다. 이유는 두 가지다:

  1) 실측 mean 이 이미 0 이다 (ang_vel 10^-3, joint_vel 10^-2 수준) — 뺄 이득이 없다.
  2) mirror.py 의 좌우 미러 변환은 ang_vel 의 wx/wz 부호를 반전시키고 관절은
     FL<->FR / RL<->RR 스왑 + hip 부호 반전을 한다. x/s 형태(원점 보존)면 이 변환과
     정확히 교환되지만, mean 을 빼면 mu 가 미러 대칭일 때만 성립한다. 실측에서
     차원별 mean/std 의 미러 오차는 joint_pos 29%, joint_vel 5.5% 였다.

RMS 를 쓰는 것도 같은 이유다 — RMS^2 = mean^2 + var 이므로 mean 을 빼지 않고도
offset 을 스케일에 반영한다.
"""

from __future__ import annotations

import torch
from torch import nn

# ── policy 관측 레이아웃 (49-dim) ────────────────────────────────────────────
#   [ 0: 3] base_ang_vel        running RMS (축별 3개 통계)
#   [ 3: 6] projected_gravity   그대로
#   [ 6: 9] velocity_commands   yaml 범위로 고정 스케일
#   [ 9:21] joint_pos_rel       그대로
#   [21:33] joint_vel           running RMS (타입별 3개, 네 다리 공유)
#   [33:45] previous action     그대로
#   [45:49] injured peg leg     그대로 (one-hot 4, Normal = [0,0,0,0])
ANG_VEL_SPAN = (0, 3)
GRAVITY_SPAN = (3, 6)
COMMAND_SPAN = (6, 9)
JOINT_POS_SPAN = (9, 21)
JOINT_VEL_SPAN = (21, 33)
ACTION_SPAN = (33, 45)
PEG_LEG_SPAN = (45, 49)
POLICY_DIM = 49

# joint 12 채널의 타입 경계. mirror.py 와 동일한 per-TYPE 순서 (hip x4, thigh x4, calf x4).
JOINT_TYPE_SPANS = ((0, 4), (4, 8), (8, 12))



class ObsGroupNormalizer(nn.Module):
    """관측 채널 그룹마다 다른 규칙을 적용하는 정규화 모듈.

    rsl_rl 의 EmpiricalNormalization 과 인터페이스가 같으므로
    ``policy.actor_obs_normalizer`` 자리에 그대로 끼울 수 있다
    (``forward`` / ``update`` / state_dict 버퍼).

    forward 는 분기 없이 ``x * inv_scale`` 한 줄이라
    TorchScript / ONNX export 가 그대로 통과한다. running 통계는 ``update`` 에서만
    갱신되고 그 결과가 ``_inv_scale`` 버퍼에 반영된다.

    Args:
        num_obs: 이 정규화기가 받는 관측 전체 차원. POLICY_DIM(49) 보다 크면
            초과분(critic/teacher 의 privileged 블록)은 손대지 않고 통과시킨다.
        command_scale: velocity_commands 3 채널을 나눌 값 (vx, vy, wz).
            yaml 의 linear_velocity_x.max / linear_velocity_y_abs /
            angular_velocity_yaw_abs 를 그대로 넘긴다.
        eps: RMS 가 0 에 가까울 때의 하한. rsl_rl 기본값과 동일하게 1e-2.
        until: 누적 샘플이 이 값을 넘으면 통계 갱신을 멈춘다 (None 이면 계속 갱신).
    """

    def __init__(
        self,
        num_obs: int,
        command_scale: tuple[float, float, float] = (1.0, 0.3, 0.4),
        eps: float = 1.0e-2,
        until: int | None = None,
    ) -> None:
        super().__init__()

        if num_obs < POLICY_DIM:
            raise ValueError(
                f"num_obs({num_obs}) 가 policy 레이아웃({POLICY_DIM}) 보다 작습니다. "
                "관측 구성을 바꿨다면 이 파일 상단의 span 상수를 함께 갱신하세요."
            )

        self.eps = float(eps)
        self.until = until

        # ── running RMS 그룹 정의 ────────────────────────────────────────────
        # 각 그룹은 (통계 1개) 를 공유하는 관측 인덱스 묶음이다.
        groups: list[list[int]] = []
        # base_ang_vel: 축별로 따로 (wx, wy, wz) — 미러가 축을 스왑하지 않고
        # 부호만 뒤집으므로 축별 스케일은 미러 안전하다.
        for i in range(*ANG_VEL_SPAN):
            groups.append([i])
        # joint_vel: 타입별로 네 다리를 pooling — 타입 내 편차는 +-6% 인데
        # 타입 간은 2.2 배라 타입 단위가 맞고, 네 다리가 같은 스케일을 쓰므로
        # mirror_joint_tensor 의 스왑/부호반전과 정확히 교환된다.
        for lo, hi in JOINT_TYPE_SPANS:
            groups.append([JOINT_VEL_SPAN[0] + k for k in range(lo, hi)])

        obs_idx: list[int] = []
        grp_idx: list[int] = []
        for g, channels in enumerate(groups):
            obs_idx.extend(channels)
            grp_idx.extend([g] * len(channels))
        num_groups = len(groups)

        # ── 고정 스케일 벡터 ───────────────────────────────────────────────
        inv_scale = torch.ones(num_obs)

        # velocity_commands: yaml 범위로 나눈다. 원점을 지나는 선형 스케일이라
        # standing env(전 채널 0) 의 "정지" 신호가 정규화 후에도 0 으로 남는다.
        cs = torch.as_tensor(command_scale, dtype=torch.float32)
        if cs.numel() != 3 or bool(torch.any(cs <= 0)):
            raise ValueError(f"command_scale 은 양수 3개여야 합니다: {command_scale}")
        inv_scale[COMMAND_SPAN[0] : COMMAND_SPAN[1]] = 1.0 / cs

        self.register_buffer("_inv_scale", inv_scale)
        self.register_buffer("_rms_obs_idx", torch.tensor(obs_idx, dtype=torch.long))
        self.register_buffer("_rms_grp_idx", torch.tensor(grp_idx, dtype=torch.long))
        self.register_buffer("_grp_size", torch.zeros(num_groups).index_add_(
            0, torch.tensor(grp_idx, dtype=torch.long), torch.ones(len(grp_idx))
        ))
        # E[x^2] 의 누적 추정치. 1.0 으로 시작하면 첫 update 전에는 사실상 항등이다.
        self.register_buffer("_mean_sq", torch.ones(num_groups))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    # ── 조회용 프로퍼티 (로깅/디버깅) ───────────────────────────────────────
    @property
    def rms(self) -> torch.Tensor:
        """그룹별 running RMS. 순서는 [wx, wy, wz, jvel_hip, jvel_thigh, jvel_calf]."""
        return torch.sqrt(self._mean_sq).clone()

    @property
    def scale(self) -> torch.Tensor:
        """관측 차원별 실제 나눗셈 값 (1/inv_scale)."""
        return (1.0 / self._inv_scale).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._inv_scale

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """running RMS 를 갱신하고 _inv_scale 에 반영한다."""
        if not self.training:
            return
        if self.until is not None and int(self.count) >= self.until:
            return

        num_samples = x.shape[0]
        self.count += num_samples
        rate = float(num_samples) / float(int(self.count))

        # 그룹별 배치 E[x^2] — 그룹 안의 채널을 함께 pooling 한다.
        sq = (x[:, self._rms_obs_idx] ** 2).mean(dim=0)          # (K,)
        batch_sum = torch.zeros_like(self._mean_sq).index_add_(0, self._rms_grp_idx, sq)
        batch_mean_sq = batch_sum / self._grp_size

        self._mean_sq += rate * (batch_mean_sq - self._mean_sq)
        self._refresh()

    @torch.jit.unused
    def _refresh(self) -> None:
        rms = torch.sqrt(torch.clamp(self._mean_sq, min=0.0)) + self.eps
        self._inv_scale[self._rms_obs_idx] = 1.0 / rms[self._rms_grp_idx]

    @torch.jit.unused
    def reset_count(self) -> None:
        """누적 카운트만 0 으로 되돌린다 (통계값은 유지).

        phase 전환에서 반드시 호출해야 한다. rsl_rl 의 누적평균은 count 가 클수록
        새 샘플의 가중치 rate = n/count 가 작아지므로, phase 1 의 누적치를 그대로
        이어받으면 phase 2 의 분포 변화가 통계에 거의 반영되지 않는다.
        """
        self.count.zero_()

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """정규화 역변환."""
        return y / self._inv_scale


def command_scale_from_cfg(command_cfg: dict) -> tuple[float, float, float]:
    """env yaml 의 command 블록에서 (vx, vy, wz) 스케일을 읽는다.

    값을 하드코딩하지 않고 여기서 읽으면 yaml 범위를 바꿔도 스케일이 함께 따라간다.
    """
    return (
        float(command_cfg["linear_velocity_x"]["max"]),
        float(command_cfg["linear_velocity_y_abs"]),
        float(command_cfg["angular_velocity_yaw_abs"]),
    )


# rsl_rl 의 각 역할이 어떤 obs_groups 키를 concat 해서 입력을 만드는지.
#   ActorCritic:     actor -> "policy",  critic -> "critic"
#   StudentTeacher:  student -> "policy", teacher -> "teacher"
_ROLE_TO_GROUP_KEY = {
    "actor": "policy",
    "critic": "critic",
    "student": "policy",
    "teacher": "teacher",
}


def install_obs_normalizer(
    policy: nn.Module,
    command_scale: tuple[float, float, float],
    obs,
    *,
    enabled: bool = False,
    eps: float = 1.0e-2,
    until: int | None = None,
) -> list[str]:
    """policy 의 관측 정규화기를 ObsGroupNormalizer 로 교체한다.

    rsl_rl 은 cfg 플래그가 False 면 normalizer 를 Identity 로 두고
    ``update_normalization()`` 도 건너뛴다. 따라서 모듈을 갈아끼우는 것과 함께
    ``*_obs_normalization`` 속성도 True 로 올려야 통계가 갱신된다.

    관측 차원은 반드시 obs 에서 직접 센다. 모듈에서 역추론하면 안 된다 —
    StudentTeacherRecurrent 는 ``self.student = MLP(rnn_hidden_dim, ...)`` 라
    첫 Linear 의 in_features 가 관측 차원이 아니라 rnn_hidden_dim(256) 이고,
    num_student_obs / num_teacher_obs 는 어디에도 보관되지 않는다.

    Args:
        policy: runner.alg.policy (ActorCritic 또는 StudentTeacher 계열).
        command_scale: velocity_commands 3 채널의 고정 스케일.
        obs: env.get_observations() 가 준 관측 TensorDict/딕셔너리.
        eps: RMS 하한.
        until: 누적 샘플이 이 값을 넘으면 통계 갱신 중지 (None 이면 계속).

    Returns:
        실제로 교체한 normalizer 이름 목록.
    """

    if not enabled:
        return []
    
    device = next(policy.parameters()).device
    obs_groups = getattr(policy, "obs_groups", None)
    if not obs_groups:
        raise RuntimeError(
            "policy.obs_groups 가 없습니다 — 관측 차원을 셀 수 없습니다."
        )

    replaced: list[str] = []
    for role, group_key in _ROLE_TO_GROUP_KEY.items():
        attr = f"{role}_obs_normalizer"
        if not hasattr(policy, attr) or group_key not in obs_groups:
            continue

        num_obs = sum(int(obs[g].shape[-1]) for g in obs_groups[group_key])
        setattr(
            policy,
            attr,
            ObsGroupNormalizer(num_obs, command_scale, eps=eps, until=until).to(device),
        )
        setattr(policy, f"{role}_obs_normalization", True)
        replaced.append(f"{attr}({num_obs})")

    if not replaced:
        raise RuntimeError("교체할 normalizer 를 찾지 못했습니다.")
    return replaced
