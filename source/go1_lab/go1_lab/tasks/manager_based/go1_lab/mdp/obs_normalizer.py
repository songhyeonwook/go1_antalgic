# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""관측 스케일링. yaml `train.normalize` 의 고정 상수를 정책 안에 버퍼로 넣는다.

정책 버퍼에 두는 이유
  * checkpoint 에 함께 저장/복원된다. phase 3 의 teacher 는 자기가 학습된 스케일을
    그대로 들고 온다 (rsl_rl 이 `actor_obs_normalizer.*` 를 `teacher_obs_normalizer`
    로 옮긴다). yaml 을 잘못 만져도 teacher 의 동작은 깨지지 않는다.
  * `runner.load` 가 strict 라, 스케일 유무가 어긋나면 Missing/Unexpected key 로
    즉시 멈춘다. yaml 을 비교하는 별도 검사가 필요 없다.
  * isaaclab_rl 의 exporter 가 정규화기를 jit/onnx 에 굽는다 (LSTM student 는 RNN
    앞에 적용). 실기 배포에 상수를 손으로 옮겨 적을 필요가 없다.

평균은 빼지 않고 `x * scale` 만 한다. 실측 평균이 0 이고(ang_vel, joint_vel),
mirror.py 의 좌우 변환이 x*s 형태와만 정확히 교환되기 때문이다.
"""

from __future__ import annotations

import torch
from torch import nn

# ── policy 관측 레이아웃 (49-dim) ────────────────────────────────────────────
#   [ 0: 3] base_ang_vel        angular_vel
#   [ 3: 6] projected_gravity   gravity
#   [ 6: 9] velocity_commands   cmd_vel
#   [ 9:21] joint_pos_rel       joint_pos
#   [21:33] joint_vel           joint_vel
#   [33:45] previous action     action
#   [45:49] injured peg leg     항상 1.0 (0/1 이진이라 나눌 의미가 없다)
ANG_VEL_SPAN = (0, 3)
GRAVITY_SPAN = (3, 6)
COMMAND_SPAN = (6, 9)
JOINT_POS_SPAN = (9, 21)
JOINT_VEL_SPAN = (21, 33)
ACTION_SPAN = (33, 45)
PEG_LEG_SPAN = (45, 49)
POLICY_DIM = 49

# teacher/critic 은 뒤에 privileged 블록이 붙는다 (Go1LabPrivilegedObsCfg 순서).
#   [49] 부목 길이 L    [50:53] base_lin_vel
PRIV_L_IDX = POLICY_DIM
PRIV_VEL_SPAN = (POLICY_DIM + 1, POLICY_DIM + 4)
FULL_DIM = POLICY_DIM + 4

# yaml 키 → 관측 span
_OBS_SPANS = {
    "angular_vel": ANG_VEL_SPAN,
    "gravity": GRAVITY_SPAN,
    "cmd_vel": COMMAND_SPAN,
    "joint_pos": JOINT_POS_SPAN,
    "joint_vel": JOINT_VEL_SPAN,
    "action": ACTION_SPAN,
}

# rsl_rl 의 각 역할이 어떤 obs_groups 키를 concat 해서 입력을 만드는지.
#   ActorCritic:     actor -> "policy",  critic -> "critic"
#   StudentTeacher:  student -> "policy", teacher -> "teacher"
_ROLE_TO_GROUP_KEY = {
    "actor": "policy",
    "critic": "critic",
    "student": "policy",
    "teacher": "teacher",
}


class ObsScaler(nn.Module):
    """관측에 고정 스케일을 곱하는 모듈. rsl_rl 의 정규화기 자리에 그대로 끼운다.

    forward 가 `x * scale` 한 줄이라 TorchScript / ONNX export 가 그대로 통과한다.
    `scale` 은 버퍼라 state_dict 에 들어가고 checkpoint 와 함께 저장/복원된다.
    """

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """rsl_rl 인터페이스 호환용. 고정 상수라 갱신하지 않는다."""
        return

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y / self.scale


def build_obs_scale(num_obs: int, obs_scales: dict, priv_scales: dict) -> torch.Tensor:
    """yaml 상수로 관측 차원만큼의 스케일 벡터를 만든다.

    num_obs 는 49(policy 만) 또는 53(policy + privileged) 이어야 한다.
    """
    if num_obs not in (POLICY_DIM, FULL_DIM):
        raise ValueError(
            f"관측 차원 {num_obs} 가 예상({POLICY_DIM} 또는 {FULL_DIM})과 다릅니다. "
            "관측 구성을 바꿨다면 이 파일 상단의 span 상수를 함께 갱신하세요."
        )

    scale = torch.ones(num_obs)
    for key, (lo, hi) in _OBS_SPANS.items():
        scale[lo:hi] = float(obs_scales.get(key, 1.0))
    # PEG_LEG_SPAN 은 손대지 않는다 (0/1 이진)
    if num_obs == FULL_DIM:
        scale[PRIV_L_IDX] = float(priv_scales.get("splint_length", 1.0))
        scale[PRIV_VEL_SPAN[0] : PRIV_VEL_SPAN[1]] = float(priv_scales.get("linear_vel", 1.0))

    if bool(torch.any(scale <= 0)):
        raise ValueError(f"관측 스케일에 0 이하 값이 있습니다: {scale.tolist()}")
    return scale


def install_obs_scaler(policy: nn.Module, obs, obs_scales: dict, priv_scales: dict) -> list[str]:
    """정책의 관측 정규화기 자리에 고정 스케일러를 끼운다.

    반드시 `runner.load()` **앞에서** 호출해야 한다. 그래야 checkpoint 의 버퍼가
    여기서 만든 값을 덮어쓰고, 스케일 유무가 어긋나면 strict load 가 멈춘다.

    관측 차원은 obs 에서 직접 센다. StudentTeacherRecurrent 는
    `self.student = MLP(rnn_hidden_dim, ...)` 라 첫 Linear 의 in_features 가
    관측 차원이 아니라 rnn_hidden_dim(256) 이기 때문이다.

    Returns:
        실제로 교체한 정규화기 이름 목록.
    """
    device = next(policy.parameters()).device
    obs_groups = getattr(policy, "obs_groups", None)
    if not obs_groups:
        raise RuntimeError("policy.obs_groups 가 없습니다 — 관측 차원을 셀 수 없습니다.")

    replaced: list[str] = []
    for role, group_key in _ROLE_TO_GROUP_KEY.items():
        attr = f"{role}_obs_normalizer"
        if not hasattr(policy, attr) or group_key not in obs_groups:
            continue
        num_obs = sum(int(obs[g].shape[-1]) for g in obs_groups[group_key])
        scale = build_obs_scale(num_obs, obs_scales, priv_scales)
        setattr(policy, attr, ObsScaler(scale).to(device))
        replaced.append(f"{attr}({num_obs})")

    if not replaced:
        raise RuntimeError("교체할 정규화기를 찾지 못했습니다.")
    return replaced

def obs_scale_summary(policy: nn.Module) -> str:
    """정책에 '실제로 들어 있는' 스케일을 그룹 단위로 요약한다.

    반드시 `runner.load()` **뒤에** 호출한다. warmstart/resume 은 checkpoint 값이,
    scratch 는 yaml 값이 최종값이라 그 차이를 여기서 확인할 수 있다.
    """
    lines: list[str] = []
    for role in _ROLE_TO_GROUP_KEY:
        scale = getattr(getattr(policy, f"{role}_obs_normalizer", None), "scale", None)
        if scale is None:
            continue
        s = scale.detach().cpu()
        parts = [f"{key}={float(s[lo]):g}" for key, (lo, _) in _OBS_SPANS.items()]
        parts.append(f"onehot={float(s[PEG_LEG_SPAN[0]]):g}")
        if int(s.numel()) == FULL_DIM:
            parts.append(f"splint_length={float(s[PRIV_L_IDX]):g}")
            parts.append(f"linear_vel={float(s[PRIV_VEL_SPAN[0]]):g}")
        lines.append(f"{role}({int(s.numel())}): " + ", ".join(parts))
    return " | ".join(lines) if lines else "disabled — raw obs (scale 1.0)"