# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go1 좌우 Mirror 유틸리티.

관측/행동을 시상면(xz) 대칭으로 뒤집어, 우측 부상(FR/RR)을 좌측 프레임으로
정규화하거나 대칭 페널티를 계산하는 데 쓴다.

미러 변환:
  - 관절: FL ↔ FR, RL ↔ RR; hip abduction 은 부호 반전
  - 각속도: wx, wz 부호 반전        - 투영 중력: gy 부호 반전
  - 속도 명령: vy_cmd, wz_cmd 반전  - 선속도: vy 부호 반전
"""

from __future__ import annotations

import torch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  관절 미러 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ⚠️ Go1 실제 관절 순서는 타입별 묶음이다 (robot.data.joint_names 로 확인):
#   [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#    FL_calf, FR_calf, RL_calf, RR_calf]  (0..11)
# 좌우 미러는 각 타입 안에서 FL↔FR, RL↔RR 를 스왑한다:
#   hip: 0↔1, 2↔3   thigh: 4↔5, 6↔7   calf: 8↔9, 10↔11
JOINT_MIRROR_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]

# hip abduction (0..3) 만 부호 반전, thigh/calf 는 pitch 라 유지
JOINT_MIRROR_SIGN = torch.tensor(
    [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)


def _get_joint_mirror_sign(device: torch.device) -> torch.Tensor:
    """디바이스에 맞는 미러 부호 텐서를 반환합니다."""
    return JOINT_MIRROR_SIGN.to(device)


def mirror_joint_tensor(x: torch.Tensor) -> torch.Tensor:
    """12차원 관절 텐서(위치/속도/행동)를 좌우 미러링합니다."""
    return x[..., JOINT_MIRROR_IDX] * _get_joint_mirror_sign(x.device)


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    """12차원 행동 벡터를 좌우 미러링합니다."""
    return mirror_joint_tensor(action)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  관측 레이아웃
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# policy 그룹은 고정 49차원이다 (선택적 꼬리 없음).
#
#   [0:3]   base_ang_vel        [3:6]   projected_gravity
#   [6:9]   velocity_commands   [9:21]  joint_pos
#   [21:33] joint_vel           [33:45] actions
#   [45:49] peg_leg one-hot     ← 정상 = [0,0,0,0]
#
# base_lin_vel 은 실기 Go1 에 없어 privileged 그룹으로 이동했다.
_ONEHOT_START = 45
POLICY_CORE_DIM = 49    # 45 + peg_leg one-hot(4)
POLICY_DIMS = (POLICY_CORE_DIM,)                    # 49

# privileged 그룹: [L(1), lin_vel(3)] = 4
PRIVILEGED_DIMS = (4,)


def mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """policy 관측을 좌우 미러링합니다 (마지막 차원으로 레이아웃 판별).

    지원 차원은 :data:`POLICY_DIMS`. 알 수 없는 차원은 조용히 잘못 미러링하는
    대신 예외를 냅니다 — 레이아웃이 어긋난 채 학습/분석이 진행되는 쪽이 훨씬
    위험하기 때문입니다.
    """
    dim = obs.shape[-1]
    if dim not in POLICY_DIMS:
        raise ValueError(
            f"policy 관측 차원 {dim} 를 미러링할 수 없습니다. 지원: {POLICY_DIMS}. "
            "관측 구성을 바꿨다면 mirror.py 의 레이아웃 주석과 POLICY_DIMS 를 함께 갱신하세요."
        )

    m = obs.clone()
    m[..., 0] = -obs[..., 0]   # wx (roll rate)
    m[..., 2] = -obs[..., 2]   # wz (yaw rate)
    m[..., 4] = -obs[..., 4]   # gy
    m[..., 7] = -obs[..., 7]   # vy_cmd
    m[..., 8] = -obs[..., 8]   # wz_cmd
    m[..., 9:21] = mirror_joint_tensor(obs[..., 9:21])    # joint_pos
    m[..., 21:33] = mirror_joint_tensor(obs[..., 21:33])  # joint_vel
    m[..., 33:45] = mirror_joint_tensor(obs[..., 33:45])  # actions

    # peg_leg one-hot [45:49]: 좌우만 스왑, 부호 반전 없음.
    # 정상 [0,0,0,0] 은 스왑해도 그대로라 별도 처리가 필요 없다.
    c = _ONEHOT_START
    m[..., c] = obs[..., c + 1]      # FL ↔ FR
    m[..., c + 1] = obs[..., c]
    m[..., c + 2] = obs[..., c + 3]  # RL ↔ RR
    m[..., c + 3] = obs[..., c + 2]
    return m

def mirror_privileged_obs(obs: torch.Tensor) -> torch.Tensor:
    """privileged 관측을 좌우 미러링합니다 (마지막 차원으로 레이아웃 판별).

    부상 다리 one-hot 을 FL↔FR (0↔1), RL↔RR (2↔3) 로 스왑하고 lin_vel 의 vy 를
    반전합니다. injured_flag / L / μ 는 미러 불변입니다.
    """
    dim = obs.shape[-1]
    if dim not in PRIVILEGED_DIMS:
        raise ValueError(
            f"privileged 관측 차원 {dim} 를 미러링할 수 없습니다. 지원: {PRIVILEGED_DIMS}."
        )

    m = obs.clone()
    m[..., 2] = -obs[..., 2]  # lin_vel 의 vy
    return m


def mirror_full_obs(obs):
    """actor 입력 전체를 미러링합니다 (dict/TensorDict 또는 flat 텐서).

    좌우 대칭 env 를 시상면으로 접어 우측 부상을 좌측 프레임으로 정규화할 때
    씁니다 — 학습된 등변성에 의존하지 않고 좌우 일치를 보장합니다.

    flat 텐서는 policy 단독(:data:`POLICY_DIMS`) 또는 policy+privileged 연결로
    해석합니다. 두 집합의 합은 서로 겹치지 않아 차원만으로 모호함 없이 분해됩니다.
    """
    # dict / TensorDict — 그룹 이름으로 분기
    if hasattr(obs, "keys") and not isinstance(obs, torch.Tensor):
        out = obs.clone() if hasattr(obs, "clone") else dict(obs)
        for key in list(obs.keys()):
            if key == "policy":
                out[key] = mirror_policy_obs(obs[key])
            elif key in ("privileged_obs", "privileged"):
                out[key] = mirror_privileged_obs(obs[key])
            else:
                out[key] = obs[key].clone()
        return out

    # flat 텐서 — policy 단독인지 policy+privileged 연결인지 차원으로 판별
    dim = obs.shape[-1]
    if dim in POLICY_DIMS:
        return mirror_policy_obs(obs)
    for p in POLICY_DIMS:
        for q in PRIVILEGED_DIMS:
            if dim == p + q:
                return torch.cat(
                    [mirror_policy_obs(obs[..., :p]), mirror_privileged_obs(obs[..., p:])],
                    dim=-1,
                )
    raise ValueError(
        f"flat 관측 차원 {dim} 를 분해할 수 없습니다. "
        f"policy={POLICY_DIMS}, privileged={PRIVILEGED_DIMS} 조합만 지원합니다."
    )
