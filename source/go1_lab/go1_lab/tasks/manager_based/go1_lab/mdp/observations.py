from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_shape,
)

from .events import CALF_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =====================================================================
# RLS 부목 길이 추정 채널 (policy/student 관측)
# =====================================================================
# 부목 길이 L 은 실기에서 어떤 인코더에도 나타나지 않고, 착지 시의 기하 구속
#   z_tip - z_stance_foot = c(q, g) + a(q, g)·L = 0
# 을 통해서만 관측된다. 이 등식은 L 에 선형이고 미지수가 스칼라이므로 RLS
# (스칼라 공분산 P)로 추정한다. 정책은 (L̂, √P) 를 명시적 입력으로 받는다.

RLS_L_PRIOR = 0.39   # L 샘플 범위 [0.33, 0.45] 의 중앙 = prior 평균 (m)
RLS_L_SCALE = 0.06   # 정규화 스케일 = 샘플 범위 반폭 (m)
RLS_P0 = RLS_L_SCALE ** 2  # prior 분산


def _ensure_rls_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "_rls_L_hat"):
        env._rls_L_hat = torch.full(
            (env.num_envs,), RLS_L_PRIOR, device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_rls_P"):
        env._rls_P = torch.full(
            (env.num_envs,), RLS_P0, device=env.device, dtype=torch.float32
        )


@generic_io_descriptor(
    observation_type="SplintEstimate",
    units="normalized",
    element_order=["L_hat_norm", "sqrtP_norm"],
    on_inspect=[record_shape, record_dtype],
)
def rls_estimate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """RLS 부목 길이 추정 [L̂_norm, √P_norm] (2차원).

    L̂_norm = (L̂ - prior) / scale,  √P_norm = √P / √P0 (prior 에서 1.0,
    추정이 확실해질수록 0 으로 감소). RLS 모듈이 붙기 전까지는 [0, 1] 상수.
    """
    _ensure_rls_buffers(env)
    l_norm = (env._rls_L_hat - RLS_L_PRIOR) / RLS_L_SCALE
    p_norm = torch.sqrt(torch.clamp(env._rls_P, min=0.0)) / math.sqrt(RLS_P0)
    return torch.stack([l_norm, p_norm], dim=-1)


# =====================================================================
# Privileged 관측 (teacher/critic 전용 — sim GT)
# =====================================================================


@generic_io_descriptor(
    observation_type="PegLegPrivileged",
    units="one_hot",
    element_order=["FL", "FR", "RL", "RR", "injured_flag"],
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """고장 다리 one-hot (FL, FR, RL, RR)와 부상 플래그를 반환합니다.

    tensor([
    [0., 0., 0., 0., 0.],  # env 0: 정상
    [1., 0., 0., 0., 1.],  # env 1: FL
    [0., 1., 0., 0., 1.],  # env 2: FR
    [0., 0., 1., 0., 1.],  # env 3: RL
    [0., 0., 0., 1., 1.],  # env 4: RR
    ])
    """
    one_hot = torch.zeros((env.num_envs, 5), device=env.device)
    if hasattr(env, "_peg_leg_index"):
        idx = env._peg_leg_index.to(dtype=torch.long)
        valid = idx >= 0
        if torch.any(valid):
            one_hot[valid, idx[valid]] = 1.0
            one_hot[valid, 4] = 1.0  # injured flag
    return one_hot  # [FL, FR, RL, RR, injured_flag]


@generic_io_descriptor(
    observation_type="PegLegPrivileged",
    units="dimensionless",
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_foot_friction(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """부상 다리 부목 끝단의 마찰 계수를 반환합니다 (정상 = 0 sentinel)."""
    if hasattr(env, "_peg_leg_foot_friction"):
        return env._peg_leg_foot_friction.unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


@generic_io_descriptor(
    observation_type="PegLegPrivileged",
    units="m",
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_splint_length(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """부목 길이 L (prismatic 관절 지시값, m)를 반환합니다 (정상 = 0 sentinel)."""
    if hasattr(env, "_peg_leg_splint_length"):
        return env._peg_leg_splint_length.unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


@generic_io_descriptor(
    observation_type="JointState",
    units="rad",
    joint_names=CALF_JOINT_NAMES,
    on_inspect=[record_shape, record_dtype],
)
def calf_pos_nominal_rel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """calf 관절각 − 부상 전 nominal (FL, FR, RL, RR 순).

    joint_pos_rel 과 달리 부상 시 재작성되는 default 가 아니라 부상 전
    nominal 기준이므로, 접힌 무릎(fold)각이 관측에서 소거되지 않습니다.
    실기에서도 계산 가능: nominal 은 알려진 상수, 관절각은 인코더 측정치.
    """
    asset = env.scene["robot"]
    joint_names = list(asset.data.joint_names)
    calf_ids = [joint_names.index(n) for n in CALF_JOINT_NAMES if n in joint_names]
    if len(calf_ids) != len(CALF_JOINT_NAMES):
        return torch.zeros((env.num_envs, len(CALF_JOINT_NAMES)), device=env.device)

    nominal = getattr(env, "_peg_leg_default_joint_pos_ref", None)
    if nominal is None:
        # 첫 리셋 이전(= 아직 아무 관절도 lock 되지 않음)에는 현재 default 가 곧 nominal.
        default = asset.data.default_joint_pos
        nominal = default[0] if default.ndim == 2 else default
    nominal = nominal.to(env.device)

    return asset.data.joint_pos[:, calf_ids] - nominal[calf_ids].unsqueeze(0)
