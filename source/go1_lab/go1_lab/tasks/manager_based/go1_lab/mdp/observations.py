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
# 을 통해서만 관측된다. 추정기 본체(FK + 스칼라 RLS + 게이트)는 mdp/rls.py.
# 이 항은 (L̂, √P) 버퍼를 정규화해 정책 입력으로 노출하기만 한다.

from .rls import RLS_L_PRIOR, RLS_L_SCALE, RLS_P0, ensure_rls_buffers  # noqa: E402


@generic_io_descriptor(
    observation_type="SplintEstimate",
    units="normalized",
    element_order=["L_hat_norm", "sqrtP_norm"],
    on_inspect=[record_shape, record_dtype],
)
def rls_estimate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """RLS 부목 길이 추정 [L̂_norm, √P_norm] (2차원).

    L̂_norm = (L̂ - prior) / scale,  √P_norm = √P / √P0 (prior 에서 1.0,
    추정이 확실해질수록 0 으로 감소). rls_params 미설정 시 prior 상수.
    """
    ensure_rls_buffers(env)
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
