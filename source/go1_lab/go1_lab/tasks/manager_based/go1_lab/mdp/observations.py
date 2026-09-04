from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_shape,
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



# =====================================================================
# Privileged 관측 (teacher/critic 전용 — sim GT)
# =====================================================================


@generic_io_descriptor(
    observation_type="PegLeg",
    units="one_hot",
    element_order=["FL", "FR", "RL", "RR"],
    on_inspect=[record_shape, record_dtype],
)
def peg_leg_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """부상 다리 one-hot (FL, FR, RL, RR). 정상은 [0, 0, 0, 0].

    tensor([
    [0., 0., 0., 0.],  # env 0: 정상
    [1., 0., 0., 0.],  # env 1: FL
    [0., 1., 0., 0.],  # env 2: FR
    [0., 0., 1., 0.],  # env 3: RL
    [0., 0., 0., 1.],  # env 4: RR
    ])

    주의: @generic_io_descriptor 는 description 이 없으면 func.__doc__ 을 읽으므로
    (IsaacLab io_descriptors.py:236) 이 docstring 을 주석으로 바꾸면 import 가
    AttributeError 로 실패한다. 반드시 docstring 형태로 유지할 것.
    """
    one_hot = torch.zeros((env.num_envs, 4), device=env.device)
    if hasattr(env, "_peg_leg_index"):
        idx = env._peg_leg_index.to(dtype=torch.long)
        valid = idx >= 0
        if torch.any(valid):
            one_hot[valid, idx[valid]] = 1.0
    return one_hot  # [FL, FR, RL, RR], 정상 = [0, 0, 0, 0]


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

