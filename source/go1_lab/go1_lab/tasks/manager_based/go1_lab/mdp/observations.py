from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_leg_index(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """부상 상태 인덱스: 0=정상, 1=FL, 2=FR, 3=RL, 4=RR."""
    if os.getenv("GO1_HIDE_PRIVILEGED_INJURY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.zeros((env.num_envs, 1), device=env.device)
    if hasattr(env, "_peg_leg_index"):
        return (env._peg_leg_index.float() + 1.0).unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


def peg_leg_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """고장 다리 one-hot (FL, FR, RL, RR)와 부상 플래그를 반환합니다."""
    one_hot = torch.zeros((env.num_envs, 5), device=env.device)
    if hasattr(env, "_peg_leg_index"):
        idx = env._peg_leg_index.to(dtype=torch.long)
        valid = idx >= 0
        if torch.any(valid):
            one_hot[valid, idx[valid]] = 1.0
            one_hot[valid, 4] = 1.0  # injured flag
    return one_hot


def peg_leg_calf_lock_angle(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """고장 다리 calf 고정 각도(rad)를 반환합니다."""
    if hasattr(env, "_peg_leg_calf_lock_angle"):
        return env._peg_leg_calf_lock_angle.unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


def peg_leg_foot_friction(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """고장 다리 발 마찰 계수를 반환합니다."""
    if os.getenv("GO1_HIDE_PRIVILEGED_INJURY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.ones((env.num_envs, 1), device=env.device)
    if hasattr(env, "_peg_leg_foot_friction"):
        return env._peg_leg_foot_friction.unsqueeze(-1)
    return torch.ones((env.num_envs, 1), device=env.device)


def peg_leg_splint_length(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """부목 등가 길이(m)를 반환합니다. Go1 링크 기구학 기반."""
    if os.getenv("GO1_HIDE_PRIVILEGED_INJURY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.zeros((env.num_envs, 1), device=env.device)
    if hasattr(env, "_peg_leg_splint_length"):
        return env._peg_leg_splint_length.unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)
