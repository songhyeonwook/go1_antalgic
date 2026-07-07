# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go1 좌우 Mirror Augmentation 유틸리티.

Isaac Lab의 UnitreeGo1RoughEnvCfg 관측 구조에 맞춰,
환경의 절반을 좌우 미러링하여 정책이 좌우 대칭 반응을 학습하게 합니다.

Go1 관절 순서 (12 joints):
  FL_hip(0), FL_thigh(1), FL_calf(2),
  FR_hip(3), FR_thigh(4), FR_calf(5),
  RL_hip(6), RL_thigh(7), RL_calf(8),
  RR_hip(9), RR_thigh(10), RR_calf(11)

미러 변환 (시상면 xz 대칭):
  - 관절: FL ↔ FR, RL ↔ RR; hip abduction은 부호 반전
  - 기저 속도: vy, wx, wz 부호 반전
  - 투영 중력: gy 부호 반전
  - 속도 명령: vy_cmd, wz_cmd 부호 반전
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Go1 Joint Mirroring Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ⚠️ Go1 actual joint order is PER-TYPE (verified from robot.data.joint_names):
#   [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#    FL_calf, FR_calf, RL_calf, RR_calf]  (indices 0..11)
# L-R mirror swaps FL↔FR and RL↔RR within each joint type:
#   hip:   0↔1, 2↔3   thigh: 4↔5, 6↔7   calf: 8↔9, 10↔11
JOINT_MIRROR_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]

# 12-joint sign: hip abduction (indices 0,1,2,3) flips (-1); thigh+calf keep (+1)
JOINT_MIRROR_SIGN = torch.tensor(
    [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)

# 다리 인덱스 미러: FL(0) ↔ FR(1), RL(2) ↔ RR(3), 정상(-1) → -1 유지
LEG_MIRROR_MAP = {-1: -1, 0: 1, 1: 0, 2: 3, 3: 2}


def _get_joint_mirror_sign(device: torch.device) -> torch.Tensor:
    """디바이스에 맞는 미러 부호 텐서를 반환합니다."""
    return JOINT_MIRROR_SIGN.to(device)


def mirror_joint_tensor(x: torch.Tensor) -> torch.Tensor:
    """12차원 관절 텐서를 좌우 미러링합니다.

    Args:
        x: (..., 12) 관절 위치/속도/행동 텐서

    Returns:
        미러링된 텐서 (..., 12)
    """
    sign = _get_joint_mirror_sign(x.device)
    return x[..., JOINT_MIRROR_IDX] * sign


def mirror_obs(obs: torch.Tensor, obs_structure: dict | None = None) -> torch.Tensor:
    """Go1 관측 벡터를 좌우 미러링합니다.

    기본 관측 구조 (UnitreeGo1RoughEnvCfg):
      [0:3]   base_lin_vel   → [vx, -vy, vz]
      [3:6]   base_ang_vel   → [-wx, wy, -wz]
      [6:9]   projected_gravity → [gx, -gy, gz]
      [9:12]  velocity_commands → [vx_cmd, -vy_cmd, -wz_cmd]
      [12:24] joint_pos      → mirror_joint_tensor
      [24:36] joint_vel      → mirror_joint_tensor
      [36:48] actions        → mirror_joint_tensor
      [48:]   height_scan 등 → 별도 처리 필요 시 그대로 유지

    Args:
        obs: (batch, obs_dim) 관측 텐서
        obs_structure: 관측 구조 오버라이드 (기본: 표준 Go1 48차원)

    Returns:
        미러링된 관측 텐서
    """
    m = obs.clone()
    dim = obs.shape[-1]

    # [0:3] base_lin_vel: vy 반전
    if dim > 1:
        m[..., 1] = -m[..., 1]

    # [3:6] base_ang_vel: wx, wz 반전
    if dim > 5:
        m[..., 3] = -m[..., 3]  # wx (roll rate)
        m[..., 5] = -m[..., 5]  # wz (yaw rate)

    # [6:9] projected_gravity: gy 반전
    if dim > 7:
        m[..., 7] = -m[..., 7]

    # [9:12] velocity_commands: vy_cmd, wz_cmd 반전
    if dim > 11:
        m[..., 10] = -m[..., 10]  # vy_cmd
        m[..., 11] = -m[..., 11]  # wz_cmd

    # [12:24] joint_pos: 좌우 swap + hip 부호 반전
    if dim >= 24:
        m[..., 12:24] = mirror_joint_tensor(obs[..., 12:24])

    # [24:36] joint_vel: 좌우 swap + hip 부호 반전
    if dim >= 36:
        m[..., 24:36] = mirror_joint_tensor(obs[..., 24:36])

    # [36:48] actions: 좌우 swap + hip 부호 반전
    if dim >= 48:
        m[..., 36:48] = mirror_joint_tensor(obs[..., 36:48])

    # [48:] height_scan 등은 그대로 유지 (height scan의 좌우 미러링은
    #        scan 포인트 배치에 따라 다르므로 기본적으로 그대로 둠)

    return m


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    """12차원 행동 벡터를 좌우 미러링합니다."""
    return mirror_joint_tensor(action)


def mirror_peg_leg_index(peg_idx: torch.Tensor) -> torch.Tensor:
    """부상 다리 인덱스를 좌우 미러링합니다.

    Args:
        peg_idx: (batch,) 부상 인덱스 (-1=정상, 0=FL, 1=FR, 2=RL, 3=RR)

    Returns:
        미러링된 인덱스 (-1=정상, 0→1, 1→0, 2→3, 3→2)
    """
    result = peg_idx.clone()
    for src, dst in LEG_MIRROR_MAP.items():
        if src != dst:
            result[peg_idx == src] = dst
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Height-scan grid mirror (RayCaster GridPatternCfg)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROPRIO_DIM = 48
# UnitreeGo1RoughEnvCfg height scanner: GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
# → 17 (x) × 11 (y) = 187 rays. L-R mirror = reflect about the sagittal (x) axis,
#   i.e. y → -y. Heights are scalar distances, so this is a pure column PERMUTATION
#   (no sign flip).
HEIGHT_SCAN_NUM_RAYS = 187
_HEIGHT_SCAN_RES = 0.1
_HEIGHT_SCAN_SIZE = (1.6, 1.0)
_HEIGHT_SCAN_ORDERING = "xy"
_HEIGHT_SCAN_PERM_CACHE: dict = {}


def height_scan_mirror_perm(device: torch.device) -> torch.Tensor:
    """Permutation mapping each height-scan ray to its left/right (y→-y) mirror.

    Reconstructs the exact grid that ``patterns.grid_pattern`` builds so the
    ordering matches the live sensor regardless of the meshgrid convention.
    """
    key = str(device)
    if key not in _HEIGHT_SCAN_PERM_CACHE:
        sx, sy = _HEIGHT_SCAN_SIZE
        x = torch.arange(-sx / 2, sx / 2 + 1.0e-9, _HEIGHT_SCAN_RES)
        y = torch.arange(-sy / 2, sy / 2 + 1.0e-9, _HEIGHT_SCAN_RES)
        indexing = "xy" if _HEIGHT_SCAN_ORDERING == "xy" else "ij"
        gx, gy = torch.meshgrid(x, y, indexing=indexing)
        xs, ys = gx.flatten(), gy.flatten()
        n = xs.numel()
        coord = {(round(xs[i].item(), 4), round(ys[i].item(), 4)): i for i in range(n)}
        perm = [coord[(round(xs[i].item(), 4), round(-ys[i].item(), 4))] for i in range(n)]
        _HEIGHT_SCAN_PERM_CACHE[key] = torch.tensor(perm, dtype=torch.long, device=device)
    return _HEIGHT_SCAN_PERM_CACHE[key]


def mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the full policy observation: proprioception [0:48] + height scan.

    Proprioception is handled by :func:`mirror_obs`. If a 187-ray height scan is
    present (obs dim ≥ 235) its block is reflected about the sagittal axis with
    :func:`height_scan_mirror_perm`. Any trailing dims are left untouched.
    """
    m = mirror_obs(obs)  # mirrors [0:48], copies the remainder verbatim
    dim = obs.shape[-1]
    lo, hi = PROPRIO_DIM, PROPRIO_DIM + HEIGHT_SCAN_NUM_RAYS
    if dim >= hi:
        perm = height_scan_mirror_perm(obs.device)
        m[..., lo:hi] = obs[..., lo:hi].index_select(-1, perm)
    return m


def mirror_privileged_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the teacher privileged obs [injury_index, splint_length, friction].

    injury_index obs values are 0=normal, 1=FL, 2=FR, 3=RL, 4=RR (peg_leg_index
    returns internal_idx+1). L-R mirror swaps FL↔FR (1↔2) and RL↔RR (3↔4);
    splint length and friction are mirror-invariant.
    """
    m = obs.clone()
    if obs.shape[-1] >= 1:
        idx = obs[..., 0]
        new = idx.clone()
        new = torch.where(idx == 1, torch.full_like(idx, 2.0), new)
        new = torch.where(idx == 2, torch.full_like(idx, 1.0), new)
        new = torch.where(idx == 3, torch.full_like(idx, 4.0), new)
        new = torch.where(idx == 4, torch.full_like(idx, 3.0), new)
        m[..., 0] = new
    return m


def mirror_full_obs(obs):
    """Mirror a full policy-input observation for left/right canonicalization.

    Handles the actor input as either a flat tensor or a dict / TensorDict.
    Flat layout = proprioception (48) [+ height_scan (187)] [+ privileged (3)].
    The privileged tail (last 3 dims: injury_index, splint, friction) is detected
    by total dim and mirrored via :func:`mirror_privileged_obs`; the leading
    proprio (+height) part via :func:`mirror_policy_obs`. Used to fold the
    bilaterally-symmetric env along its sagittal axis (FL↔FR AND RL↔RR), giving
    EXACT left/right consistency for deployment regardless of learned equivariance.
    """
    # dict / TensorDict with named groups
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
    # flat tensor
    dim = obs.shape[-1]
    priv = dim in (PROPRIO_DIM + 3, PROPRIO_DIM + HEIGHT_SCAN_NUM_RAYS + 3)  # 51 or 238
    body = obs[..., :-3] if priv else obs
    m = mirror_policy_obs(body)
    if priv:
        m = torch.cat([m, mirror_privileged_obs(obs[..., -3:])], dim=-1)
    return m


def mirror_sigma(x: torch.Tensor) -> torch.Tensor:
    """Mirror a per-joint std vector: left/right swap only (no sign flip)."""
    return x[..., JOINT_MIRROR_IDX]


@torch.no_grad()
def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """RSL-RL symmetry augmentation callback for Go1 left/right mirroring.

    Used by FEEDFORWARD policies (e.g. the RMA MLP teacher) via
    ``RslRlSymmetryCfg(use_data_augmentation=True)``. Each observation group is
    mirrored with the appropriate transform:
      - "policy"          → proprioception (48) + height-scan grid (187)
      - "privileged_obs"  → injury index FL↔FR / RL↔RR (splint, friction kept)
    Actions get the 12-joint L/R swap + hip-abduction sign flip. The reward is
    never touched — this enforces left/right equivariance structurally.

    (Recurrent policies cannot use this path — masks/hidden are not doubled; use
    ``SymmetricPPO`` storage-level doubling instead.)
    """
    _ = env

    if obs is not None:
        batch_size = obs.batch_size[0]
        repeat_dims = [2] + [1] * (obs.ndim - 1)
        obs_aug = obs.repeat(*repeat_dims)
        for key in obs.keys():
            if key == "policy":
                mirrored = mirror_policy_obs(obs["policy"])
            elif key in ("privileged_obs", "privileged"):
                mirrored = mirror_privileged_obs(obs[key])
            else:
                mirrored = obs[key].clone()
            obs_aug[key][:batch_size] = obs[key][:]
            obs_aug[key][batch_size:] = mirrored
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(
            (batch_size * 2, *actions.shape[1:]), device=actions.device, dtype=actions.dtype
        )
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = mirror_action(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
