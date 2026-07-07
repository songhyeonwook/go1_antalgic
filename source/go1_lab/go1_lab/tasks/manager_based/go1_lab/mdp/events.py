# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""이벤트 함수들 - 의족(Peg Leg) 시나리오를 위한 랜덤화 함수들.

⚠️ 핵심 설계 원칙 (ActuatorNetMLP 호환):
  Go1 로봇은 Explicit Actuator(ActuatorNetMLP)를 사용합니다.
  이 모델에서는 PhysX 내부 PD 제어기가 비활성화되어 있으므로,
  robot.data.joint_stiffness에 값을 쓰는 것은 아무런 물리적 효과가 없습니다.

  관절을 "고정"하려면:
    (1) default_joint_pos를 lock angle로 설정 (action=0일 때 target=lock_angle이 되도록)
    (2) 매 스텝 action masking: 부상 calf joint의 action을 0으로 강제
  이 두 가지를 함께 해야 합니다.
"""

from __future__ import annotations

import math
import os
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import events as mdp_events
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CALF_JOINT_NAMES = ["FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
# Proximal "hip" actuated DOFs per leg (Go1 URDF naming):
#   *_hip_joint   = hip abduction/adduction (roll)
#   *_thigh_joint = hip flexion/extension   (pitch)
HIP_JOINT_NAMES = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
THIGH_JOINT_NAMES = ["FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"]

# Go1 링크 길이 (URDF 기준, m)
GO1_THIGH_LENGTH = 0.213
GO1_CALF_LENGTH = 0.213
GO1_MAX_SPLINT_LENGTH = GO1_THIGH_LENGTH + GO1_CALF_LENGTH  # 0.426m (완전 신전)
GO1_MIN_SPLINT_LENGTH = 0.08  # 기구학적 하한


# =====================================================================
# 부상 다리 hip-joint 토크 제한 (논문 §4.2)
# =====================================================================
#   "additionally reducing the affected hip-joint torque limits to 5% of
#    nominal to mimic peri-articular damage."
#
# Go1은 ActuatorNetMLP(explicit actuator)를 쓰며, 매 substep compute()에서
#   applied_effort = clip(network_torque, -effort_limit, +effort_limit)
# 로 토크를 클리핑합니다. 따라서 부상 다리 hip joint의 `effort_limit`을
# nominal(23.7N·m)의 5%로 낮추면 actuator가 자동으로 매 스텝 토크를 5%로
# 제한합니다 — 논문의 "torque limit 5%"와 정확히 일치하는 구현입니다.
#
# 환경변수:
#   GO1_PEG_HIP_TORQUE_SCALE   : nominal 대비 비율 (기본 0.05 = 5%)
#   GO1_PEG_WEAKEN_JOINTS      : 약화할 proximal 관절 (기본 "hip")
#       "hip"        → *_hip_joint (abduction, 논문 "hip-joint" 직역)
#       "thigh"      → *_thigh_joint (hip flexion)
#       "hip,thigh"  → 둘 다 (anatomical hip 전체)
def _peg_hip_torque_scale() -> float:
    try:
        return float(os.getenv("GO1_PEG_HIP_TORQUE_SCALE", "0.05"))
    except ValueError:
        return 0.05


def _peg_weaken_joint_names(leg_idx: int) -> list[str]:
    which = os.getenv("GO1_PEG_WEAKEN_JOINTS", "hip").strip().lower()
    names: list[str] = []
    if "hip" in which:
        names.append(HIP_JOINT_NAMES[leg_idx])
    if "thigh" in which:
        names.append(THIGH_JOINT_NAMES[leg_idx])
    return names


def _ensure_hip_effort_cache(env: "ManagerBasedRLEnv", robot: Articulation) -> None:
    """actuator별 nominal effort_limit 스냅샷과 leg→(actuator, col) 매핑을 1회 캐싱."""
    if getattr(env, "_peg_hip_effort_ready", False):
        return
    cols: dict[int, list[tuple[object, int, str]]] = {0: [], 1: [], 2: [], 3: []}
    for actuator in getattr(robot, "actuators", {}).values():
        effort_limit = getattr(actuator, "effort_limit", None)
        if not torch.is_tensor(effort_limit):
            continue
        # nominal 스냅샷 (최초 1회)
        if not hasattr(actuator, "_peg_nominal_effort_limit"):
            actuator._peg_nominal_effort_limit = effort_limit.clone()
        names = list(getattr(actuator, "joint_names", []))
        for leg_idx in range(4):
            for jname in (HIP_JOINT_NAMES[leg_idx], THIGH_JOINT_NAMES[leg_idx]):
                if jname in names:
                    cols[leg_idx].append((actuator, names.index(jname), jname))
    env._peg_hip_actuator_cols = cols
    env._peg_hip_effort_ready = True


def apply_peg_leg_hip_torque_limit(
    env: "ManagerBasedRLEnv",
    robot: Articulation,
    env_ids_t: torch.Tensor,
    sampled_leg_idx: torch.Tensor,
) -> None:
    """리셋 시 부상 다리 hip-joint effort_limit을 nominal의 일정 비율(기본 5%)로 낮춘다.

    healthy(leg_idx<0) 환경은 nominal로 복구한다. effort_limit는 actuator 버퍼이므로
    다음 리셋까지 유지되며, ActuatorNetMLP가 매 substep 이 값으로 토크를 클리핑한다.
    """
    scale = _peg_hip_torque_scale()
    _ensure_hip_effort_cache(env, robot)
    if not getattr(env, "_peg_hip_effort_ready", False):
        return

    # (1) 리셋 대상 환경의 effort_limit을 전부 nominal로 복구
    for actuator in getattr(robot, "actuators", {}).values():
        nominal = getattr(actuator, "_peg_nominal_effort_limit", None)
        if nominal is not None and torch.is_tensor(actuator.effort_limit):
            actuator.effort_limit[env_ids_t] = nominal[env_ids_t]

    if scale >= 1.0:
        return

    which = os.getenv("GO1_PEG_WEAKEN_JOINTS", "hip").strip().lower()
    want_hip = "hip" in which
    want_thigh = "thigh" in which

    # (2) 부상 다리의 hip/thigh effort_limit을 scale배로 약화 (leg별 벡터화)
    for leg_idx in range(4):
        mask = sampled_leg_idx == leg_idx
        if not mask.any():
            continue
        sel = env_ids_t[mask]
        for actuator, col, jname in env._peg_hip_actuator_cols.get(leg_idx, []):
            is_hip = jname.endswith("_hip_joint")
            is_thigh = jname.endswith("_thigh_joint")
            if (is_hip and want_hip) or (is_thigh and want_thigh):
                nominal = actuator._peg_nominal_effort_limit
                actuator.effort_limit[sel, col] = nominal[sel, col] * scale


def apply_peg_leg_calf_stiffness(
    env: "ManagerBasedRLEnv",
    robot: Articulation,
    env_ids_t: torch.Tensor,
    sampled_leg_idx: torch.Tensor,
) -> None:
    """Make the injured (splinted) calf a COMPLIANT spring at reset: set its PD
    stiffness/damping to GO1_SPLINT_CALF_STIFFNESS (default unset = nominal, rigid
    hold). A real splint flexes slightly under load; a FINITE-stiffness knee
    ABSORBS the loading impact, so the splinted leg can bear partial load without
    the trunk collapsing (the rigid-knee → sink failure). Healthy legs keep nominal
    stiffness. Per-leg, joint-name based (Go1 joint order is per-TYPE, not per-leg).
    """
    _kp = os.getenv("GO1_SPLINT_CALF_STIFFNESS", "").strip()
    if not _kp:
        return
    kp = float(_kp)
    kd = float(os.getenv("GO1_SPLINT_CALF_DAMPING", "0.5"))
    for actuator in getattr(robot, "actuators", {}).values():
        st = getattr(actuator, "stiffness", None)
        dm = getattr(actuator, "damping", None)
        if not (torch.is_tensor(st) and st.ndim == 2):
            continue
        if not hasattr(actuator, "_peg_nominal_calf_stiffness"):
            actuator._peg_nominal_calf_stiffness = st.clone()
            actuator._peg_nominal_calf_damping = dm.clone() if torch.is_tensor(dm) else None
        # restore reset envs to nominal first (the previous episode's injured calf)
        st[env_ids_t] = actuator._peg_nominal_calf_stiffness[env_ids_t]
        if torch.is_tensor(dm) and actuator._peg_nominal_calf_damping is not None:
            dm[env_ids_t] = actuator._peg_nominal_calf_damping[env_ids_t]
        # soft spring on the injured calf
        for leg_idx in range(4):
            mask = sampled_leg_idx == leg_idx
            if not mask.any():
                continue
            try:
                calf_idx = robot.data.joint_names.index(CALF_JOINT_NAMES[leg_idx])
            except ValueError:
                continue
            if calf_idx < st.shape[1]:
                sel = env_ids_t[mask]
                st[sel, calf_idx] = kp
                if torch.is_tensor(dm):
                    dm[sel, calf_idx] = kd


def splint_length_to_calf_angle(splint_length: torch.Tensor) -> torch.Tensor:
    """부목 등가 길이(m) → Go1 calf joint 고정 각도(rad, 음수).

    역기구학: cos(θ) = (L² - L_thigh² - L_calf²) / (2 · L_thigh · L_calf)
    Go1 관례상 calf 각도는 음수(굽힘 방향).
    """
    l_t, l_c = GO1_THIGH_LENGTH, GO1_CALF_LENGTH
    cos_val = (splint_length**2 - l_t**2 - l_c**2) / (2.0 * l_t * l_c)
    cos_val = torch.clamp(cos_val, -1.0, 1.0)
    return -torch.acos(cos_val)


def calf_angle_to_splint_length(calf_angle: torch.Tensor) -> torch.Tensor:
    """Go1 calf joint 각도(rad, 음수) → 부목 등가 길이(m).

    정기구학: L² = L_thigh² + L_calf² + 2 · L_thigh · L_calf · cos(θ)
    """
    l_t, l_c = GO1_THIGH_LENGTH, GO1_CALF_LENGTH
    l_sq = l_t**2 + l_c**2 + 2.0 * l_t * l_c * torch.cos(calf_angle)
    return torch.sqrt(torch.clamp(l_sq, min=1e-6))


def _resolve_env_ids(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _ensure_peg_leg_buffers(env: "ManagerBasedRLEnv") -> None:
    """환경 객체에 의족 메타데이터 버퍼를 생성합니다."""
    if not hasattr(env, "_peg_leg_index"):
        env._peg_leg_index = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    if not hasattr(env, "_peg_leg_calf_joint_index"):
        env._peg_leg_calf_joint_index = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    if not hasattr(env, "_peg_leg_calf_lock_angle"):
        env._peg_leg_calf_lock_angle = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_peg_leg_splint_length"):
        env._peg_leg_splint_length = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_peg_leg_foot_friction"):
        env._peg_leg_foot_friction = torch.ones(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_peg_leg_default_joint_pos_ref"):
        env._peg_leg_default_joint_pos_ref = None


# 내부 인덱스: 0=FL, 1=FR, 2=RL, 3=RR  (privileged obs에서는 +1 하여 1-4로 노출)
_TARGET_LEG_MAP: dict[str, int] = {"fl": 0, "fr": 1, "rl": 2, "rr": 3}


def _sample_peg_leg_indices(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    prob_peg_leg: float,
    target_leg: str = "random",
) -> torch.Tensor:
    """각 환경마다 고장 다리 인덱스를 샘플링합니다. 정상은 -1.

    Args:
        prob_peg_leg: 부상 발생 확률 (0.0 ~ 1.0).
        target_leg: "random" → reset마다 부상 다리를 stratified-random 균등 샘플,
                    "fl"/"fr"/"rl"/"rr" → 해당 다리 고정,
                    "normal" → 전부 정상(-1),
                    "balanced"/"balanced_random" → Normal/FL/FR/RL/RR 균등 샘플.
    """
    n = env_ids.numel()
    mode = str(target_leg).strip().lower()

    if mode == "normal" or prob_peg_leg <= 0.0:
        return torch.full((n,), -1, device=env.device, dtype=torch.long)

    # 특정 다리 고정 모드 (평가용)
    if mode in _TARGET_LEG_MAP:
        fixed_idx = _TARGET_LEG_MAP[mode]
        peg_indices = torch.full((n,), fixed_idx, device=env.device, dtype=torch.long)
        if prob_peg_leg >= 1.0:
            return peg_indices
        active = torch.rand((n,), device=env.device) < float(prob_peg_leg)
        return torch.where(active, peg_indices, torch.full_like(peg_indices, -1))

    # Balanced 모드: 1:1:1:1:1 (Normal, FL, FR, RL, RR) 균등 배정.
    # 기존 env-id 고정 round-robin은 특정 leg condition이 특정 terrain/env shard와
    # 반복적으로 묶일 수 있으므로, 기본은 무작위 permutation으로 둡니다.
    if mode in {"balanced", "balanced_random"}:
        repeats = int(math.ceil(n / 5))
        indices = torch.arange(5, device=env.device, dtype=torch.long).repeat(repeats)[:n]
        perm = torch.randperm(n, device=env.device)
        indices = indices[perm]
        # 0=FL, 1=FR, 2=RL, 3=RR, 4=Normal(-1)
        return torch.where(indices == 4, torch.full_like(indices, -1), indices)

    # 라운드 로빈 모드: FL→FR→RL→RR 순환으로 정확히 균등 배정
    # 재현성/디버그용으로만 사용합니다.
    if mode == "round_robin":
        if not hasattr(env, "_peg_leg_rr_counter"):
            env._peg_leg_rr_counter = 0
        peg_indices = torch.arange(n, device=env.device, dtype=torch.long)
        peg_indices = (peg_indices + env._peg_leg_rr_counter) % 4
        env._peg_leg_rr_counter = (env._peg_leg_rr_counter + n) % 4
        if prob_peg_leg >= 1.0:
            return peg_indices
        active = torch.rand((n,), device=env.device) < float(prob_peg_leg)
        return torch.where(active, peg_indices, torch.full_like(peg_indices, -1))

    # Random 모드: 부상 여부는 확률로 정하되, 부상 다리 라벨은 FL/FR/RL/RR가
    # 거의 정확히 균등하도록 무작위 permutation으로 배정합니다.
    # - iid random: 짧은 학습 구간에서 다리별 표본 수가 흔들릴 수 있음
    # - env-id round-robin: 특정 leg condition이 특정 env shard와 묶일 수 있음
    # - stratified random: 표본 수 균형과 env-id 비고정을 동시에 만족
    if mode != "random":
        mode = "random"
    active = torch.rand((n,), device=env.device) < float(prob_peg_leg)
    if prob_peg_leg >= 1.0:
        active = torch.ones((n,), device=env.device, dtype=torch.bool)

    result = torch.full((n,), -1, device=env.device, dtype=torch.long)
    num_active = int(active.sum().item())
    if num_active == 0:
        return result

    repeats = int(math.ceil(num_active / 4))
    leg_labels = torch.arange(4, device=env.device, dtype=torch.long).repeat(repeats)[:num_active]
    leg_labels = leg_labels[torch.randperm(num_active, device=env.device)]
    result[torch.where(active)[0]] = leg_labels
    return result


def _sample_calf_lock_angles(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, angle_range: tuple[float, float]
) -> torch.Tensor:
    lo, hi = float(angle_range[0]), float(angle_range[1])
    return lo + (hi - lo) * torch.rand((env_ids.numel(),), device=env.device)


def _sample_foot_friction(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, friction_range: tuple[float, float]
) -> torch.Tensor:
    lo, hi = float(friction_range[0]), float(friction_range[1])
    return lo + (hi - lo) * torch.rand((env_ids.numel(),), device=env.device)


def _get_peg_leg_per_env(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None
) -> dict[int, int | None]:
    """환경별 고장 다리 매핑(dict[env_id, leg_idx|None])을 반환합니다."""
    env_ids_t = _resolve_env_ids(env, env_ids)
    result: dict[int, int | None] = {}

    if hasattr(env, "_peg_leg_index"):
        peg_indices = env._peg_leg_index[env_ids_t].detach().cpu().tolist()
        for env_id, peg_idx in zip(
            env_ids_t.detach().cpu().tolist(), peg_indices, strict=False
        ):
            result[env_id] = None if peg_idx < 0 else int(peg_idx)
        return result

    # 초기 호환용 fallback: 기존 group 규칙
    for env_id in env_ids_t.detach().cpu().tolist():
        group = env_id % 5
        result[env_id] = None if group == 0 else group - 1
    return result


def randomize_peg_leg_geometry(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = (0.5, 1.5),
    prob_peg_leg: float = 0.2,
):
    """[비활성화] 기하학적 랜덤화(다리 길이)는 현재 실험에서 사용하지 않습니다."""
    _ = (env, env_ids, asset_cfg, scale_range, prob_peg_leg)
    return


def randomize_peg_leg_dynamics(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (0.1, 2.0),
    prob_peg_leg: float = 0.2,
):
    """[비활성화] 동역학 랜덤화(질량 변화)는 현재 실험에서 사용하지 않습니다."""
    _ = (env, env_ids, asset_cfg, mass_range, prob_peg_leg)
    return


def _sample_splint_lengths(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, length_range: tuple[float, float]
) -> torch.Tensor:
    lo = max(float(length_range[0]), GO1_MIN_SPLINT_LENGTH)
    hi = min(float(length_range[1]), GO1_MAX_SPLINT_LENGTH)
    return lo + (hi - lo) * torch.rand((env_ids.numel(),), device=env.device)


def randomize_peg_leg_actuation(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    prob_peg_leg: float = 1.0,
    prob_joint_disabled: float = 1.0,
    actuation_mode: str = "locked",
    locked_joint_angle_range: tuple[float, float] = (-1.5, -0.8),
    splint_length_range: tuple[float, float] | None = None,
    foot_friction_range: tuple[float, float] = (0.2, 1.2),
    target_leg: str = "random",
):
    """의족 시나리오를 위한 리셋 이벤트 (ActuatorNetMLP 호환).

    ⚠️ Go1은 ActuatorNetMLP(Explicit Actuator)를 사용합니다.
    joint_stiffness/damping에 값을 써도 물리적 효과가 없습니다!

    대신 이 함수는:
      (1) 부상 다리/부목 길이를 샘플링하여 메타데이터 버퍼에 저장
      (2) default_joint_pos를 lock angle로 설정 (action=0 → target=lock_angle)
      (3) joint_pos, joint_vel을 lock angle/0으로 초기화

    매 스텝 action masking은 Go1LabEnv.step()에서 수행합니다.

    ⭐ 커리큘럼 지원:
      env._curriculum_prob_peg_leg (float) — 커리큘럼이 설정한 부상 확률 (우선 적용)
      env._curriculum_splint_range (tuple) — 커리큘럼이 설정한 부목 길이 범위 (우선 적용)
    """
    robot: Articulation = env.scene[asset_cfg.name]
    env_ids_t = _resolve_env_ids(env, env_ids)
    _ensure_peg_leg_buffers(env)

    # ⭐ 커리큘럼 파라미터 우선 적용
    cur_prob = getattr(env, "_curriculum_prob_peg_leg", None)
    cur_splint = getattr(env, "_curriculum_splint_range", None)
    effective_prob = float(cur_prob) if cur_prob is not None else prob_peg_leg
    effective_splint = cur_splint if cur_splint is not None else splint_length_range

    sampled_leg_idx = _sample_peg_leg_indices(
        env,
        env_ids_t,
        prob_peg_leg=effective_prob,
        target_leg=target_leg,
    )
    sampled_foot_friction = _sample_foot_friction(
        env, env_ids_t, friction_range=foot_friction_range
    )

    if effective_splint is not None:
        sampled_lengths = _sample_splint_lengths(
            env, env_ids_t, length_range=effective_splint
        )
        sampled_lock_angles = splint_length_to_calf_angle(sampled_lengths)
    else:
        sampled_lock_angles = _sample_calf_lock_angles(
            env, env_ids_t, angle_range=locked_joint_angle_range
        )
        sampled_lengths = calf_angle_to_splint_length(sampled_lock_angles)

    # Functional-splint override: lock the calf at a FIXED angle (e.g. -1.5 = the
    # normal Go1 stance) instead of a SHORTENED peg. Keeps the leg at near-normal
    # length with the foot at the normal ground position, so the rigid (stiff-knee)
    # leg can be LOADED like a real splint that immobilises the joint at a
    # functional position (not a short dangling peg). GO1_SPLINT_CALF_ANGLE=-1.5.
    _fixed_calf = os.getenv("GO1_SPLINT_CALF_ANGLE", "").strip()
    if _fixed_calf:
        sampled_lock_angles = torch.full_like(sampled_lock_angles, float(_fixed_calf))
        sampled_lengths = calf_angle_to_splint_length(sampled_lock_angles)

    env._peg_leg_index[env_ids_t] = sampled_leg_idx
    env._peg_leg_calf_lock_angle[env_ids_t] = sampled_lock_angles
    env._peg_leg_splint_length[env_ids_t] = sampled_lengths
    env._peg_leg_foot_friction[env_ids_t] = sampled_foot_friction

    # ━━━ 논문 §4.2: 부상 다리 hip-joint 토크를 nominal의 5%로 약화 ━━━
    apply_peg_leg_hip_torque_limit(env, robot, env_ids_t, sampled_leg_idx)
    # 부목 무릎을 compliant spring으로 (GO1_SPLINT_CALF_STIFFNESS) — 충격 흡수 → 적재 가능
    apply_peg_leg_calf_stiffness(env, robot, env_ids_t, sampled_leg_idx)

    num_joints = len(robot.data.joint_names)

    # default_joint_pos의 원본 저장 (최초 1회)
    if hasattr(robot.data, "default_joint_pos"):
        if env._peg_leg_default_joint_pos_ref is None:
            ref = robot.data.default_joint_pos
            env._peg_leg_default_joint_pos_ref = (
                ref[0].clone() if ref.ndim == 2 else ref.clone()
            )

    # ━━━ 리셋 대상 환경의 default_joint_pos를 원본으로 복구 ━━━
    # 이전 에피소드에서 부상 다리였던 관절의 default_pos가 lock angle로 바뀌어 있으므로,
    # 새 에피소드를 위해 먼저 전체를 원본으로 복구합니다.
    if env._peg_leg_default_joint_pos_ref is not None and hasattr(
        robot.data, "default_joint_pos"
    ):
        if robot.data.default_joint_pos.ndim == 2:
            robot.data.default_joint_pos[env_ids_t] = (
                env._peg_leg_default_joint_pos_ref.unsqueeze(0).expand(
                    env_ids_t.numel(), -1
                )
            )
        # 1D fallback은 일반적으로 Go1에서 사용하지 않지만 안전을 위해 유지

    # ━━━ calf joint 인덱스 매핑 (벡터화) ━━━
    # Go1 관절 순서: [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, ...]
    # calf joint index = leg_idx * 3 + 2
    env._peg_leg_calf_joint_index[env_ids_t] = -1

    for local_i, env_id_t in enumerate(env_ids_t):
        env_id = int(env_id_t.item())
        leg_idx = int(sampled_leg_idx[local_i].item())
        if leg_idx < 0:
            continue

        joint_name = CALF_JOINT_NAMES[leg_idx]
        try:
            joint_idx = robot.data.joint_names.index(joint_name)
        except ValueError:
            continue
        env._peg_leg_calf_joint_index[env_id] = joint_idx

        if float(torch.rand((), device=env.device).item()) > float(prob_joint_disabled):
            continue

        target_lock_angle = float(sampled_lock_angles[local_i].item())

        # (1) default_joint_pos를 lock angle로 설정
        #     → action=0일 때: target = default_pos + 0*scale = lock_angle
        #     → ActuatorNetMLP가 target ≈ current이면 토크 ≈ 0 → 관절 고정
        if (
            hasattr(robot.data, "default_joint_pos")
            and robot.data.default_joint_pos.ndim >= 2
        ):
            robot.data.default_joint_pos[env_id, joint_idx] = target_lock_angle

        # (2) 현재 joint position/velocity도 lock angle으로 초기화
        if hasattr(robot.data, "joint_pos") and robot.data.joint_pos.ndim >= 2:
            robot.data.joint_pos[env_id, joint_idx] = target_lock_angle
        if hasattr(robot.data, "joint_vel") and robot.data.joint_vel.ndim >= 2:
            robot.data.joint_vel[env_id, joint_idx] = 0.0
        if (
            hasattr(robot.data, "joint_pos_target")
            and robot.data.joint_pos_target.ndim >= 2
        ):
            robot.data.joint_pos_target[env_id, joint_idx] = target_lock_angle

    # 마찰 랜덤화는 API 차이로 실패할 수 있으므로 try/except로 안전 처리
    for local_i, env_id_t in enumerate(env_ids_t):
        env_id = int(env_id_t.item())
        leg_idx = int(sampled_leg_idx[local_i].item())
        if leg_idx < 0:
            continue
        try:
            # By default the friction is applied to the whole robot (all feet).
            # GO1_INJURED_FOOT_FRICTION_ONLY=1 targets ONLY the injured foot, so a
            # low value models a slippery splint sole / dragging toe (knuckling)
            # that SKIMS without catching, while the healthy feet keep grip and can
            # still propel. A global low value instead makes the whole robot
            # slippery -> it can only stand (zero task), an artifact.
            _fric_cfg = SceneEntityCfg(asset_cfg.name)
            if os.getenv("GO1_INJURED_FOOT_FRICTION_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}:
                _fric_cfg = SceneEntityCfg(asset_cfg.name, body_names=[f"{FOOT_BODY_NAMES[leg_idx]}.*"])
            mdp_events.randomize_rigid_body_material(
                env=env,
                env_ids=torch.tensor([env_id], device=env.device, dtype=torch.long),
                asset_cfg=_fric_cfg,
                static_friction_range=(
                    float(sampled_foot_friction[local_i].item()),
                    float(sampled_foot_friction[local_i].item()),
                ),
                dynamic_friction_range=(
                    float(sampled_foot_friction[local_i].item()),
                    float(sampled_foot_friction[local_i].item()),
                ),
                restitution_range=(0.0, 0.0),
            )
        except Exception:
            # 백엔드/버전별 시그니처 차이를 허용하고, privileged obs 용 샘플값만 유지
            pass


def enforce_peg_leg_constraints(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
):
    """매 스텝 호출: 부상 다리의 action을 0으로 마스킹하고 joint_pos를 강제 고정합니다.

    ⚠️ 이 함수는 mode="interval"(interval_range_s 최소)로 등록되어 매 환경 스텝마다 호출됩니다.

    Isaac Lab의 step 흐름:
      1. action_manager.process_action(action)  ← action이 버퍼에 저장됨
      2. physics loop:
         a. action_manager.apply_action()  ← target = default_pos + action * scale
         b. scene.write_data_to_sim()      ← PhysX에 기록
         c. sim.step()                     ← 물리 시뮬레이션
         d. scene.update()                 ← 센서 업데이트
      3. reward/termination 계산
      4. event_manager.apply(mode="interval")  ← 여기서 호출됨

    interval 이벤트는 physics loop 이후에 실행되므로,
    action masking만으로는 한 스텝 지연이 발생합니다.
    따라서 joint_pos 직접 강제도 함께 수행하여 다음 스텝 시작 시 관절이 올바른 위치에 있도록 합니다.

    추가로, action_manager의 내부 action 버퍼를 직접 수정하여
    다음 physics loop의 apply_action()에서 target이 lock_angle이 되도록 합니다.
    """
    if not hasattr(env, "_peg_leg_index"):
        return

    robot: Articulation = env.scene[asset_cfg.name]
    peg_leg_idx = env._peg_leg_index  # (num_envs,) -1=정상, 0~3=부상 다리
    lock_angles = env._peg_leg_calf_lock_angle  # (num_envs,)
    calf_joint_indices = env._peg_leg_calf_joint_index  # (num_envs,) -1=정상

    # 부상 환경만 선택
    is_injured = peg_leg_idx >= 0
    if not is_injured.any():
        return

    injured_env_ids = torch.where(is_injured)[0]
    injured_calf_joints = calf_joint_indices[injured_env_ids]  # (N,)
    injured_lock_angles = lock_angles[injured_env_ids]  # (N,)

    # ━━━ (1) Action Masking ━━━
    # action_manager의 내부 action 버퍼에서 부상 calf joint의 action을 0으로 강제
    # Go1: action dim = 12 (4 legs × 3 joints), calf = leg_idx * 3 + 2
    try:
        # action_manager._action은 process_action()에서 저장된 raw action
        action_buf = env.action_manager.action
        if action_buf is not None and action_buf.ndim == 2:
            peg_legs = peg_leg_idx[injured_env_ids]
            calf_action_indices = peg_legs * 3 + 2  # Go1: hip=0, thigh=1, calf=2
            # 벡터화 인덱싱
            action_buf[injured_env_ids, calf_action_indices] = 0.0
    except Exception:
        pass

    # ━━━ (2) Joint Position/Velocity Enforcement ━━━
    # 물리 엔진이 관절을 살짝 움직였을 수 있으므로, 다음 스텝 시작 전에 강제 복귀
    if hasattr(robot.data, "joint_pos") and robot.data.joint_pos.ndim >= 2:
        robot.data.joint_pos[injured_env_ids, injured_calf_joints] = injured_lock_angles
    if hasattr(robot.data, "joint_vel") and robot.data.joint_vel.ndim >= 2:
        robot.data.joint_vel[injured_env_ids, injured_calf_joints] = 0.0
    if (
        hasattr(robot.data, "joint_pos_target")
        and robot.data.joint_pos_target.ndim >= 2
    ):
        robot.data.joint_pos_target[injured_env_ids, injured_calf_joints] = (
            injured_lock_angles
        )


# =====================================================================
# 커리큘럼 함수
# =====================================================================

# Go1 기본 calf 각도 -1.5 rad → 다리 유효 길이
_GO1_DEFAULT_LEG_LENGTH = 0.312  # calf_angle_to_splint_length(tensor(-1.5))


def peg_leg_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    prob_start: float = 0.1,
    prob_end: float = 0.5,
    prob_ramp_steps: int = 3000,
    splint_start: float = 0.30,
    splint_end: float = 0.20,
    splint_lo_start: float | None = None,
    splint_lo_end: float | None = None,
    splint_ramp_steps: int = 5000,
) -> dict:
    """부상 난이도를 학습 진행에 따라 점진적으로 증가시키는 커리큘럼.

    Phase 1: 부상 확률 10%→50% (iter 0→3000)
      → 정책이 먼저 정상 보행을 유지하면서 소수의 부상 env에서 적응 시작

    Phase 2: 부목 길이 0.30m→0.20m (iter 0→5000)
      → 처음엔 거의 정상 길이(0.31m)와 비슷한 0.30m부터 시작
      → 점점 짧아지면서 절뚝임이 더 필요한 시나리오로 전환

    Args:
        prob_start/prob_end: 부상 확률 시작/종료값
        prob_ramp_steps: 부상 확률이 최종값에 도달하는 이터레이션 수
        splint_start/splint_end: 부목 길이 상한의 시작/종료값 (m)
        splint_lo_start/splint_lo_end: 부목 길이 하한의 시작/종료값 (m).
            None이면 기존처럼 상한의 80%를 하한으로 사용.
        splint_ramp_steps: 부목 길이가 최종값에 도달하는 이터레이션 수

    Returns:
        dict: TensorBoard에 로깅할 커리큘럼 상태
    """
    step = env.common_step_counter

    # ━━━ (1) 부상 확률 커리큘럼 ━━━
    prob_alpha = min(1.0, step / max(1, prob_ramp_steps))
    cur_prob = prob_start + (prob_end - prob_start) * prob_alpha

    # ━━━ (2) 부목 길이 커리큘럼 ━━━
    # splint_start(쉬움, 정상에 가까움) → splint_end(어려움, 짧음)
    splint_alpha = min(1.0, step / max(1, splint_ramp_steps))
    cur_splint_hi = splint_start + (splint_end - splint_start) * splint_alpha
    if splint_lo_start is None or splint_lo_end is None:
        # 하한은 상한의 80%로 유지 (기존 동작)
        cur_splint_lo = max(GO1_MIN_SPLINT_LENGTH, cur_splint_hi * 0.8)
    else:
        cur_splint_lo = splint_lo_start + (splint_lo_end - splint_lo_start) * splint_alpha
        cur_splint_lo = max(GO1_MIN_SPLINT_LENGTH, min(cur_splint_lo, cur_splint_hi))

    # env에 커리큘럼 파라미터 저장 → randomize_peg_leg_actuation이 읽음
    env._curriculum_prob_peg_leg = cur_prob
    env._curriculum_splint_range = (cur_splint_lo, cur_splint_hi)

    return {
        "prob_peg_leg": cur_prob,
        "splint_hi": cur_splint_hi,
        "splint_lo": cur_splint_lo,
    }
