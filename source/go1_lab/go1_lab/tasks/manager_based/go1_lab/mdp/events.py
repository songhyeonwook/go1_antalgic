# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""이벤트 함수들 - 의족(Peg Leg) 시나리오를 위한 랜덤화 함수들.

⚠️ 핵심 설계 원칙 (explicit actuator 호환):
  Go1은 explicit actuator를 사용합니다 — 기본은 ActuatorNetMLP, 실험 구성
  (GO1_PD_ACTUATOR=1)에서는 DCMotor PD(Kp~20, Kd~0.5). 두 경우 모두 PhysX
  내부 PD 제어기가 비활성이므로, robot.data.joint_stiffness(PhysX 게인)에
  값을 쓰는 것은 아무런 물리적 효과가 없습니다. (반면 DCMotor의
  actuator.stiffness/damping 버퍼는 토크 계산에 직접 쓰이므로
  apply_peg_leg_calf_stiffness가 이를 이용해 부목 무릎을 compliant하게 만듭니다.)

  관절을 "고정"하려면:
    (1) default_joint_pos를 lock angle로 설정 (action=0일 때 target=lock_angle이 되도록)
    (2) 매 스텝 action masking: 부상 calf joint의 action을 0으로 강제
    (3) 리셋 시 write_joint_state_to_sim 으로 실제 PhysX 관절 상태를 lock angle 에 배치
  이 세 가지를 함께 해야 하며, 관절을 그 자세로 붙잡는 힘은 PD 액추에이터가 냅니다.
  robot.data.joint_pos 에 직접 대입하면 안 됩니다 — 자세한 이유는
  randomize_peg_leg_actuation / enforce_peg_leg_constraints 참고.
"""

from __future__ import annotations

import math
import os
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
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
# Go1의 explicit actuator(ActuatorNetMLP/DCMotor 공통)는 매 substep compute()에서
#   applied_effort = clip(computed_torque, -effort_limit, +effort_limit)
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
    torque_scale: float,
    weaken_joints: str
) -> None:
    """리셋 시 부상 다리 hip-joint effort_limit을 nominal의 일정 비율(기본 5%)로 낮춘다.

    healthy(leg_idx<0) 환경은 nominal로 복구한다. effort_limit는 actuator 버퍼이므로
    다음 리셋까지 유지되며, explicit actuator(실험 구성은 DCMotor)가 매 substep
    _clip_effort 로 이 값을 적용한다.
    """
    scale = float(torque_scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"torque_scale must be in [0, 1], got {scale}")
        
    _ensure_hip_effort_cache(env, robot)
    if not getattr(env, "_peg_hip_effort_ready", False):
        return

    # (1) 리셋 대상 환경의 effort_limit을 전부 nominal로 복구
    for actuator in getattr(robot, "actuators", {}).values():
        nominal = getattr(actuator, "_peg_nominal_effort_limit", None)
        if nominal is not None and torch.is_tensor(actuator.effort_limit):
            actuator.effort_limit[env_ids_t] = nominal[env_ids_t]

    # 1.0이면 약화하지 않음
    if scale >= 1.0:
        return

    which = weaken_joints.strip().lower()
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
    stiffness: float | None,
    damping: float,
) -> None:
    """Make the injured (splinted) calf a COMPLIANT spring at reset: set its PD
    stiffness/damping to GO1_SPLINT_CALF_STIFFNESS (default unset = nominal, rigid
    hold). A real splint flexes slightly under load; a FINITE-stiffness knee
    ABSORBS the loading impact, so the splinted leg can bear partial load without
    the trunk collapsing (the rigid-knee → sink failure). Healthy legs keep nominal
    stiffness. Per-leg, joint-name based (Go1 joint order is per-TYPE, not per-leg).
    """
    if stiffness is None:
        return
    
    kp = float(stiffness)  
    kd = float(damping)  
    
    for actuator in getattr(robot, "actuators", {}).values():
        st = getattr(actuator, "stiffness", None)
        dm = getattr(actuator, "damping", None)
        if not (torch.is_tensor(st) and st.ndim == 2):
            continue
        
        # 최초 한 번 정상 Kp/Kd 저장
        if not hasattr(actuator, "_peg_nominal_calf_stiffness"):
            actuator._peg_nominal_calf_stiffness = st.clone()
            actuator._peg_nominal_calf_damping = dm.clone() if torch.is_tensor(dm) else None
        
        # 이번에 reset된 환경을 정상 gain으로 먼저 복구
        st[env_ids_t] = actuator._peg_nominal_calf_stiffness[env_ids_t]
        if torch.is_tensor(dm) and actuator._peg_nominal_calf_damping is not None:
            dm[env_ids_t] = actuator._peg_nominal_calf_damping[env_ids_t]
        
        # 이번 episode에서 선택된 부상 calf에 적용
        for leg_idx in range(4):
            mask = sampled_leg_idx == leg_idx
            if not mask.any():
                continue
            try:
                calf_idx = robot.data.joint_names.index(CALF_JOINT_NAMES[leg_idx])
            except ValueError as exc:
                raise RuntimeError(
                    f"[peg-leg] stiffness_joint_{leg_idx}: "
                    f"'{CALF_JOINT_NAMES[leg_idx]}' 를 joint_names 에서 찾지 못해 "
                    "부목 무릎 강성을 적용할 수 없습니다."
                ) from exc
                
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


def _warn_once(env: "ManagerBasedRLEnv", key: str, msg: str) -> None:
    """리셋마다 반복되는 경고를 조건당 한 번만 출력합니다.

    이 파일의 실패는 대부분 '조용히 지나가면 로그가 오히려 멀쩡해 보이는' 종류입니다
    (부상이 물리에 적용되지 않으면 보상과 에피소드 길이가 정상보다 좋아짐). 그래서
    except 로 넘길지언정 침묵하지는 않습니다.
    """
    seen = getattr(env, "_peg_warned_keys", None)
    if seen is None:
        seen = set()
        env._peg_warned_keys = seen
    if key in seen:
        return
    seen.add(key)
    print(f"[peg-leg] WARNING: {msg}", flush=True)


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
        # 정상 = 0 (부목 없음 sentinel), 부상 env 만 리셋 이벤트가 샘플값으로 채움
        env._peg_leg_foot_friction = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_peg_leg_default_joint_pos_ref"):
        env._peg_leg_default_joint_pos_ref = None


# 내부 인덱스: 0=FL, 1=FR, 2=RL, 3=RR (정상 = -1).
# privileged obs 로는 peg_leg_one_hot 이 [FL,FR,RL,RR,injured_flag] 5차원으로 노출합니다
# (GO1_INJURY_ONEHOT=1, 학습 기본값).
_TARGET_LEG_MAP: dict[str, int] = {"fl": 0, "fr": 1, "rl": 2, "rr": 3}

# 각 환경에 어느 다리가 부상인지 결정해서 다리 인덱스를 반환하는 함수
def _sample_peg_leg_indices(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    prob_peg_leg: float,
    target_leg: str = "random",
    healthy_slots: int =4,
) -> torch.Tensor:
    """각 환경마다 고장 다리 인덱스를 샘플링합니다. 정상은 -1.

    target_leg:
        "env_fixed":
            env_id % (healthy_slots + 4)로 조건을 고정합니다.

            healthy_slots=4이면:
            정상 50%
            FL/FR/RL/RR 각각 12.5%

        "random":
            reset마다 부상 여부와 부상 다리를 다시 샘플링합니다.
    """
    n = env_ids.numel()
    mode = str(target_leg).strip().lower()

    if mode == "normal" or prob_peg_leg <= 0.0:
        return torch.full((n,), -1, device=env.device, dtype=torch.long)

    # env-id 고정: 앞의 H 슬롯 = Normal, 뒤의 4 슬롯 = FL/FR/RL/RR (주기 H+4).
    # 리셋해도 조건이 바뀌지 않으므로 조건별 학습 스텝 수가 구조적으로 균등합니다.
    #
    # H = GO1_ENV_FIXED_HEALTHY_SLOTS (기본 4 → 부상 50%).
    #   H=4: 정상 50% / 각 다리 12.5%  ← 기본
    #   H=1: 정상 20% / 각 다리 20%    (부상 80% — 너무 어려움, 아래 참고)
    # 어떤 H 에서도 네 부상 조건의 env 수는 정확히 같으므로 균등 학습량은 유지됩니다.

    if mode in {"env_fixed", "balanced_env"}:
        healthy_slots = max(1, int(healthy_slots))
        period = healthy_slots + 4
        group = (env_ids % period).to(torch.long)
        
        return torch.where(
            group < healthy_slots, torch.full_like(group, -1), group - healthy_slots
        )

    # 특정 다리 고정 모드 (평가용)
    if mode in _TARGET_LEG_MAP:
        fixed_idx = _TARGET_LEG_MAP[mode]
        peg_indices = torch.full((n,), fixed_idx, device=env.device, dtype=torch.long)
        if prob_peg_leg >= 1.0:
            return peg_indices
        active = torch.rand((n,), device=env.device) < float(prob_peg_leg)
        return torch.where(active, peg_indices, torch.full_like(peg_indices, -1))

    # Balanced 모드: 리셋 배치 안에서 1:1:1:1:1 (Normal, FL, FR, RL, RR) 균등 배정을
    # 무작위 permutation 으로 수행합니다. 평가용(GO1_EVAL_MODE=balanced)이며, 학습은
    # 조건별 학습량까지 균등한 env_fixed 를 기본으로 씁니다.
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


def _foot_shape_spans(
    env: "ManagerBasedRLEnv", robot: Articulation
) -> dict[int, tuple[int, int] | None]:
    """leg_idx → 그 발 body 가 차지하는 material shape 인덱스 구간 (최초 1회 캐싱).

    get_material_properties() 는 (num_envs, num_shapes, 3) 이고 shape 는 body 순서로
    나열되므로, body 별 shape 개수를 누적해 구간을 구합니다 (Isaac Lab 의
    randomize_rigid_body_material 이 쓰는 것과 동일한 방식).
    """
    cached = getattr(env, "_peg_foot_shape_spans", None)
    if cached is not None:
        return cached

    spans: dict[int, tuple[int, int] | None] = {i: None for i in range(4)}
    try:
        num_shapes_per_body = []
        for link_path in robot.root_physx_view.link_paths[0]:
            link_view = robot._physics_sim_view.create_rigid_body_view(link_path)
            num_shapes_per_body.append(link_view.max_shapes)
        if sum(num_shapes_per_body) != robot.root_physx_view.max_shapes:
            raise ValueError(
                f"shape-per-body 합계 {sum(num_shapes_per_body)} != "
                f"max_shapes {robot.root_physx_view.max_shapes}"
            )
        body_names = list(robot.body_names)
        for leg_idx, foot in enumerate(FOOT_BODY_NAMES):
            # USD 에 따라 접미사가 붙을 수 있어 prefix 매칭 (기존 "FL_foot.*" 정규식과 동일 의도)
            match = [b for b in body_names if b.startswith(foot)]
            if not match:
                continue
            b_idx = body_names.index(match[0])
            start = sum(num_shapes_per_body[:b_idx])
            spans[leg_idx] = (start, start + num_shapes_per_body[b_idx])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"[peg-leg] foot shape mapping failed ({type(exc).__name__}: {exc}); "
            "GO1_INJURED_FOOT_FRICTION_ONLY will have no effect"
        ) from exc

    env._peg_foot_shape_spans = spans
    return spans


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
    locked_joint_angle_range: tuple[float, float] = (-1.5, -0.8),
    splint_length_range: tuple[float, float] | None = None,
    foot_friction_range: tuple[float, float] = (0.2, 1.2),
    injured_foot_friction_only: bool = False,
    target_leg: str = "random",
    hip_torque_scale: float = 1.0,
    weaken_joints: str = "hip",
    splint_calf_stiffness: float | None = None,
    splint_calf_damping: float = 0.5,
    healthy_slots: int = 4
):
    """의족 시나리오를 위한 리셋 이벤트 (explicit actuator 호환).

    이 함수는:
      (1) 부상 다리/부목 길이/발 마찰을 샘플링하여 메타데이터 버퍼에 저장
      (2) default_joint_pos 를 lock angle 로 재작성 — 관측용. joint_pos_rel 이
          이 값을 live 로 빼므로 부상 calf 채널이 ≈0 이 되고, 그 소거 보상은
          calf_pos_nominal_rel 이 담당합니다. action offset 은 액션 항 init 시
          clone 된 값이라 이 재작성은 action 경로에는 영향이 없습니다.
      (3) write_joint_state_to_sim 으로 부상 calf 를 실제 PhysX 상태에 배치
      (4) hip effort_limit 약화 / calf 강성 / 발 마찰을 물리에 적용

    관절을 실제로 붙잡는 것은 Go1LabEnv._enforce_peg_leg_joint_targets (매 sub-step
    target 덮어쓰기) + PD 이고, 매 스텝 action masking 은 Go1LabEnv.step() 이 합니다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    env_ids_t = _resolve_env_ids(env, env_ids)
    _ensure_peg_leg_buffers(env)

    # 커리큘럼 파라미터 우선 적용
    cur_prob = getattr(env, "_curriculum_prob_peg_leg", None)
    cur_splint = getattr(env, "_curriculum_splint_range", None)
    effective_prob = float(cur_prob) if cur_prob is not None else prob_peg_leg
    effective_splint = cur_splint if cur_splint is not None else splint_length_range

    # 부상다리 인덱스 추출
    sampled_leg_idx = _sample_peg_leg_indices(
        env,
        env_ids_t,
        prob_peg_leg=effective_prob,
        target_leg=target_leg,
        healthy_slots=healthy_slots
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


    healthy = sampled_leg_idx < 0
    sampled_lengths = torch.where(
        healthy, torch.zeros_like(sampled_lengths), sampled_lengths
    )
    sampled_foot_friction = torch.where(
        healthy, torch.zeros_like(sampled_foot_friction), sampled_foot_friction
    )

    # 저장
    env._peg_leg_index[env_ids_t] = sampled_leg_idx
    env._peg_leg_calf_lock_angle[env_ids_t] = sampled_lock_angles
    env._peg_leg_splint_length[env_ids_t] = sampled_lengths
    env._peg_leg_foot_friction[env_ids_t] = sampled_foot_friction

    # ━━━ 논문 §4.2: 부상 다리 hip-joint 토크를 nominal의 5%로 약화 ━━━
    apply_peg_leg_hip_torque_limit(env, robot, env_ids_t, sampled_leg_idx, hip_torque_scale, weaken_joints)
    # 부목 무릎을 compliant spring으로 (GO1_SPLINT_CALF_STIFFNESS) — 충격 흡수 → 적재 가능
    apply_peg_leg_calf_stiffness(env, robot, env_ids_t, sampled_leg_idx, splint_calf_stiffness, splint_calf_damping)

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

    # ━━━ calf joint 인덱스 매핑 ━━━
    # 관절 순서는 per-TYPE(hip 4, thigh 4, calf 4)이라 per-leg 공식(leg*3+2)은
    # 틀립니다. 항상 joint_names에서 이름으로 리졸브합니다.
    env._peg_leg_calf_joint_index[env_ids_t] = -1

    for local_i, env_id_t in enumerate(env_ids_t):
        env_id = int(env_id_t.item())
        leg_idx = int(sampled_leg_idx[local_i].item())
        if leg_idx < 0:
            continue

        joint_name = CALF_JOINT_NAMES[leg_idx]
        try:
            joint_idx = robot.data.joint_names.index(joint_name)
        except ValueError as exc:
            raise RuntimeError(
                f"[peg-leg] Required calf joint '{joint_name}' was not found in "
                f"robot.data.joint_names. "
                f"Leg index {leg_idx} is marked as injured in the observation, "
                "but its calf joint cannot be physically locked. "
                "This would cause an observation-physics mismatch."
            ) from exc
    
        env._peg_leg_calf_joint_index[env_id] = joint_idx

        if float(torch.rand((), device=env.device).item()) > float(prob_joint_disabled):
            continue

        target_lock_angle = float(sampled_lock_angles[local_i].item())

        # (1) default_joint_pos 를 lock angle 로 재작성 (관측용).
        #     joint_pos_rel = joint_pos - default 이므로 부상 calf 채널이 ≈0 이 됨.
        #     action offset 은 init 시 clone 이라 action 경로에는 영향 없음 —
        #     실제 고정은 Go1LabEnv._enforce_peg_leg_joint_targets + PD.
        if (
            hasattr(robot.data, "default_joint_pos")
            and robot.data.default_joint_pos.ndim >= 2
        ):
            robot.data.default_joint_pos[env_id, joint_idx] = target_lock_angle

        # (2) joint_pos_target 도 lock angle 로 (이건 평범한 쓰기 가능 필드입니다)
        if (
            hasattr(robot.data, "joint_pos_target")
            and robot.data.joint_pos_target.ndim >= 2
        ):
            robot.data.joint_pos_target[env_id, joint_idx] = target_lock_angle

    # ━━━ 부상 calf 를 실제 PhysX 상태로 배치 (leg 별 배치 쓰기) ━━━
    for _leg in range(4):
        _mask = sampled_leg_idx == _leg
        if not _mask.any():
            continue
        try:
            _j = robot.data.joint_names.index(CALF_JOINT_NAMES[_leg])
        except ValueError as exc:
            raise RuntimeError(
            f"[peg-leg] Required calf joint '{CALF_JOINT_NAMES[_leg]}' was not found "
            f"in robot.data.joint_names. Cannot place the injured calf at the lock angle; "
            f"splint length would not be reflected in the physics."
        ) from exc
            
        _sel = env_ids_t[_mask]
        _ang = sampled_lock_angles[_mask].to(robot.device).unsqueeze(-1)
        robot.write_joint_state_to_sim(
            position=_ang,
            velocity=torch.zeros_like(_ang),
            joint_ids=[_j],
            env_ids=_sel,
        )

    # ━━━ 발 마찰: nominal 복원 후 부상 발에 peg 마찰 재적용 ━━━
    # effort_limit / calf stiffness 와 동일한 "복원 후 재적용" 패턴이며, 한 번의
    # PhysX 쓰기로 처리합니다. 매 리셋마다 리셋 대상 env 를 nominal(startup DR
    # 결과 = healthy)로 되돌린 뒤, 부상 env 만 샘플된 마찰로 덮어씁니다.
    try:
        _phys_view = robot.root_physx_view
        _mats = _phys_view.get_material_properties()  # (num_envs, num_shapes, 3)
        if getattr(env, "_peg_nominal_material", None) is None:
            # 첫 리셋 시점의 material = startup DR 결과 = healthy nominal
            env._peg_nominal_material = _mats.clone()
        _ids = env_ids_t.to(_mats.device)
        _mats[_ids] = env._peg_nominal_material.to(_mats.device)[_ids]

        _injured = sampled_leg_idx >= 0
        if _injured.any():
            _inj_ids = env_ids_t[_injured].to(_mats.device)
            _inj_fric = sampled_foot_friction[_injured].to(_mats.device, _mats.dtype)
            # 기본은 로봇 전체(모든 shape). GO1_INJURED_FOOT_FRICTION_ONLY=1 이면
            # 부상 발 body 의 shape 구간에만 적용 — 낮은 값이 "미끄러운 부목 밑창/
            # 끌리는 발끝"을 모델링하고 건강한 발은 접지력을 유지해 추진할 수 있게
            # 합니다. 전체에 낮은 값을 주면 로봇이 통째로 미끄러워 서 있기만 하는
            # artifact 가 됩니다.
            if injured_foot_friction_only:
                _spans = _foot_shape_spans(env, robot)
                _inj_legs = sampled_leg_idx[_injured]
                for _leg in range(4):
                    _m = _inj_legs == _leg
                    if not _m.any() or _spans.get(_leg) is None:
                        continue
                    _s, _e = _spans[_leg]
                    _rows = _inj_ids[_m.to(_inj_ids.device)]
                    _vals = _inj_fric[_m.to(_inj_fric.device)].unsqueeze(-1)
                    _mats[_rows, _s:_e, 0] = _vals  # static
                    _mats[_rows, _s:_e, 1] = _vals  # dynamic
            else:
                _mats[_inj_ids, :, 0] = _inj_fric.unsqueeze(-1)
                _mats[_inj_ids, :, 1] = _inj_fric.unsqueeze(-1)

        _phys_view.set_material_properties(_mats, _ids)
    except Exception as exc:  
        raise RuntimeError(
            f"[peg-leg] Failed to apply injured-foot friction "
            f"({type(exc).__name__}: {exc}). "
            "The peg-leg friction configuration cannot be reflected in the physics."
        ) from exc

def enforce_peg_leg_constraints(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
):
    """매 스텝 호출: 부상 calf 의 action 을 0 으로, 목표각을 lock angle 로 유지합니다.

    Isaac Lab의 step 흐름:
      1. action_manager.process_action(action)  ← action이 버퍼에 저장됨
      2. physics loop:
         a. action_manager.apply_action()  ← target = offset(init 시 default clone) + action*scale
         b. scene.write_data_to_sim()      ← PhysX에 기록
         c. sim.step()                     ← 물리 시뮬레이션
         d. scene.update()                 ← 센서 업데이트
      3. reward/termination 계산
      4. event_manager.apply(mode="interval")  ← 여기서 호출됨

    interval 이벤트는 physics loop 이후에 돌므로 여기서의 action 쓰기는 다음 스텝의
    last_action 관측에만 반영됩니다. 실제 마스킹은 physics loop 이전에 도는
    Go1LabEnv.step() / PegLegActionMaskWrapper 가 담당합니다.

    관절을 lock angle 에 붙잡는 것은 PD 액추에이터의 몫이므로 여기서는 측정값
    (joint_pos/joint_vel)을 건드리지 않습니다 — 그 이유는 아래 (2) 참고.
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
    # action_manager의 내부 action 버퍼에서 부상 calf joint의 action을 0으로 강제.
    try:
        # action_manager.action은 process_action()에서 저장된 raw action
        action_buf = env.action_manager.action
        if action_buf is not None and action_buf.ndim == 2:
            # 벡터화 인덱싱 (per-type calf 인덱스, 8~11 범위)
            action_buf[injured_env_ids, injured_calf_joints] = 0.0
    except Exception as exc:  # noqa: BLE001
        # 실제 마스킹은 physics loop 이전의 Go1LabEnv.step()/래퍼가 하므로 여기 실패는
        # last_action 관측만 왜곡합니다. 그래도 침묵하지는 않습니다.
        raise RuntimeError(
            f"[peg-leg] Failed to mask injured calf action in "
            f"action_manager.action ({type(exc).__name__}: {exc}). "
            "The last_action observation would not match the action "
            "actually applied to the injured joint."
        ) from exc

    # ━━━ (2) Joint Target Enforcement ━━━
    # 목표각만 lock angle 로 유지합니다. ⚠️ 측정값(robot.data.joint_pos/joint_vel)에
    # 대입 금지 — PhysX 읽기 캐시라 sim 에 안 써지고, 액추에이터가 그 캐시로 PD 오차를
    # 계산하므로 스푸핑되어 홀딩 토크가 0 이 됩니다 (실측: 0.00 Nm vs 정상 4.74 Nm).
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
    steps_per_iteration: int = 24,
    target_leg: str = "random",
    healthy_slots: int = 4,
) -> dict:
    """부상 난이도를 학습 진행에 따라 점진적으로 증가시키는 커리큘럼.

    원래는 부상 확률을 점점 높여가는 방식으로 학습했는데 이제는 고정되어 학습됨
    cur_prob 가 의마가 없다.
    """
    steps_per_iteration = max(1, int(steps_per_iteration),)
    step = (env.common_step_counter / steps_per_iteration)

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

    # TensorBoard 에는 '실제' 부상 비율을 보고합니다. env_fixed 모드는 prob_peg_leg 를
    # 무시하고 조건을 env id 에 고정하므로, 램프 값을 그대로 로깅하면 실제 0.50 인데
    # 0.13 으로 표시되는 식으로 어긋납니다.
    target_mode = target_leg.strip().lower()
    
    if target_mode in {"env_fixed", "balanced_env",}:
        reported_prob = (4.0 / (int(healthy_slots) + 4))
    else:
        reported_prob = cur_prob

    return {
        "prob_peg_leg": reported_prob,
        "splint_hi": cur_splint_hi,
        "splint_lo": cur_splint_lo,
    }
