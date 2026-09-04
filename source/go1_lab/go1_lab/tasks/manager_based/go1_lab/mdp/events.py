# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from go1_lab.splint import LEGS, SPLINT_MAX, SPLINT_MIN, set_splint_presence


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CALF_JOINT_NAMES = [f"{leg}_calf_joint" for leg in LEGS]
SPLINT_JOINT_NAMES = [f"{leg}_splint_joint" for leg in LEGS]
SPLINT_BODY_NAMES = [f"{leg}_splint" for leg in LEGS]
FOOT_BODY_NAMES = [f"{leg}_foot" for leg in LEGS]
# Proximal "hip" actuated DOFs per leg (Go1 URDF naming):
#   *_hip_joint   = hip abduction/adduction (roll)
#   *_thigh_joint = hip flexion/extension   (pitch)
HIP_JOINT_NAMES = [f"{leg}_hip_joint" for leg in LEGS]
THIGH_JOINT_NAMES = [f"{leg}_thigh_joint" for leg in LEGS]

# per-env limit 으로 부목 관절을 잠글 때의 허용 폭 (m)
_SPLINT_LIMIT_EPS = 1.0e-4


# =====================================================================
# 부상 다리 hip-joint 토크 제한 (논문 §4.2)
# =====================================================================
#   "additionally reducing the affected hip-joint torque limits to 5% of
#    nominal to mimic peri-articular damage."
#
# Go1 의 explicit actuator 는 매 substep compute() 에서
#   applied_effort = clip(computed_torque, -effort_limit, +effort_limit)
# 로 토크를 클리핑하므로, 부상 다리 hip joint 의 effort_limit 을 낮추면
# actuator 가 자동으로 매 스텝 토크를 제한한다.


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
    weaken_joints: str,
) -> None:
    """리셋 시 부상 다리 hip-joint effort_limit을 nominal의 일정 비율로 낮춘다.

    healthy(leg_idx<0) 환경은 nominal로 복구한다. effort_limit는 actuator 버퍼이므로
    다음 리셋까지 유지되며, explicit actuator(DCMotor)가 매 substep _clip_effort 로
    이 값을 적용한다.
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
    locked_leg_idx: torch.Tensor,
    stiffness: float | None,
    damping: float,
) -> None:
    """
    잠긴 무릎(calf)을 유한 강성 스프링으로 만든다 (리셋 시).

    부목으로 고정된 무릎이므로 정상 관절(Kp=20, 정책이 제어)보다 높은 게인을 준다
    — 접촉 충격에 fold 각이 밀리지 않아야 한다. 다만 kinematic weld 처럼 무한히
    단단하게 만들지는 않는다: 유한 강성이라 적재 충격을 일부 흡수해 몸통 붕괴를 막는다.
    healthy / 잠기지 않은 다리는 nominal 게인을 유지한다.
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

        actuator_names = list(getattr(actuator, "joint_names", []))

        # 이번 episode에서 잠긴 calf에 적용
        for leg_idx in range(4):
            mask = locked_leg_idx == leg_idx
            if not mask.any():
                continue
            if CALF_JOINT_NAMES[leg_idx] not in actuator_names:
                continue
            calf_col = actuator_names.index(CALF_JOINT_NAMES[leg_idx])
            sel = env_ids_t[mask]
            st[sel, calf_col] = kp
            if torch.is_tensor(dm):
                dm[sel, calf_col] = kd


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
    """환경 객체에 부상 메타데이터 버퍼를 생성합니다."""
    if not hasattr(env, "_peg_leg_index"):
        env._peg_leg_index = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    if not hasattr(env, "_peg_leg_lock_active"):
        env._peg_leg_lock_active = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.bool
        )
    if not hasattr(env, "_peg_leg_calf_joint_index"):
        # 잠긴 calf 의 joint index (16-joint 공간). 정상/미잠금 = -1
        env._peg_leg_calf_joint_index = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    if not hasattr(env, "_peg_leg_calf_action_index"):
        # 잠긴 calf 의 action index (12-action 공간). 정상/미잠금 = -1
        env._peg_leg_calf_action_index = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    if not hasattr(env, "_peg_leg_calf_lock_angle"):
        env._peg_leg_calf_lock_angle = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_peg_leg_splint_length"):
        # 정상 = 0 sentinel, 부상 env 만 리셋 이벤트가 샘플값으로 채움
        env._peg_leg_splint_length = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_peg_leg_foot_friction"):
        # 정상 = 0 sentinel (부목 없음). 부상 env 는 부목 끝단 마찰 샘플값
        env._peg_leg_foot_friction = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_peg_leg_default_joint_pos_ref"):
        env._peg_leg_default_joint_pos_ref = None


def _ensure_splint_layout(env: "ManagerBasedRLEnv", robot: Articulation) -> None:
    """부목/calf 관절 인덱스 (joint 공간)를 1회 리졸브해 캐싱합니다.

    관절 순서는 per-TYPE 이고 부목 관절 위치는 USD 파싱 순서에 의존하므로
    항상 이름으로 리졸브합니다.
    """
    if getattr(env, "_splint_layout_ready", False):
        return
    joint_names = list(robot.data.joint_names)
    missing = [n for n in SPLINT_JOINT_NAMES if n not in joint_names]
    if missing:
        raise RuntimeError(
            f"[peg-leg] 부목 관절 {missing} 이 robot.data.joint_names 에 없습니다. "
            "robot.spawn.usd_path 가 부목 USD(build_cached_splint_usd 산출물)를 "
            "가리키는지 확인하세요."
        )
    env._splint_joint_ids = [joint_names.index(n) for n in SPLINT_JOINT_NAMES]
    env._calf_joint_ids = [joint_names.index(n) for n in CALF_JOINT_NAMES]
    env._splint_layout_ready = True


def _ensure_calf_action_ids(env: "ManagerBasedRLEnv", robot: Articulation) -> None:
    """calf 관절의 action-vector 인덱스를 1회 리졸브해 캐싱합니다.

    action_dim(12) ≠ num_joints(16) 이므로 joint index 를 action 버퍼 인덱스로
    쓰면 안 됩니다. 액션 항이 리졸브한 관절 이름 목록에서 위치를 찾습니다.
    """
    if getattr(env, "_calf_action_ids", None) is not None:
        return
    ids = [-1, -1, -1, -1]
    try:
        terms = env.action_manager._terms  # dict[str, ActionTerm], 순서 보존
    except AttributeError:
        terms = {}
    offset = 0
    for term in terms.values():
        joint_names = getattr(term, "_joint_names", None)
        if joint_names is None:
            offset += term.action_dim
            continue
        if not isinstance(joint_names, (list, tuple)):
            joint_names = list(robot.data.joint_names)
        for k, calf_name in enumerate(CALF_JOINT_NAMES):
            if calf_name in joint_names:
                ids[k] = offset + list(joint_names).index(calf_name)
        offset += term.action_dim
    if any(i < 0 for i in ids):
        _warn_once(
            env,
            "calf_action_ids",
            f"calf action index 리졸브 실패 ({ids}). 부상 calf 의 action 마스킹이 "
            "동작하지 않습니다 — action term joint_names 설정을 확인하세요.",
        )
    env._calf_action_ids = ids


def _assign_leg_by_ratio(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    leg_ratios,
    deterministic: bool,
) -> torch.Tensor:
    """비율에 따라 [Normal, FL, FR, RL, RR] 을 배정한다. 정상은 -1.

    leg_ratios: (normal, FL, FR, RL, RR) 순 5개. 합이 1 이 아니면 정규화한다.

    deterministic=True  (env_fixed): env_id 를 [0,1) 로 펼쳐 누적비율 구간에 넣는다.
        리셋해도 조건이 안 바뀌므로 부목 presence 토글(CPU USD 연산) 비용이 0 이고,
        조건별 학습 스텝 수가 구조적으로 균등하다.
    deterministic=False (random): 리셋마다 같은 분포에서 다시 추첨한다.
    """
    if leg_ratios is None:
        raise ValueError("leg_ratios 가 필요합니다 (normal, FL, FR, RL, RR 순 5개).")

    ratios = torch.as_tensor(leg_ratios, dtype=torch.float32, device=env.device)
    if ratios.numel() != 5:
        raise ValueError(
            f"leg_ratios 는 5개여야 합니다 (normal, FL, FR, RL, RR): {leg_ratios}"
        )
    if bool(torch.any(ratios < 0)) or float(ratios.sum()) <= 0.0:
        raise ValueError(f"leg_ratios 는 음수가 없고 합이 양수여야 합니다: {leg_ratios}")

    cum = torch.cumsum(ratios / ratios.sum(), dim=0)
    cum[-1] = 1.0 + 1e-6            # 부동소수 오차로 마지막 구간이 새지 않게

    if deterministic:
        pos = (env_ids.to(torch.float32) + 0.5) / float(max(1, env.num_envs))
    else:
        pos = torch.rand((env_ids.numel(),), device=env.device)

    bucket = torch.searchsorted(cum.contiguous(), pos.contiguous()).clamp(max=4)
    # bucket 0 = Normal(-1), 1~4 = FL/FR/RL/RR (0~3)
    return torch.where(bucket == 0, torch.full_like(bucket, -1), bucket - 1)

_LEG_LINK_LENGTH = 0.213      # Go1 thigh / calf 링크 길이 (URDF)


def _sample_calf_fold_angle(
    splint_lengths: torch.Tensor,
    injured: torch.Tensor,
    *,
    random_mode: bool,
    fixed_angle: float,
    min_fold: float,
    max_fold: float,
    padding: float,
) -> torch.Tensor:
    """env 별 calf 접힘각 [rad]. 정상 env 는 0.

    thigh joint → 발 거리는 reach(q) = 2*L_link*|cos(q/2)| 이므로,
    발끝이 부목 끝단보다 padding 만큼 안쪽에 오는 각은
        a = -2*arccos((splint_length - padding) / (2*L_link))
    이다. 이 a 를 [min_fold, max_fold] 로 클리핑해 상한으로 쓰고,
    [min_fold, 상한] 에서 균등 추첨한다.

    부목이 다리 최대신장(2*L_link)보다 길면 a 가 0 에 붙으므로
    max_fold 가 상한을 결정한다 (무릎이 펴지는 것을 막는다).
    """
    if not random_mode:
        angles = torch.full_like(splint_lengths, float(fixed_angle))
        return torch.where(injured, angles, torch.zeros_like(angles))

    span = 2.0 * _LEG_LINK_LENGTH
    ratio = torch.clamp((splint_lengths - float(padding)) / span, max=1.0)
    a = -2.0 * torch.acos(ratio)

    upper = torch.clamp(a, min=float(min_fold), max=float(max_fold))
    lower = torch.full_like(upper, float(min_fold))

    angles = lower + (upper - lower) * torch.rand_like(upper)
    return torch.where(injured, angles, torch.zeros_like(angles))


def _get_peg_leg_per_env(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None
) -> dict[int, int | None]:
    """환경별 고장 다리 매핑(dict[env_id, leg_idx|None])을 반환합니다.

    분석 스크립트(play_result.py, analyze_student.py)가 사용합니다.
    버퍼가 아직 없으면(리셋 이벤트 이전) 전부 정상(None)으로 간주합니다.
    """
    env_ids_t = _resolve_env_ids(env, env_ids)
    result: dict[int, int | None] = {}

    if hasattr(env, "_peg_leg_index"):
        peg_indices = env._peg_leg_index[env_ids_t].detach().cpu().tolist()
        for env_id, peg_idx in zip(
            env_ids_t.detach().cpu().tolist(), peg_indices, strict=False
        ):
            result[env_id] = None if peg_idx < 0 else int(peg_idx)
        return result

    for env_id in env_ids_t.detach().cpu().tolist():
        result[env_id] = None
    return result


def _sample_foot_friction(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, friction_range: tuple[float, float]
) -> torch.Tensor:
    lo, hi = float(friction_range[0]), float(friction_range[1])
    return lo + (hi - lo) * torch.rand((env_ids.numel(),), device=env.device)


def _sample_splint_lengths(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, length_range: tuple[float, float]
) -> torch.Tensor:
    """부목 길이 L 을 균등 샘플합니다. USD 관절 한계 [SPLINT_MIN, SPLINT_MAX] 로
    클램프합니다 — 범위를 넘겨 샘플하면 USD limit 에 조용히 clip 되는 함정이
    있으므로 (usd_builder 참고) 여기서 명시적으로 막습니다."""
    lo = max(float(length_range[0]), SPLINT_MIN)
    hi = min(float(length_range[1]), SPLINT_MAX)
    if lo > hi:
        raise ValueError(
            f"splint_length_range {length_range} 가 USD 설계 한계 "
            f"[{SPLINT_MIN}, {SPLINT_MAX}] 와 겹치지 않습니다."
        )
    return lo + (hi - lo) * torch.rand((env_ids.numel(),), device=env.device)


def _splint_shape_spans(
    env: "ManagerBasedRLEnv", robot: Articulation
) -> dict[int, tuple[int, int] | None]:
    """leg_idx → 부목 body 가 차지하는 material shape 인덱스 구간 (최초 1회 캐싱).

    get_material_properties() 는 (num_envs, num_shapes, 3) 이고 shape 는 body 순서로
    나열되므로, body 별 shape 개수를 누적해 구간을 구합니다 (Isaac Lab 의
    randomize_rigid_body_material 이 쓰는 것과 동일한 방식).
    콜라이더를 비활성해도 shape 레이아웃은 유지됩니다 (spike[2] 로 실측 확인).
    """
    cached = getattr(env, "_peg_splint_shape_spans", None)
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
        for leg_idx, splint_body in enumerate(SPLINT_BODY_NAMES):
            if splint_body not in body_names:
                continue
            b_idx = body_names.index(splint_body)
            start = sum(num_shapes_per_body[:b_idx])
            spans[leg_idx] = (start, start + num_shapes_per_body[b_idx])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"[peg-leg] splint shape mapping failed ({type(exc).__name__}: {exc}); "
            "injured_splint_friction_only cannot be applied"
        ) from exc

    env._peg_splint_shape_spans = spans
    return spans


def initialize_splint_presence(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    attach: str = "thigh",
):
    """startup 이벤트: 모든 부목을 '없음' 상태로 동기화합니다.

    USD 상의 부목은 4 다리 모두 기본 '존재'(렌더+질량+콜라이더) 상태이므로,
    학습 시작 시 _peg_leg_index(전부 -1) 와 USD presence 상태의 불변식
    "presence == _peg_leg_index" 를 여기서 확립합니다. 이후에는
    randomize_peg_leg_actuation 이 diff 로만 갱신하므로, env_fixed 모드에서는
    첫 리셋 이후 presence 토글 비용이 0 입니다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    _ensure_peg_leg_buffers(env)
    _ensure_splint_layout(env, robot)
    env._splint_attach = str(attach).strip().lower()
    set_splint_presence(env.scene, robot, env._peg_leg_index, env._splint_attach)


def randomize_peg_leg_actuation(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    splint_length_range: tuple[float, float] = (0.33, 0.45),
    foot_friction_range: tuple[float, float] = (0.2, 1.2),
    injured_splint_friction_only: bool = True,
    calf_random: bool = False,
    calf_fixed_angle: float = -2.55,
    calf_min_fold: float = -2.55,
    calf_max_fold: float = -1.0,
    calf_padding: float = 0.02,
    attach: str = "thigh",
    hip_torque_scale: float = 1.0,
    weaken_joints: str = "hip",
    splint_calf_stiffness: float | None = None,
    splint_calf_damping: float = 0.5,
    leg_deterministic: bool = True,
    leg_ratios=None,
):
    """부목 부상 시나리오 리셋 이벤트.

    이 함수는:
      (1) 부상 다리 / 부목 길이 L / 부목 끝단 마찰을 샘플링해 버퍼에 저장
      (2) 부목 presence 를 diff 로 갱신 (바뀐 env 만 — env_fixed 면 첫 리셋뿐)
      (3) 부목 관절을 L 로 배치하고 per-env limit + drive target 으로 잠금
      (5) hip effort_limit 약화 / calf 강성 / 부목 끝단 마찰을 물리에 적용

    관절을 실제로 붙잡는 것은 Go1LabEnv._enforce_peg_leg_joint_targets (매 sub-step
    target 덮어쓰기) + PD 이고, 매 스텝 action masking 은 Go1LabEnv.step() 이 합니다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    env_ids_t = _resolve_env_ids(env, env_ids)
    n = env_ids_t.numel()
    _ensure_peg_leg_buffers(env)
    _ensure_splint_layout(env, robot)
    _ensure_calf_action_ids(env, robot)


    # ━━━ 샘플링 ━━━
    sampled_leg_idx = _assign_leg_by_ratio(
        env, env_ids_t, leg_ratios, deterministic=leg_deterministic
    )

    sampled_lengths = _sample_splint_lengths(env, env_ids_t, splint_length_range)
    sampled_foot_friction = _sample_foot_friction(
        env, env_ids_t, friction_range=foot_friction_range
    )

    healthy = sampled_leg_idx < 0
    sampled_lengths = torch.where(
        healthy, torch.zeros_like(sampled_lengths), sampled_lengths
    )
    sampled_foot_friction = torch.where(
        healthy, torch.zeros_like(sampled_foot_friction), sampled_foot_friction
    )
    # calf 잠금 여부 (기본 1.0 → 부상이면 항상 잠금)
    lock_active = (~healthy)

    # ━━━ (1) 부목 presence — 바뀐 env 만 갱신 ━━━
    prev_leg = env._peg_leg_index[env_ids_t]
    changed = prev_leg != sampled_leg_idx
    env._peg_leg_index[env_ids_t] = sampled_leg_idx
    if bool(changed.any()):
        set_splint_presence(
            env.scene,
            robot,
            env._peg_leg_index,
            getattr(env, "_splint_attach", attach),
            env_ids=env_ids_t[changed],
        )

    # ━━━ 버퍼 저장 ━━━
    fold_angles = _sample_calf_fold_angle(
        sampled_lengths, lock_active,
        random_mode=calf_random,
        fixed_angle=calf_fixed_angle,
        min_fold=calf_min_fold,
        max_fold=calf_max_fold,
        padding=calf_padding,
    )
    env._peg_leg_calf_lock_angle[env_ids_t] = fold_angles

    env._peg_leg_splint_length[env_ids_t] = sampled_lengths
    env._peg_leg_foot_friction[env_ids_t] = sampled_foot_friction
    env._peg_leg_lock_active[env_ids_t] = lock_active

    # ━━━ 부상 다리 hip-joint 토크 약화 + 잠긴 calf 의 compliant 강성 ━━━
    apply_peg_leg_hip_torque_limit(
        env, robot, env_ids_t, sampled_leg_idx, hip_torque_scale, weaken_joints
    )
    locked_leg_idx = torch.where(
        lock_active, sampled_leg_idx, torch.full_like(sampled_leg_idx, -1)
    )
    apply_peg_leg_calf_stiffness(
        env, robot, env_ids_t, locked_leg_idx, splint_calf_stiffness, splint_calf_damping
    )

    # action offset 은 액션 항 init 시 clone 된 값이라 action 경로에는 영향 없음.
    if hasattr(robot.data, "default_joint_pos"):
        if env._peg_leg_default_joint_pos_ref is None:
            ref = robot.data.default_joint_pos
            env._peg_leg_default_joint_pos_ref = (
                ref[0].clone() if ref.ndim == 2 else ref.clone()
            )
        if robot.data.default_joint_pos.ndim == 2:
            robot.data.default_joint_pos[env_ids_t] = (
                env._peg_leg_default_joint_pos_ref.unsqueeze(0).expand(n, -1)
            )

    # ━━━ calf joint/action 인덱스 버퍼 + fold 배치 (leg 별 벡터화) ━━━
    env._peg_leg_calf_joint_index[env_ids_t] = -1
    env._peg_leg_calf_action_index[env_ids_t] = -1

    calf_action_ids = getattr(env, "_calf_action_ids", [-1, -1, -1, -1])
    for k in range(4):
        mask = locked_leg_idx == k
        if not bool(mask.any()):
            continue
        sel = env_ids_t[mask]
        calf_j = env._calf_joint_ids[k]

        env._peg_leg_calf_joint_index[sel] = calf_j
        if calf_action_ids[k] >= 0:
            env._peg_leg_calf_action_index[sel] = calf_action_ids[k]


        # (b) 실제 PhysX 관절 상태를 fold 각에 배치
        ang = fold_angles[mask].to(robot.device).unsqueeze(-1)
        robot.write_joint_state_to_sim(
            position=ang,
            velocity=torch.zeros_like(ang),
            joint_ids=[calf_j],
            env_ids=sel,
        )

    # ━━━ 부목 관절: 부상 다리만 L, 나머지는 SPLINT_MIN 에 주차 ━━━
    splint_ids = env._splint_joint_ids
    sp_pos = torch.full((n, 4), SPLINT_MIN, device=robot.device)
    has = ~healthy
    if bool(has.any()):
        rows = torch.nonzero(has, as_tuple=False).squeeze(-1)
        cols = sampled_leg_idx[has]
        sp_pos[rows, cols] = sampled_lengths[rows]
    robot.write_joint_state_to_sim(
        position=sp_pos,
        velocity=torch.zeros_like(sp_pos),
        joint_ids=splint_ids,
        env_ids=env_ids_t,
    )
    # per-env limit 으로 잠근다 (부상 다리 관절만 L 로 좁힌다)
    for k in range(4):
        mask_k = sampled_leg_idx == k
        lo = torch.where(
            mask_k, sampled_lengths - _SPLINT_LIMIT_EPS,
            torch.full((n,), SPLINT_MIN - _SPLINT_LIMIT_EPS, device=robot.device),
        )
        hi = torch.where(
            mask_k, sampled_lengths + _SPLINT_LIMIT_EPS,
            torch.full((n,), SPLINT_MIN + _SPLINT_LIMIT_EPS, device=robot.device),
        )
        robot.write_joint_position_limit_to_sim(
            torch.stack([lo, hi], dim=-1).unsqueeze(1),
            joint_ids=[splint_ids[k]],
            env_ids=env_ids_t,
            warn_limit_violation=False,
        )
    # drive target 도 같은 값으로 (limit + drive 이중 잠금)
    robot.set_joint_position_target(sp_pos, joint_ids=splint_ids, env_ids=env_ids_t)

    # ━━━ 부목 끝단 마찰: nominal 복원 후 부상 부목에 샘플값 재적용 ━━━
    # effort_limit / calf stiffness 와 동일한 "복원 후 재적용" 패턴. 매 리셋마다
    # 리셋 대상 env 를 nominal(startup DR 결과)로 되돌린 뒤, 부상 env 만 덮어씁니다.
    try:
        _phys_view = robot.root_physx_view
        _mats = _phys_view.get_material_properties()  # (num_envs, num_shapes, 3)
        if getattr(env, "_peg_nominal_material", None) is None:
            # 첫 리셋 시점의 material = startup DR 결과 = healthy nominal
            env._peg_nominal_material = _mats.clone()
        _ids = env_ids_t.to(_mats.device)
        _mats[_ids] = env._peg_nominal_material.to(_mats.device)[_ids]

        _injured = sampled_leg_idx >= 0
        if bool(_injured.any()):
            _inj_ids = env_ids_t[_injured].to(_mats.device)
            _inj_fric = sampled_foot_friction[_injured].to(_mats.device, _mats.dtype)
            # 기본은 부상 다리의 부목 끝단 shape 에만 적용 — 낮은 값이 "미끄러운
            # 부목 밑창"을 모델링하고 건강한 발은 접지력을 유지해 추진할 수 있게
            # 합니다. false 면 로봇 전체 shape 에 적용 (전신이 미끄러워 서 있기만
            # 하는 artifact 가 생길 수 있어 비권장).
            if injured_splint_friction_only:
                _spans = _splint_shape_spans(env, robot)
                _inj_legs = sampled_leg_idx[_injured]
                for _leg in range(4):
                    _m = _inj_legs == _leg
                    if not bool(_m.any()) or _spans.get(_leg) is None:
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
            f"[peg-leg] Failed to apply splint-tip friction "
            f"({type(exc).__name__}: {exc}). "
            "The sampled friction cannot be reflected in the physics."
        ) from exc


def enforce_peg_leg_constraints(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
):
    """매 스텝 호출: 잠긴 calf 의 action 을 0 으로, 목표각을 fold 각으로 유지합니다.

    interval 이벤트는 physics loop 이후에 돌므로 여기서의 action 쓰기는 다음 스텝의
    last_action 관측에만 반영됩니다. 실제 마스킹은 physics loop 이전에 도는
    Go1LabEnv.step() 이 담당합니다.

    관절을 fold 각에 붙잡는 것은 PD 액추에이터의 몫이므로 여기서는 측정값
    (joint_pos/joint_vel)을 건드리지 않습니다.
    """
    if not hasattr(env, "_peg_leg_index"):
        return

    robot: Articulation = env.scene[asset_cfg.name]
    lock_angles = env._peg_leg_calf_lock_angle  # (num_envs,)
    calf_joint_indices = env._peg_leg_calf_joint_index  # (num_envs,) -1=미잠금
    calf_action_indices = env._peg_leg_calf_action_index  # (num_envs,) -1=미잠금

    locked = env._peg_leg_lock_active & (calf_joint_indices >= 0)
    if not bool(locked.any()):
        return

    locked_env_ids = torch.where(locked)[0]

    # ━━━ (1) Action Masking (action-index 공간) ━━━
    try:
        action_buf = env.action_manager.action
        if action_buf is not None and action_buf.ndim == 2:
            act_ids = calf_action_indices[locked_env_ids]
            valid = act_ids >= 0
            if bool(valid.any()):
                action_buf[locked_env_ids[valid], act_ids[valid]] = 0.0
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"[peg-leg] Failed to mask injured calf action in "
            f"action_manager.action ({type(exc).__name__}: {exc}). "
            "The last_action observation would not match the action "
            "actually applied to the injured joint."
        ) from exc

    # ━━━ (2) Joint Target Enforcement (joint-index 공간) ━━━
    # ⚠️ 측정값(robot.data.joint_pos/joint_vel) 대입 금지 — PhysX 읽기 캐시라 sim 에
    # 안 써지고, 액추에이터가 그 캐시로 PD 오차를 계산하므로 스푸핑되어 홀딩 토크가
    # 0 이 됩니다 (실측: 0.00 Nm vs 정상 4.74 Nm).
    if (
        hasattr(robot.data, "joint_pos_target")
        and robot.data.joint_pos_target.ndim >= 2
    ):
        robot.data.joint_pos_target[
            locked_env_ids, calf_joint_indices[locked_env_ids]
        ] = lock_angles[locked_env_ids]


