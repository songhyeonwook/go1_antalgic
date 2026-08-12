# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common reward terms for the Go1 Lab environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .events import (
    CALF_JOINT_NAMES,
    HIP_JOINT_NAMES,
    SPLINT_BODY_NAMES,
    THIGH_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _peg_leg_index_per_env(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """각 env의 고장 다리 인덱스(-1, 0..3)를 반환합니다.

    ⚠️ 중요: Phase1 healthy 학습에서는 `randomize_peg_leg_actuation` 이벤트가 등록되지
    않아 `env._peg_leg_index` 버퍼가 생성되지 않습니다. 과거의 `env_id % 5` fallback은
    80% env 를 부상 처리하여 `reward_trot_synchronization` 등 정상 env 전용 보상을
    대부분의 env에서 꺼버리는 심각한 버그를 유발했습니다.

    새 fallback: 버퍼가 없으면 "모두 정상(-1)" 으로 간주합니다. Phase2/3 에서는 reset
    이벤트가 버퍼를 생성·갱신하므로 이 경로는 healthy 에서만 쓰입니다.
    """
    if hasattr(env, "_peg_leg_index"):
        return env._peg_leg_index.to(device=env.device, dtype=torch.long)
    return torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)


def _step_ramp(env: "ManagerBasedRLEnv", ramp_start_steps: int = 0, ramp_duration_steps: int = 1) -> float:
    """현재 common_step_counter 기준 선형 ramp 계수 [0, 1]."""
    step = float(getattr(env, "common_step_counter", 0))
    start = float(ramp_start_steps)
    duration = max(float(ramp_duration_steps), 1.0)
    return float(max(0.0, min(1.0, (step - start) / duration)))


def _foot_force_tensor(env: "ManagerBasedRLEnv", sensor_name: str, use_z_only: bool) -> tuple[torch.Tensor, list[int | None]]:
    """발 링크별 접촉력을 반환합니다. shape: (num_envs, 4)."""
    try:
        contact_sensor = env.scene[sensor_name]
    except Exception:
        return torch.zeros((env.num_envs, 4), device=env.device), [None, None, None, None]

    contact_forces_data = contact_sensor.data.net_forces_w
    if contact_forces_data is None:
        return torch.zeros((env.num_envs, 4), device=env.device), [None, None, None, None]

    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    sensor_body_names = contact_sensor.body_names
    foot_indices: list[int | None] = []

    def find_body_idx(name: str) -> int | None:
        for idx, body_name in enumerate(sensor_body_names):
            if body_name == name or (name in body_name):
                return idx
        return None

    for foot_name in foot_names:
        foot_idx = find_body_idx(foot_name)
        if foot_idx is None:
            for alt_name in [
                foot_name.replace("_foot", "_foot_link"),
                foot_name.replace("_foot", "_foot_link_0"),
                foot_name.lower(),
            ]:
                foot_idx = find_body_idx(alt_name)
                if foot_idx is not None:
                    break
        foot_indices.append(foot_idx)

    out = torch.zeros((env.num_envs, 4), device=env.device)
    for i, foot_idx in enumerate(foot_indices):
        if foot_idx is None or foot_idx >= contact_forces_data.shape[1]:
            continue
        forces = contact_forces_data[:, foot_idx]
        out[:, i] = torch.abs(forces[:, 2]) if use_z_only else torch.norm(forces, dim=1)
    return out, foot_indices


def _foot_force_ema(
    env: "ManagerBasedRLEnv",
    sensor_name: str,
    use_z_only: bool,
    ema_alpha: float,
) -> torch.Tensor:
    """발 접촉력의 per-env EMA를 반환합니다.

    Trot은 좌우 다리가 같은 순간에 같은 힘을 내는 보행이 아닙니다. 좌우 force 대칭은
    instantaneous force가 아니라 시간 평균 기준으로 평가해야 하므로 EMA를 사용합니다.
    """
    step = int(getattr(env, "common_step_counter", 0))
    cached_step = getattr(env, "_go1_foot_force_ema_step", None)
    cached_alpha = getattr(env, "_go1_foot_force_ema_alpha", None)
    cached_use_z = getattr(env, "_go1_foot_force_ema_use_z_only", None)
    cached_sensor = getattr(env, "_go1_foot_force_ema_sensor_name", None)
    cached_ema = getattr(env, "_go1_foot_force_ema", None)
    if (
        cached_ema is not None
        and cached_step == step
        and cached_alpha == float(ema_alpha)
        and cached_use_z == bool(use_z_only)
        and cached_sensor == sensor_name
    ):
        return cached_ema

    contact_by_foot, _ = _foot_force_tensor(env, sensor_name=sensor_name, use_z_only=use_z_only)
    alpha = float(max(0.0, min(0.9999, ema_alpha)))

    ema = getattr(env, "_go1_foot_force_ema", None)
    if ema is None or ema.shape != contact_by_foot.shape:
        ema = contact_by_foot.detach().clone()
    else:
        reset_buf = getattr(env, "reset_buf", None)
        if reset_buf is not None:
            reset_mask = reset_buf.to(device=env.device, dtype=torch.bool)
            if reset_mask.shape[0] == ema.shape[0] and reset_mask.any():
                ema[reset_mask] = contact_by_foot.detach()[reset_mask]
        ema.mul_(alpha).add_(contact_by_foot.detach(), alpha=1.0 - alpha)

    env._go1_foot_force_ema = ema
    env._go1_foot_force_ema_step = step
    env._go1_foot_force_ema_alpha = float(ema_alpha)
    env._go1_foot_force_ema_use_z_only = bool(use_z_only)
    env._go1_foot_force_ema_sensor_name = sensor_name
    return ema


def _link_force_tensor(
    env: "ManagerBasedRLEnv",
    sensor_name: str,
    link_name_candidates: list[str],
    use_z_only: bool,
) -> tuple[torch.Tensor, list[int | None]]:
    """지정 링크 후보군의 접촉력을 반환합니다. shape: (num_envs, num_links)."""
    try:
        contact_sensor = env.scene[sensor_name]
    except Exception:
        return torch.zeros((env.num_envs, len(link_name_candidates)), device=env.device), [None] * len(link_name_candidates)

    contact_forces_data = contact_sensor.data.net_forces_w
    if contact_forces_data is None:
        return torch.zeros((env.num_envs, len(link_name_candidates)), device=env.device), [None] * len(link_name_candidates)

    sensor_body_names = contact_sensor.body_names

    def find_body_idx(name: str) -> int | None:
        lowered = name.lower()
        for idx, body_name in enumerate(sensor_body_names):
            body_name_l = body_name.lower()
            if body_name_l == lowered or (lowered in body_name_l):
                return idx
        return None

    indices: list[int | None] = [find_body_idx(name) for name in link_name_candidates]
    out = torch.zeros((env.num_envs, len(link_name_candidates)), device=env.device)
    for i, body_idx in enumerate(indices):
        if body_idx is None or body_idx >= contact_forces_data.shape[1]:
            continue
        forces = contact_forces_data[:, body_idx]
        out[:, i] = torch.abs(forces[:, 2]) if use_z_only else torch.norm(forces, dim=1)
    return out, indices


def _splint_force_tensor(
    env: "ManagerBasedRLEnv", sensor_name: str, use_z_only: bool
) -> torch.Tensor:
    """부목 링크별 접촉력을 반환합니다. shape: (num_envs, 4).

    부목 모델 v2 에서 부상 다리의 접지는 발이 아니라 부목 끝단에서 일어나므로,
    부상 다리의 '하중'은 이 텐서로 측정해야 합니다 (발/calf 접촉은 통증 담당).
    """
    forces, _ = _link_force_tensor(
        env,
        sensor_name=sensor_name,
        link_name_candidates=list(SPLINT_BODY_NAMES),
        use_z_only=use_z_only,
    )
    return forces


def penalize_knee_shin_contact(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "contact_forces",
    force_threshold: float = 5.0,
    max_overload: float = 160.0,
    use_z_only: bool = True,
) -> torch.Tensor:
    """calf(무릎/정강이) 접촉을 패널티로 처리해 무릎 보행을 억제합니다."""
    _ = asset_cfg
    calf_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    contact_by_calf, _ = _link_force_tensor(
        env,
        sensor_name=sensor_name,
        link_name_candidates=calf_names,
        use_z_only=use_z_only,
    )
    overload = torch.clamp(contact_by_calf - float(force_threshold), min=0.0)
    penalty = torch.sum(torch.clamp(overload, max=float(max_overload)), dim=1)
    return penalty


def reward_trot_synchronization(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    contact_threshold: float = 1.0,
    use_z_only: bool = True,
    command_name: str = "base_velocity",
    vel_gate_threshold: float = 0.1,
    vel_gate_sharpness: float = 10.0,
    ramp_start_steps: int = 0,
    ramp_duration_steps: int = 1,
) -> torch.Tensor:
    """대각선 다리 쌍의 접지 동기화를 보상하여 trot 보행을 유도합니다.

    Trot 패턴: FL+RR 동시 접지/이탈, FR+RL 동시 접지/이탈
    정상 env에만 적용하고, 부상 env에서는 적응적 리듬 변화를 허용합니다.

    Pronking 방지: 4발이 모두 같은 상태(전부 접지 or 전부 체공)이면 점수 0.

    ⚠️ 속도 명령 게이팅 (v3 → v4): 이 보상이 너무 강하면 로봇이 **제자리에서**만
    trot 리듬을 타서 터레인 커리큘럼이 0 으로 수렴하는 "trot-in-place" 국소최적이
    발생합니다. 속도 명령 크기 ||v_cmd|| 가 작을 때는 보상을 서서히 줄여서,
    "움직이면서 trot" 해야만 보상을 받게 유도합니다:

        gate = tanh(sharpness · max(||v_cmd|| - gate_threshold, 0))
    """
    contact_by_foot, _ = _foot_force_tensor(env, sensor_name=sensor_name, use_z_only=use_z_only)
    peg_leg_idx = _peg_leg_index_per_env(env)

    in_contact = (contact_by_foot > float(contact_threshold)).float()  # (E, 4)

    diag1_sync = 1.0 - torch.abs(in_contact[:, 0] - in_contact[:, 3])  # FL-RR 동기
    diag2_sync = 1.0 - torch.abs(in_contact[:, 1] - in_contact[:, 2])  # FR-RL 동기
    anti_lr = torch.abs(in_contact[:, 0] - in_contact[:, 1])  # FL-FR 반위상
    anti_fb = torch.abs(in_contact[:, 2] - in_contact[:, 3])  # RL-RR 반위상

    trot_score = (diag1_sync + diag2_sync + anti_lr + anti_fb) / 4.0

    # Pronking/bounding 감지: 4발이 모두 같은 상태이면 trot이 아님 → 점수 0
    all_same = (
        (in_contact[:, 0] == in_contact[:, 1])
        & (in_contact[:, 1] == in_contact[:, 2])
        & (in_contact[:, 2] == in_contact[:, 3])
    )
    trot_score[all_same] = 0.0

    # 속도 명령 게이트: 명령 속도가 임계값 이하면 보상 축소 → 제자리 trot 방지
    try:
        cmd = env.command_manager.get_command(command_name)  # (E, >=3) [vx, vy, wz, ...]
        cmd_vxy = torch.linalg.norm(cmd[:, :2], dim=1)  # (E,)
        gate = torch.tanh(
            float(vel_gate_sharpness) * torch.clamp(cmd_vxy - float(vel_gate_threshold), min=0.0)
        )
        trot_score = trot_score * gate
    except Exception:
        # command manager 를 찾지 못하면 게이트 없이 사용 (이전 동작과 호환)
        pass

    reward = torch.zeros(env.num_envs, device=env.device)
    is_normal = peg_leg_idx < 0
    reward[is_normal] = trot_score[is_normal] * _step_ramp(env, ramp_start_steps, ramp_duration_steps)
    return reward


# 대칭보행 패널티
def penalize_contact_force_asymmetry(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    use_z_only: bool = True,
    ema_alpha: float = 0.995,
    ramp_start_steps: int = 0,
    ramp_duration_steps: int = 1,
) -> torch.Tensor:
    """좌우 다리 쌍(FL-FR, RL-RR)의 시간평균 접촉력 비대칭을 패널티로 부여합니다.

    정상 보행에서도 좌우 대칭 보행을 유도하고,
    부상 시에는 건측-환측 하중 차이가 자연스러우므로 부상 env는 제외합니다.
    """
    
    # 시간 평균 접촉력 계산
    contact_by_foot = _foot_force_ema(env, sensor_name=sensor_name, use_z_only=use_z_only, ema_alpha=ema_alpha) # 지수 이동평균 사용
    peg_leg_idx = _peg_leg_index_per_env(env)
    is_normal = peg_leg_idx < 0 # 정상 환경만 선택

    diff_front = torch.abs(contact_by_foot[:, 0] - contact_by_foot[:, 1])
    diff_rear = torch.abs(contact_by_foot[:, 2] - contact_by_foot[:, 3])
    asym = diff_front + diff_rear # 전체 비대칭 정도 계산

    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[is_normal] = asym[is_normal] * _step_ramp(env, ramp_start_steps, ramp_duration_steps)
    return penalty


def penalize_duty_factor_asymmetry(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    contact_threshold: float = 1.0,
    use_z_only: bool = True,
    ema_alpha: float = 0.995,
    ramp_start_steps: int = 0,
    ramp_duration_steps: int = 1,
) -> torch.Tensor:
    """좌우 다리 쌍(FL-FR, RL-RR)의 시간평균 접지율 비대칭을 패널티로 부여합니다.

    Phase 1 healthy baseline의 목표는 특정 gait pattern 처방이 아니라
    같은 축의 좌우 다리가 비슷한 duty factor를 갖는 것입니다.
    """
    contact_by_foot, _ = _foot_force_tensor(env, sensor_name=sensor_name, use_z_only=use_z_only)
    in_contact = (contact_by_foot > float(contact_threshold)).float()
    alpha = float(max(0.0, min(0.9999, ema_alpha)))

    ema = getattr(env, "_go1_foot_contact_ema", None)
    if ema is None or ema.shape != in_contact.shape:
        ema = in_contact.detach().clone()
    else:
        reset_buf = getattr(env, "reset_buf", None)
        if reset_buf is not None:
            reset_mask = reset_buf.to(device=env.device, dtype=torch.bool)
            if reset_mask.shape[0] == ema.shape[0] and reset_mask.any():
                ema[reset_mask] = in_contact.detach()[reset_mask]
        ema.mul_(alpha).add_(in_contact.detach(), alpha=1.0 - alpha)

    env._go1_foot_contact_ema = ema

    peg_leg_idx = _peg_leg_index_per_env(env)
    is_normal = peg_leg_idx < 0

    diff_front = torch.abs(ema[:, 0] - ema[:, 1])
    diff_rear = torch.abs(ema[:, 2] - ema[:, 3])
    asym = diff_front + diff_rear

    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[is_normal] = asym[is_normal] * _step_ramp(env, ramp_start_steps, ramp_duration_steps)
    return penalty


def penalize_front_rear_load_distribution(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    target_front_fraction: float = 0.60,
    tolerance: float = 0.03,
    use_z_only: bool = True,
    ema_alpha: float = 0.995,
    ramp_start_steps: int = 0,
    ramp_duration_steps: int = 1,
) -> torch.Tensor:
    """정상 보행에서 앞/뒤 하중 비율을 목표값으로 유도합니다.

    실제 사족 보행 동물은 정적 하중이 앞쪽으로 치우치는 경향이 있으므로,
    좌우는 대칭으로 두되 front pair 전체 하중이 전체의 일정 비율이 되도록 맞춥니다.
    기본 목표는 front 60%, rear 40%입니다.
    """
    contact_by_foot = _foot_force_ema(env, sensor_name=sensor_name, use_z_only=use_z_only, ema_alpha=ema_alpha)
    peg_leg_idx = _peg_leg_index_per_env(env)
    is_normal = peg_leg_idx < 0

    front_load = contact_by_foot[:, 0] + contact_by_foot[:, 1]
    rear_load = contact_by_foot[:, 2] + contact_by_foot[:, 3]
    total_load = torch.clamp(front_load + rear_load, min=1.0)

    front_fraction = front_load / total_load
    fraction_error = torch.clamp(
        torch.abs(front_fraction - float(target_front_fraction)) - float(tolerance),
        min=0.0,
    )

    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[is_normal] = fraction_error[is_normal] * total_load[is_normal] * _step_ramp(
        env, ramp_start_steps, ramp_duration_steps
    )
    return penalty


def penalize_diagonal_load_asymmetry(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    use_z_only: bool = True,
    ema_alpha: float = 0.995,
    ramp_start_steps: int = 0,
    ramp_duration_steps: int = 1,
) -> torch.Tensor:
    """트롯 대각쌍 간 시간평균 하중 불균형을 패널티로 부여합니다.

    Trot 보행의 대각쌍:
      - diag1 = FL + RR
      - diag2 = FR + RL

    정책이 한쪽 대각쌍에만 체중을 싣는 "lopsided trot" 을 방지하기 위해
    (diag1 - diag2) 의 절대값을 페널티로 더합니다.

    contact_force_symmetry 는 좌우(FL-FR, RL-RR) 만 제약하기 때문에,
    대각 편향(FL+RR vs FR+RL) 은 별도로 패널티해야 정책이 수렴 시 균형 트롯으로 가집니다.

    부상 env 에서는 자연스러운 환측-건측 비대칭이므로 제외합니다.
    """
    contact_by_foot = _foot_force_ema(env, sensor_name=sensor_name, use_z_only=use_z_only, ema_alpha=ema_alpha)
    peg_leg_idx = _peg_leg_index_per_env(env)
    is_normal = peg_leg_idx < 0

    diag1 = contact_by_foot[:, 0] + contact_by_foot[:, 3]  # FL + RR
    diag2 = contact_by_foot[:, 1] + contact_by_foot[:, 2]  # FR + RL
    asym = torch.abs(diag1 - diag2)

    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[is_normal] = asym[is_normal] * _step_ramp(env, ramp_start_steps, ramp_duration_steps)
    return penalty


def penalty_pain(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "contact_forces",
    failure_force_threshold: float = 60.0,
    pain_scale: float = 0.08,
    max_exp_argument: float = 8.0,
    max_penalty: float = 200.0,
    base_contact_cost: float = 0.0,
    contact_detect_threshold: float = 1.0,
    
    base_contact_cost_severe_multiplier: float = 1.0,
    base_contact_cost_mild_multiplier: float = 1.0,
    include_calf: bool = True,
    severity_scaled: bool = False,
    severe_splint_length: float = 0.20,
    mild_splint_length: float = 0.30,
    threshold_severe_multiplier: float = 0.80,
    threshold_mild_multiplier: float = 1.15,
    scale_severe_multiplier: float = 1.25,
    scale_mild_multiplier: float = 0.85,
) -> torch.Tensor:
    """Nociceptor-inspired pain penalty — implements paper equation (4).

    Paper formulation (Section 4.4):
        Cpain(Fz) = Pbase × 1[contact] + max(0, exp(α(Fz − Fth)) − 1)

    where
        Pbase  = base_contact_cost         (paper default 0.05)
        Fth    = failure_force_threshold   (paper default 10.0 N)
        α      = pain_scale                (paper default 2.0)

    Biological accuracy:
        • Sub-threshold (Fz < Fth): only Pbase penalty per contact step.
          Mirrors the low-level baseline firing of biological nociceptors
          during non-damaging mechanical contact.
        • Supra-threshold (Fz > Fth): exponential penalty that becomes
          rapidly intolerable. Mirrors the A-δ and C-fibre burst firing
          triggered by tissue-damaging loads — the same mechanism that
          drives antalgic gait in injured animals.
        • The gradient dCpain/dFz = α·exp(α(Fz−Fth)) is continuous and
          zero at threshold, removing the discontinuity of a hard barrier.

    Severity scaling (when severity_scaled=True):
        Shorter/more-damaged splints receive a lower Fth and higher α,
        reflecting the biological observation that injured tissue has a
        reduced mechanical pain threshold (peripheral sensitisation).
        base_contact_cost is also severity-scaled so that severe splints
        carry higher baseline contact cost, creating the severity trend
        required by the paper's biomechanical comparison protocol.

    Normal-gait symmetry and mirror injury symmetry:
        This function applies only to the designated injured leg.
        Because FL/FR are treated with identical parameter values,
        the injury response is automatically mirror-symmetric, provided
        the healthy baseline gait (Phase 1) has no left-right bias.

    Leg force aggregation:
        leg_force = |Fz(foot)| [+ |Fz(calf)| if include_calf]
        Including calf naturally prevents knee-walking exploits.
        부목 모델 v2: 부목 끝단({leg}_splint) 접촉은 여기 포함하지 않습니다 —
        부목으로 딛는 것은 정상적인 하중 경로(무통)이고, 들려 있어야 할 발이나
        calf 가 지면에 닿는 것만 통증입니다.

    Args:
        failure_force_threshold: Fth — pain onset threshold (N).
            Below this value only Pbase applies (sub-threshold).
        pain_scale: α — exponential steepness above threshold.
            Higher values create a sharper nociceptor-like response.
        base_contact_cost: Pbase — per-contact-step baseline penalty.
            Discourages persistent stance at sub-threshold force.
        base_contact_cost_severe/mild_multiplier: severity scaling of
            Pbase. Severe splints carry higher contact cost, driving
            the severity trend in force reduction.
        
        include_calf: if True, add calf contact force to leg_force
            to prevent the knee-walking exploit.
    """
    _ = asset_cfg
    contact_by_foot, _ = _foot_force_tensor(env, sensor_name=sensor_name, use_z_only=True)
    if include_calf:
        calf_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        contact_by_calf, _ = _link_force_tensor(
            env,
            sensor_name=sensor_name,
            link_name_candidates=calf_names,
            use_z_only=True,
        )
    else:
        contact_by_calf = torch.zeros_like(contact_by_foot)

    peg_leg_idx = _peg_leg_index_per_env(env)

    penalty = torch.zeros(env.num_envs, device=env.device)
    threshold = float(failure_force_threshold)
    scale = float(pain_scale)
    base_cost = float(base_contact_cost)
    detect_th = float(contact_detect_threshold)

    for leg in range(4):
        mask = peg_leg_idx == leg

        if not mask.any():
            continue

        leg_force = (
            contact_by_foot[mask, leg]
            + contact_by_calf[mask, leg]
        )

        if base_cost > 0.0:
            is_contact = (
                leg_force > detect_th
            ).float()

            penalty[mask] += base_cost * is_contact

        overload = torch.clamp(
            leg_force - threshold,
            min=0.0,
        )

        exp_arg = torch.clamp(
            scale * overload,
            min=0.0,
            max=float(max_exp_argument),
        )

        penalty[mask] += torch.clamp(
            torch.expm1(exp_arg),
            max=float(max_penalty),
        )
    return penalty


def penalize_base_height_floor(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    height_floor: float = 0.32,
) -> torch.Tensor:
    """One-sided anti-collapse floor: steep squared penalty for trunk world-height
    BELOW ``height_floor`` only (zero above). Unlike base_height_l2 (two-sided,
    weak), this strongly forbids the trunk from sinking toward the ground while
    NOT penalising a body that stays high. Used to stop the policy from loading a
    SHORT peg by collapsing into a deep squat (root_too_low) — it must instead
    keep the body up; loading then decreases naturally with injury severity. Flat
    terrain only (uses world z directly)."""
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    return torch.square(torch.clamp(float(height_floor) - h, min=0.0))


def _splint_severity_alpha(
    env: "ManagerBasedRLEnv",
    severe_splint_length: float,
    mild_splint_length: float,
) -> torch.Tensor:
    """Return 0 at severe_splint_length and 1 at mild_splint_length.

    부목 모델 v2 에서는 severe(긴 부목, nominal leg reach 에서 멂)가 mild(짧은
    부목)보다 수치상 클 수 있으므로 부호 있는 분모로 양방향을 지원합니다.
    """
    splint_length = getattr(env, "_peg_leg_splint_length", None)
    if splint_length is None:
        return torch.ones(env.num_envs, device=env.device)

    lo = float(severe_splint_length)
    hi = float(mild_splint_length)
    denom = hi - lo
    if abs(denom) < 1e-6:
        return torch.ones(env.num_envs, device=env.device)
    return torch.clamp((splint_length.to(env.device) - lo) / denom, min=0.0, max=1.0)

def penalize_injured_limb_force_nonuse(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    use_z_only: bool = True,
    ema_alpha: float = 0.995,
    severe_splint_length: float = 0.20,
    mild_splint_length: float = 0.30,
    min_force_severe: float = 2.0,
    min_force_mild: float = 11.0,
    front_leg_multiplier: float = 1.15,
    rear_leg_multiplier: float = 1.0,
    ramp_start_steps: int = 1000,
    ramp_duration_steps: int = 8000,
    include_calf: bool = True,
) -> torch.Tensor:
    """Penalize complete injured-limb non-use while still allowing unloading.

    This is the minimal "use the limb" term: pain alone makes the optimal policy
    abandon the injured limb and walk as a tripod (non-use is the robust global
    optimum because any contact costs Pbase and a quadruped is statically stable
    on three legs). This term enforces only the *premise* of the splint problem —
    the damaged limb must still bear a minimum residual load (a viability/use
    floor) — and leaves *how* it is used (force magnitude, duty, asymmetry, CoM
    shift) to emerge from the pain penalty.

    부목 모델 v2: 부상 다리의 하중 경로는 부목 끝단({leg}_splint 링크)이므로
    하중은 splint 접촉력으로 측정합니다. include_calf=True 는 발+calf 접촉력도
    하중에 합산하는 구모델 호환 옵션인데, v2 에서는 발/calf 접촉이 통증
    (penalty_pain) 대상이므로 하중으로 이중 계상하지 않도록 False 를 권장합니다.
    """
    contact_by_leg = _splint_force_tensor(env, sensor_name=sensor_name, use_z_only=use_z_only)
    if include_calf:
        contact_by_foot, _ = _foot_force_tensor(env, sensor_name=sensor_name, use_z_only=use_z_only)
        calf_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        contact_by_calf, _ = _link_force_tensor(
            env,
            sensor_name=sensor_name,
            link_name_candidates=calf_names,
            use_z_only=use_z_only,
        )
        contact_by_leg = contact_by_leg + contact_by_foot + contact_by_calf
    peg_leg_idx = _peg_leg_index_per_env(env)

    injured_force = torch.zeros(env.num_envs, device=env.device)
    for leg in range(4):
        mask = peg_leg_idx == leg
        if mask.any():
            injured_force[mask] = contact_by_leg[mask, leg]

    alpha = float(max(0.0, min(0.9999, ema_alpha)))
    ema = getattr(env, "_go1_injured_force_ema", None)
    prev_idx = getattr(env, "_go1_injured_force_ema_idx", None)
    prev_splint = getattr(env, "_go1_injured_force_ema_splint", None)
    splint_length = getattr(env, "_peg_leg_splint_length", None)
    if ema is None or ema.shape != injured_force.shape:
        ema = injured_force.detach().clone()
    else:
        changed = prev_idx is None or prev_idx.shape != peg_leg_idx.shape
        if changed:
            changed_mask = torch.ones_like(peg_leg_idx, dtype=torch.bool)
        else:
            changed_mask = prev_idx.to(env.device) != peg_leg_idx
            if splint_length is not None:
                if prev_splint is None or prev_splint.shape != splint_length.shape:
                    changed_mask = torch.ones_like(changed_mask, dtype=torch.bool)
                else:
                    changed_mask = changed_mask | (
                        torch.abs(prev_splint.to(env.device) - splint_length.to(env.device)) > 1e-4
                    )
        if changed_mask.any():
            ema[changed_mask] = injured_force.detach()[changed_mask]
        ema.mul_(alpha).add_(injured_force.detach(), alpha=1.0 - alpha)
    env._go1_injured_force_ema = ema
    env._go1_injured_force_ema_idx = peg_leg_idx.detach().clone()
    if splint_length is not None:
        env._go1_injured_force_ema_splint = splint_length.detach().clone()

    severity_alpha = _splint_severity_alpha(env, severe_splint_length, mild_splint_length)
    target = float(min_force_severe) + (
        float(min_force_mild) - float(min_force_severe)
    ) * severity_alpha
    front_mask = (peg_leg_idx == 0) | (peg_leg_idx == 1)
    target = torch.where(
        front_mask,
        target * float(front_leg_multiplier),
        target * float(rear_leg_multiplier),
    )

    is_injured = peg_leg_idx >= 0
    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[is_injured] = torch.clamp(target[is_injured] - ema[is_injured], min=0.0)
    return penalty * _step_ramp(env, ramp_start_steps, ramp_duration_steps)
    

def penalize_injured_limb_load_duty_nonuse(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    load_contact_threshold: float = 10.0,
    use_z_only: bool = True,
    ema_alpha: float = 0.995,
    severe_splint_length: float = 0.20,
    mild_splint_length: float = 0.30,
    min_duty_severe: float = 0.05,
    min_duty_mild: float = 0.28,
    front_leg_multiplier: float = 1.10,
    rear_leg_multiplier: float = 1.0,
    ramp_start_steps: int = 1000,
    ramp_duration_steps: int = 8000,
) -> torch.Tensor:
    """Penalize near-zero load-bearing duty on the injured limb.

    This is a weak regularizer for the analysis metric, not a target gait
    template. It only activates when the time-averaged load-bearing duty falls
    below a severity-aware floor.

    부목 모델 v2: 유효 하중 접지는 부목 끝단({leg}_splint) 접촉력으로 판정합니다.
    """
    contact_by_splint = _splint_force_tensor(env, sensor_name=sensor_name, use_z_only=use_z_only)
    peg_leg_idx = _peg_leg_index_per_env(env)

    injured_contact = torch.zeros(env.num_envs, device=env.device)
    for leg in range(4):
        mask = peg_leg_idx == leg
        if mask.any():
            injured_contact[mask] = (
                contact_by_splint[mask, leg] > float(load_contact_threshold)
            ).float()

    alpha = float(max(0.0, min(0.9999, ema_alpha)))
    ema = getattr(env, "_go1_injured_load_duty_ema", None)
    prev_idx = getattr(env, "_go1_injured_load_duty_ema_idx", None)
    prev_splint = getattr(env, "_go1_injured_load_duty_ema_splint", None)
    splint_length = getattr(env, "_peg_leg_splint_length", None)
    if ema is None or ema.shape != injured_contact.shape:
        ema = injured_contact.detach().clone()
    else:
        changed = prev_idx is None or prev_idx.shape != peg_leg_idx.shape
        if changed:
            changed_mask = torch.ones_like(peg_leg_idx, dtype=torch.bool)
        else:
            changed_mask = prev_idx.to(env.device) != peg_leg_idx
            if splint_length is not None:
                if prev_splint is None or prev_splint.shape != splint_length.shape:
                    changed_mask = torch.ones_like(changed_mask, dtype=torch.bool)
                else:
                    changed_mask = changed_mask | (
                        torch.abs(prev_splint.to(env.device) - splint_length.to(env.device)) > 1e-4
                    )
        if changed_mask.any():
            ema[changed_mask] = injured_contact.detach()[changed_mask]
        ema.mul_(alpha).add_(injured_contact.detach(), alpha=1.0 - alpha)
    env._go1_injured_load_duty_ema = ema
    env._go1_injured_load_duty_ema_idx = peg_leg_idx.detach().clone()
    if splint_length is not None:
        env._go1_injured_load_duty_ema_splint = splint_length.detach().clone()

    severity_alpha = _splint_severity_alpha(env, severe_splint_length, mild_splint_length)
    target = float(min_duty_severe) + (
        float(min_duty_mild) - float(min_duty_severe)
    ) * severity_alpha
    front_mask = (peg_leg_idx == 0) | (peg_leg_idx == 1)
    target = torch.where(
        front_mask,
        torch.clamp(target * float(front_leg_multiplier), max=0.5),
        torch.clamp(target * float(rear_leg_multiplier), max=0.5),
    )

    is_injured = peg_leg_idx >= 0
    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[is_injured] = torch.clamp(target[is_injured] - ema[is_injured], min=0.0)
    return penalty * _step_ramp(env, ramp_start_steps, ramp_duration_steps)

def penalize_joint_mirror_asymmetry(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """§4.7 symmetry-encouraging baseline penalty: ||q - M(q)||^2.

    M is the left/right joint mirror (FL↔FR, RL↔RR with hip-abduction sign flip).
    Applied to ALL envs (incl. injured) to force a left-right symmetric joint
    configuration even under injury — the 'symmetry-encouraging' paradigm whose
    forced symmetry is expected to FAIL the injured-animal biomechanical match
    (it suppresses the antalgic asymmetry). The Go1 default pose (hip ±0.1) is
    already mirror-symmetric so a symmetric stance incurs zero penalty.

    부목 관절 4개가 추가되어 num_joints=16 이므로, mirror_joint_tensor 가
    가정하는 per-TYPE 12관절 순서를 이름으로 명시 리졸브해 선택합니다.
    """
    from .mirror import mirror_joint_tensor

    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = list(asset.data.joint_names)
    # mirror.py 의 JOINT_MIRROR_IDX 와 동일한 per-TYPE 순서 (hips, thighs, calves)
    leg_ids = [
        joint_names.index(n)
        for n in (*HIP_JOINT_NAMES, *THIGH_JOINT_NAMES, *CALF_JOINT_NAMES)
        if n in joint_names
    ]
    q = asset.data.joint_pos[:, leg_ids]
    qm = mirror_joint_tensor(q)
    return torch.sum((q - qm) ** 2, dim=-1)
