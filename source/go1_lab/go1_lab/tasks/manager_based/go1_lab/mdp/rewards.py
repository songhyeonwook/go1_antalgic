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
    #각 env의 고장 다리 인덱스(-1, 0..3)를 반환합니다.
    
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
        print(f"[WARN] ContactSensor '{sensor_name}' not found in scene. Returning zeros.")
        return torch.zeros((env.num_envs, 4), device=env.device), [None, None, None, None]

    contact_forces_data = contact_sensor.data.net_forces_w
    if contact_forces_data is None:
        print(f"[WARN] ContactSensor '{sensor_name}' has no net_forces_w data. Returning zeros.")
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


def _body_weight_tensor(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    """env 별 로봇 체중 mg [N]. shape: (num_envs,).

    질량 랜덤화(add_base_mass, front_payload)는 startup 이벤트라 학습 중 바뀌지
    않으므로 처음 한 번 계산해 캐시한다. 부목 링크의 질량 은닉(tiny) 은
    체중에 무시할 수준이므로 별도 보정하지 않는다.
    """
    cached = getattr(env, "_go1_body_weight_n", None)
    if cached is not None and cached.shape[0] == env.num_envs:
        return cached
    robot: Articulation = env.scene[asset_name]
    masses = robot.root_physx_view.get_masses()  # (num_envs, num_bodies), CPU
    g = 9.81
    try:
        g = abs(float(env.sim.cfg.gravity[2])) or 9.81
    except Exception:
        pass
    mg = (masses.sum(dim=1).to(device=env.device, dtype=torch.float32)) * g
    env._go1_body_weight_n = mg
    return mg


def penalty_pain(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "contact_forces",
    threshold_bw: float = 0.01,
    scale_bw: float = 0.25,
    exponent: float = 1.0,
    splint_transmission: float = 0.5,
    include_calf: bool = True,
    include_splint: bool = True,
    body_weight_n: float | None = None,
) -> torch.Tensor:
    """부상 다리 통각(nociceptor) 비용 C_pain ∈ [0, 1].

    유효 통증 하중 (eq. fpain):
      F_pain = F_z^foot + F_z^calf + η · F_z^splint
        η = splint_transmission ∈ [0, 1] — 부목을 거쳐 부상 조직에 전달되는
        하중 비율. 부목 모델에서는 발/calf 가 기구적으로 들려 있어
        F_foot = F_calf = 0 이고 F_pain = η·F_splint 로 환원된다.

    통각 비용 (eq. cpain):
      C_pain = min( ( [F_pain − θ]_+ / ρ )^n , 1 )
        θ = threshold_bw · mg   (기본 0.01 mg — 무신호 역치)
        ρ = scale_bw · mg       (기본 mg/4 — 정상 다리 1개의 정적 하중 분담)
        n = exponent            (기본 1)
      정규화: 부상 다리가 정상 다리의 정적 하중 분담(mg/4)을 그대로 지면 C_pain = 1.

    역치·단조 증가·포화의 세 정성적 성질만 만족하는 최소 스칼라 비용이며,
    W_pain 과 Δt 는 reward manager (weight · dt) 가 곱한다.
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
    if include_splint:
        contact_by_splint = _splint_force_tensor(
            env, sensor_name=sensor_name, use_z_only=True
        )
    else:
        contact_by_splint = torch.zeros_like(contact_by_foot)

    if body_weight_n is None:
        mg = _body_weight_tensor(env)
    else:
        mg = torch.full((env.num_envs,), float(body_weight_n), device=env.device)
    theta = float(threshold_bw) * mg
    rho = torch.clamp(float(scale_bw) * mg, min=1e-6)
    n = float(exponent)
    eta = float(splint_transmission)

    peg_leg_idx = _peg_leg_index_per_env(env)
    penalty = torch.zeros(env.num_envs, device=env.device)

    for leg in range(4):
        mask = peg_leg_idx == leg
        if not mask.any():
            continue

        leg_force = (
            contact_by_foot[mask, leg]
            + contact_by_calf[mask, leg]
            + eta * contact_by_splint[mask, leg]
        )
        overload = torch.clamp(leg_force - theta[mask], min=0.0) / rho[mask]
        if n != 1.0:
            overload = overload.pow(n)
        penalty[mask] = torch.clamp(overload, max=1.0)
    return penalty


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
    diff2 = (q - qm) ** 2                                # (N, 12) per-TYPE 순서

    # 잠긴 calf 는 action 이 마스킹된 비제어 관절이다. 페널티에 남기면
    # 건강한 미러 무릎을 접는 것만이 페널티를 줄이는 길이 되어(실측:
    # 앞다리 부상 시 양 앞무릎이 접혀 몸통 앞쪽 붕괴, 높이 0.40→0.16 m)
    # baseline 이 불공정해진다. 부상 env 에서는 잠긴 calf 와 그 미러 짝
    # (calf 블록 = 인덱스 8..11, FL↔FR / RL↔RR)의 기여를 제외한다.
    # 부상 다리의 hip/thigh 는 여전히 제어 가능하므로 대칭 대상으로 유지.
    peg_leg_idx = _peg_leg_index_per_env(env)            # (N,) -1=정상
    injured = peg_leg_idx >= 0
    if bool(injured.any()):
        mirror_of = torch.tensor([1, 0, 3, 2], device=env.device)
        k = peg_leg_idx.clamp(min=0)
        rows = torch.arange(env.num_envs, device=env.device)
        mask = torch.ones_like(diff2)
        mask[rows[injured], (8 + k)[injured]] = 0.0
        mask[rows[injured], (8 + mirror_of[k])[injured]] = 0.0
        diff2 = diff2 * mask

    return torch.sum(diff2, dim=-1)


# 발을 끄는 것에 대한 패널티가 아님. 접촉하는 것에 대한 패널티
