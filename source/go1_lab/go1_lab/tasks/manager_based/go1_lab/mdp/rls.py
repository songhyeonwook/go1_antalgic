"""부목 길이 L 의 온라인 RLS 추정기 (착지 기하구속 + 토크 게이트).

test/analyze_dump.py 로 오프라인 검증된 알고리즘의 torch 벡터화 이식본:
  P2-pain8 롤아웃에서 median |L̂−L| 0.2 mm, 96/96 에피소드 커버리지.

측정 모델 (착지 순간의 기하구속, base 프레임):
    끝단 깊이 ĝ·(a + b·L) = 기준 발 깊이 − c0
    →  (ĝ·b)·L = ĝ·(p_foot − a) − c0        (L 에 선형, 스칼라)
  ĝ: 중력 하향 단위벡터 (projected_gravity — IMU 신호)
  a, b: 부목 끝단 p_tip = a + b·L 의 FK 계수 (관절 인코더만 사용)
  c0: 끝단/발 접촉구 반경 차 (설계 상수)

게이트 (실기 신호만):
  - 부목 접지: 부상 다리 √(τ_hip² + τ_thigh²) > torque_gate_nm
    (부목이 thigh 에 강결합 → 접지 하중이 근위 관절 토크로 전달)
  - 기준 발: 정상 다리 중 최대 접촉력 발 > foot_stance_n
    ⚠️ sim 에서는 접촉센서 사용. 실기에서는 표준 프로프리오셉티브 발 스탠스
    추정으로 대체 (사족보행 상태추정의 기성 기법 — 부목 감지가 신규 부분)
  - innovation 게이트: |y − coef·L̂| > innovation_gate_m 샘플 기각
  - 서브샘플: episode step % update_stride == 0 (상관 노이즈 완화)

상태: env._rls_L_hat (N,), env._rls_P (N,) — observations.rls_estimate 가
정규화해 policy 관측 [L̂_norm, √P_norm] 으로 노출한다.
리셋: events.randomize_peg_leg_actuation 이 reset_rls() 호출 → prior 복귀.

순환 import 방지를 위해 이 모듈은 mdp 내 다른 모듈을 import 하지 않는다.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from go1_lab.splint import LEGS, SPLINT_LATERAL, SPLINT_PITCH, SPLINT_TIP_RADIUS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── prior / 정규화 (rls_estimate 관측과 공유) ──────────────────────────
RLS_L_PRIOR = 0.39   # L 샘플 범위 [0.33, 0.45] 의 중앙 (m)
RLS_L_SCALE = 0.06   # 정규화 스케일 = 범위 반폭 (m)
RLS_P0 = RLS_L_SCALE ** 2

# ── Go1 기구 상수 (URDF; test/analyze_dump.py 에서 FK 오차 0.0 mm 검증) ──
_HIP_X, _HIP_Y = 0.1881, 0.04675
_THIGH_Y = 0.08
_THIGH_LEN = 0.213
_CALF_LEN = 0.213
_LEG_SIDE = (+1.0, -1.0, +1.0, -1.0)    # FL, FR, RL, RR
_LEG_FRONT = (+1.0, +1.0, -1.0, -1.0)
_FOOT_RADIUS = 0.02                     # Go1 발 접촉구 반경 (URDF)
# 접촉구 반경 차 c0 = 부목 끝단 − 발
_C0 = SPLINT_TIP_RADIUS - _FOOT_RADIUS

_HIP_JOINTS = [f"{leg}_hip_joint" for leg in LEGS]
_THIGH_JOINTS = [f"{leg}_thigh_joint" for leg in LEGS]
_CALF_JOINTS = [f"{leg}_calf_joint" for leg in LEGS]
_FOOT_BODIES = [f"{leg}_foot" for leg in LEGS]


def ensure_rls_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "_rls_L_hat"):
        env._rls_L_hat = torch.full(
            (env.num_envs,), RLS_L_PRIOR, device=env.device, dtype=torch.float32
        )
    if not hasattr(env, "_rls_P"):
        env._rls_P = torch.full(
            (env.num_envs,), RLS_P0, device=env.device, dtype=torch.float32
        )


def reset_rls(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """리셋된 env 의 추정 상태를 prior 로 되돌린다 (에피소드마다 L 재추첨)."""
    ensure_rls_buffers(env)
    env._rls_L_hat[env_ids] = RLS_L_PRIOR
    env._rls_P[env_ids] = RLS_P0


def _rx(t: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(t), torch.sin(t)
    o, z = torch.ones_like(t), torch.zeros_like(t)
    return torch.stack([
        torch.stack([o, z, z], -1),
        torch.stack([z, c, -s], -1),
        torch.stack([z, s, c], -1),
    ], -2)


def _ry(t: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(t), torch.sin(t)
    o, z = torch.ones_like(t), torch.zeros_like(t)
    return torch.stack([
        torch.stack([c, z, s], -1),
        torch.stack([z, o, z], -1),
        torch.stack([-s, z, c], -1),
    ], -2)


def _ensure_rls_layout(env: "ManagerBasedRLEnv", robot) -> None:
    """관절/바디 인덱스와 상수 텐서를 1회 캐싱한다."""
    if getattr(env, "_rls_layout_ready", False):
        return
    jn = list(robot.data.joint_names)
    dev = env.device
    env._rls_hip_j = torch.tensor([jn.index(n) for n in _HIP_JOINTS], device=dev)
    env._rls_thigh_j = torch.tensor([jn.index(n) for n in _THIGH_JOINTS], device=dev)
    env._rls_calf_j = torch.tensor([jn.index(n) for n in _CALF_JOINTS], device=dev)
    contacts = env.scene["contact_forces"]
    bn = list(contacts.body_names)
    env._rls_foot_b = torch.tensor([bn.index(n) for n in _FOOT_BODIES], device=dev)
    env._rls_side = torch.tensor(_LEG_SIDE, device=dev)
    env._rls_front = torch.tensor(_LEG_FRONT, device=dev)
    # 부목 축 방향 (thigh 프레임, LEGS 공통): -Z 에서 +X 로 SPLINT_PITCH 기움
    env._rls_dir = torch.tensor(
        [math.sin(SPLINT_PITCH), 0.0, -math.cos(SPLINT_PITCH)], device=dev
    )
    env._rls_layout_ready = True


def update_rls(env: "ManagerBasedRLEnv", params: dict) -> None:
    """매 env-step 호출: 게이트 통과 샘플로 (L̂, P) 를 갱신한다.

    비용: 4-leg 배치 FK + 원소별 스칼라 RLS — 전부 벡터화, per-env 루프 없음.
    """
    if not hasattr(env, "_peg_leg_lock_active"):
        return
    ensure_rls_buffers(env)
    robot = env.scene["robot"]
    _ensure_rls_layout(env, robot)

    stride = int(params["update_stride"])
    leg_idx = env._peg_leg_index                       # (N,) -1=정상
    candidate = (
        env._peg_leg_lock_active
        & (leg_idx >= 0)
        & (env.episode_length_buf > 0)
        & (env.episode_length_buf % stride == 0)
    )
    if not bool(candidate.any()):
        return

    dev = env.device
    N = env.num_envs
    leg_safe = leg_idx.clamp(min=0)                    # gather 용 (-1 → 0)
    ar = torch.arange(N, device=dev)

    # ── (1) 토크 게이트: 부상 다리 √(τ_hip² + τ_thigh²) ──
    tau = robot.data.applied_torque                    # (N, num_joints)
    tau_hip = tau.gather(1, env._rls_hip_j[leg_safe].unsqueeze(1)).squeeze(1)
    tau_thigh = tau.gather(1, env._rls_thigh_j[leg_safe].unsqueeze(1)).squeeze(1)
    splint_stance = torch.hypot(tau_hip, tau_thigh) > float(params["torque_gate_nm"])

    # ── (2) 기준 발: 정상 다리 중 최대 접촉력 (sim: 접촉센서) ──
    contacts = env.scene["contact_forces"]
    fz = contacts.data.net_forces_w[:, env._rls_foot_b, 2].abs()   # (N, 4)
    fz = fz.scatter(1, leg_safe.unsqueeze(1), 0.0)     # 부상 다리 제외
    ref_force, ref_leg = fz.max(dim=1)
    foot_ok = ref_force > float(params["foot_stance_n"])

    active = candidate & splint_stance & foot_ok
    if not bool(active.any()):
        return

    # ── (3) FK (base 프레임, 4-leg 배치) ──
    jp = robot.data.joint_pos
    q_hip = jp[:, env._rls_hip_j]                      # (N, 4)
    q_thigh = jp[:, env._rls_thigh_j]
    q_calf = jp[:, env._rls_calf_j]

    R1 = _rx(q_hip)                                    # (N, 4, 3, 3)
    R2 = R1 @ _ry(q_thigh)
    R3 = R2 @ _ry(q_calf)

    p_hip = torch.stack([
        env._rls_front.expand(N, 4) * _HIP_X,
        env._rls_side.expand(N, 4) * _HIP_Y,
        torch.zeros(N, 4, device=dev),
    ], -1)                                             # (N, 4, 3)
    off_thigh = torch.stack([
        torch.zeros(N, 4, device=dev),
        env._rls_side.expand(N, 4) * _THIGH_Y,
        torch.zeros(N, 4, device=dev),
    ], -1)
    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    p_thigh = p_hip + (R1 @ off_thigh.unsqueeze(-1)).squeeze(-1)
    p_calf = p_thigh + _THIGH_LEN * (R2 @ down.expand(N, 4, 3).unsqueeze(-1)).squeeze(-1)
    p_foot = p_calf + _CALF_LEN * (R3 @ down.expand(N, 4, 3).unsqueeze(-1)).squeeze(-1)

    # 부상 다리의 부목 계수: p_tip = a + b·L
    anchor = torch.stack([
        torch.zeros(N, 4, device=dev),
        env._rls_side.expand(N, 4) * SPLINT_LATERAL,
        torch.zeros(N, 4, device=dev),
    ], -1)
    a_all = p_thigh + (R2 @ anchor.unsqueeze(-1)).squeeze(-1)          # (N, 4, 3)
    b_all = (R2 @ env._rls_dir.expand(N, 4, 3).unsqueeze(-1)).squeeze(-1)
    a = a_all[ar, leg_safe]                            # (N, 3)
    b = b_all[ar, leg_safe]
    p_ref = p_foot[ar, ref_leg]                        # 기준 발 위치

    # ── (4) 스칼라 측정 갱신 ──
    g = robot.data.projected_gravity_b
    g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    coef = (g * b).sum(-1)                             # ĝ·b (하향 성분, >0)
    y = (g * (p_ref - a)).sum(-1) - _C0

    L_hat, P = env._rls_L_hat, env._rls_P
    innov = y - coef * L_hat
    valid = (
        active
        & (coef > float(params["min_axis_coef"]))
        & (innov.abs() < float(params["innovation_gate_m"]))
    )
    if not bool(valid.any()):
        return

    r = float(params["meas_noise_std"]) ** 2
    K = P * coef / (coef * coef * P + r)
    L_hat[valid] = L_hat[valid] + (K * innov)[valid]
    P[valid] = (P * (1.0 - K * coef))[valid]
