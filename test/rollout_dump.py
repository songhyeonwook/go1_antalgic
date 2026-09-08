"""학습된 정책의 롤아웃을 .npz 로 저장하는 덤퍼 (phase 2/3 분석용).

저장된 데이터는 학습 루프 밖에서 다음 분석에 쓴다:
  [1] 오프라인 추정기 bake-off — 같은 로그 위에서 RLS(착지 등식 + 토크 잔차
      게이트) / MLP / LSTM 회귀를 돌려 GT L 대비 수렴 곡선 비교
  [2] 토크 잔차 접촉 감지기 채점 — GT 접촉력(sim 전용)으로 정밀도/재현율 측정
  [3] 보행 형태 분석 — 부목 duty factor, 착지 규칙성 (RLS 등식 공급량)
  [4] L 식별 가능성 — 관절각만으로 L 이 새는지 재확인
  [5] 통증 C_pain 분포 — penalty_pain 이 매 스텝 반환한 원시값 (pain_report.py)

env / agent / 체크포인트 로딩은 test.py / play_result.py 와 같은 utils.eval_common
경로를 쓴다 (phase yaml → ExperimentConfig → 레지스트리 cfg → runner).

사용 (phase 2 종료 후):
    cd /home/shw/go1_lod/test
    PYTHONPATH=/home/shw/go1_lod/source/go1_lab python3 rollout_dump.py \
        --phase_config_path ../scripts/rsl_rl/configs/phase/2/phase2_at.yaml \
        --checkpoint <model_*.pt 경로> \
        --num_envs 40 --steps 2500 --out dumps/p2_balanced.npz

    # 조건(--peg_leg): balanced(기본) = env 블록 1:1:1:1:1 (Normal/FL/FR/RL/RR)
    #                 normal/fl/fr/rl/rr = 단일 조건

⚠️ 평가용 덤프이므로 peg-leg 커리큘럼은 비활성화한다 — 켜두면 step counter가
0 이라 부목 길이가 초기 좁은 범위로 고정되어 L 커버리지가 죽는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
RSL_DIR = SCRIPT_DIR.parent / "scripts" / "rsl_rl"
sys.path.insert(0, str(RSL_DIR))

from utils.eval_common import (  # noqa: E402
    LEGS, add_common_args, build_agent_cfg, build_env_cfg, check_splint_length_arg,
    load_config, log, make_gym_env, make_policy, peg_leg_enabled, print_conditions,
    print_header, resolve_checkpoint, resolve_eval_mode, step_env, wrap_rsl,
)

parser = argparse.ArgumentParser(description="정책 롤아웃을 npz 로 덤프")
add_common_args(parser)
parser.add_argument("--steps", type=int, default=2500, help="50 Hz 기준 2500 step = 50 s")
parser.add_argument("--warmup", type=int, default=100, help="기록 전 버리는 초기 step")
parser.add_argument("--fixed_x", type=float, default=None, help="전진 명령 고정 (기본: 샘플링)")
parser.add_argument("--fixed_mu", type=float, default=None,
                    help="부목 끝단 마찰 고정 (μ 강건성 스윕용, 기본: yaml 범위 샘플링)")
parser.add_argument("--l_obs_fixed", type=float, default=None,
                    help="정책에 주입할 GT L 채널값 고정 [m] — 물리 부목 길이는 그대로 (L 민감도용)")
parser.add_argument("--l_obs_offset", type=float, default=None,
                    help="정책의 GT L 채널에 더할 오프셋 [m] — 물리는 그대로")
parser.add_argument("--out", type=str, default=None, help="저장 경로 (.npz). 기본: dumps/<체크포인트명>.npz")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
check_splint_length_arg(args.splint_length)

if args.fixed_mu is not None and not (0.0 < args.fixed_mu <= 4.0):
    parser.error(f"--fixed_mu 는 (0, 4] 범위여야 합니다: {args.fixed_mu} "
                 "(음수는 PhysX 가 조용히 0 으로 클램프해 라벨-실측 불일치 발생)")

config = load_config(args)
checkpoint_path = resolve_checkpoint(args)
if not peg_leg_enabled(config):
    raise RuntimeError("peg_leg.enabled=true 환경(phase 2/3 yaml)에서만 덤프할 수 있습니다.")
eval_mode = resolve_eval_mode(args, config, default="balanced")
num_envs = args.num_envs if args.num_envs is not None else 40
seed = args.seed if args.seed is not None else config.train.seed
device = args.device

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac Sim 시작 이후 import ──────────────────────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402


def resolve_out_path() -> Path:
    if args.out:
        return Path(args.out).expanduser().resolve()
    if args.l_obs_fixed is not None or args.l_obs_offset is not None:
        tag = (f"fix{args.l_obs_fixed}" if args.l_obs_fixed is not None
               else f"off{args.l_obs_offset:+g}")
        return SCRIPT_DIR / "dumps" / f"lsens_{tag}.npz"
    if args.fixed_mu is not None:
        # μ 스윕 계약: mu_robustness_report.py 가 이 이름을 읽는다.
        # μ 를 파일명에 넣지 않으면 스윕 실행이 서로를 덮어쓴다.
        return SCRIPT_DIR / "dumps" / f"mu_sweep_{args.fixed_mu}.npz"
    return (SCRIPT_DIR / "dumps" /
            f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_{eval_mode}.npz")


def main() -> None:
    out_path = resolve_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = build_env_cfg(
        config, num_envs=num_envs, seed=seed, device=device, eval_mode=eval_mode,
        splint_length=args.splint_length, clean=args.clean,
    )
    agent_cfg = build_agent_cfg(config, seed=seed, device=device)

    peg_event = env_cfg.events.randomize_peg_leg_actuation
    if args.fixed_mu is not None:
        # μ 강건성 평가: 부목 끝단 마찰을 단일 값으로 고정 (DR 범위 밖 외삽 가능)
        peg_event.params["foot_friction_range"] = (args.fixed_mu, args.fixed_mu)

    # 커리큘럼 제거 — 평가에서는 yaml 의 전체 L 범위를 그대로 샘플해야 한다
    if getattr(env_cfg.curriculum, "peg_leg_difficulty", None) is not None:
        env_cfg.curriculum.peg_leg_difficulty = None

    if args.fixed_x is not None:
        ranges = env_cfg.commands.base_velocity.ranges
        ranges.lin_vel_x = (args.fixed_x, args.fixed_x)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)

    print_header(config, checkpoint_path, device, seed, num_envs, eval_mode, env_cfg)
    env = wrap_rsl(make_gym_env(config, env_cfg), agent_cfg)
    base = env.unwrapped
    print_conditions(base)
    _, policy_fn, policy_module = make_policy(env, agent_cfg, config, checkpoint_path, device)

    # ── 인덱스 준비 (이름 기반) ──
    robot = base.scene["robot"]
    contacts = base.scene["contact_forces"]
    joint_names = list(robot.data.joint_names)
    leg_joint_names = [n for n in joint_names
                       if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
    leg_j = [joint_names.index(n) for n in leg_joint_names]
    body_names = list(contacts.body_names)
    splint_b = [body_names.index(f"{leg}_splint") for leg in LEGS]
    foot_b = [body_names.index(f"{leg}_foot") for leg in LEGS]
    calf_b = [body_names.index(f"{leg}_calf") for leg in LEGS]
    # GT body 위치 (FK 구현 검증 전용 — 추정기 입력으로는 사용 금지)
    robot_bodies = list(robot.body_names)
    rb_foot = [robot_bodies.index(f"{leg}_foot") for leg in LEGS]
    rb_splint = [robot_bodies.index(f"{leg}_splint") for leg in LEGS]
    rb_thigh = [robot_bodies.index(f"{leg}_thigh") for leg in LEGS]

    # C_pain: reward manager 의 _step_reward 는 func·weight (dt 제거) 이므로
    # weight 로 나누면 penalty_pain 이 이 스텝에 반환한 원시 C_pain 이 된다.
    # 직접 재호출하지 않는 이유: step 내부 리셋 이후 센서 버퍼가 바뀔 수 있어
    # "리워드가 본 값" 과 어긋난다.
    rm = base.reward_manager
    pain_idx = rm.active_terms.index("penalty_pain") if "penalty_pain" in rm.active_terms else None
    pain_weight = float(rm.get_term_cfg("penalty_pain").weight) if pain_idx is not None else 0.0
    pain_params = ({k: v for k, v in rm.get_term_cfg("penalty_pain").params.items()
                    if isinstance(v, (int, float, bool, str))} if pain_idx is not None else {})
    if pain_idx is None or pain_weight == 0.0:
        log("[WARN] penalty_pain 리워드 항 없음/가중치 0 — pain 은 0 으로 기록")
        pain_idx = None

    N, T = num_envs, args.steps
    rec: dict[str, list] = {k: [] for k in (
        "obs_policy", "obs_privileged", "action",
        "joint_pos", "joint_vel", "applied_torque_leg",
        "root_state", "projected_gravity", "commands",
        "contact_splint", "contact_foot", "contact_calf",
        "gt_leg", "gt_L", "gt_mu", "lock_active", "dones",
        "pos_feet_w", "pos_splint_w", "pos_thigh_w",
        "pain",
    )}

    def grab(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    obs = env.get_observations()
    priv_dim = int(base.obs_buf["privileged_obs"].shape[1])
    log(f"[INFO] 덤프 시작: N={N}, steps={T} (+warmup {args.warmup}), 조건={eval_mode}")
    log(f"[INFO] privileged obs dim={priv_dim}, pain term={'있음' if pain_idx is not None else '없음'} "
        f"(weight={pain_weight})")
    t0 = time.time()

    # privileged 레이아웃: L(1) lin_vel(3) — L 채널은 index 0 (Go1LabPrivilegedObsCfg).
    # 부상 env 마스크는 obs 가 아니라 GT 버퍼(_peg_leg_index)로 잡는다.
    L_OBS_IDX = 0
    _perturb_l = args.l_obs_fixed is not None or args.l_obs_offset is not None

    def perturb(o):
        """정책이 보는 GT L 채널만 조작 (부상 env 한정). 물리·기록 버퍼는 무손상."""
        if not _perturb_l:
            return o
        o = o.clone()
        p = o["privileged_obs"]
        m = base._peg_leg_index >= 0
        if args.l_obs_fixed is not None:
            p[m, L_OBS_IDX] = args.l_obs_fixed
        if args.l_obs_offset is not None:
            p[m, L_OBS_IDX] = p[m, L_OBS_IDX] + args.l_obs_offset
        return o

    with torch.inference_mode():
        for step in range(args.warmup + T):
            obs, _, dones, _, actions = step_env(env, policy_fn, policy_module, perturb(obs))

            if step < args.warmup:
                continue

            f = contacts.data.net_forces_w  # (N, bodies, 3)
            rec["obs_policy"].append(grab(base.obs_buf["policy"]))
            rec["obs_privileged"].append(grab(base.obs_buf["privileged_obs"]))
            rec["action"].append(grab(actions))
            rec["joint_pos"].append(grab(robot.data.joint_pos))
            rec["joint_vel"].append(grab(robot.data.joint_vel))
            rec["applied_torque_leg"].append(grab(robot.data.applied_torque[:, leg_j]))
            rec["root_state"].append(grab(robot.data.root_state_w))
            rec["projected_gravity"].append(grab(robot.data.projected_gravity_b))
            rec["commands"].append(grab(base.command_manager.get_command("base_velocity")))
            rec["contact_splint"].append(grab(f[:, splint_b]))
            rec["contact_foot"].append(grab(f[:, foot_b]))
            rec["contact_calf"].append(grab(f[:, calf_b]))
            rec["pos_feet_w"].append(grab(robot.data.body_pos_w[:, rb_foot]))
            rec["pos_splint_w"].append(grab(robot.data.body_pos_w[:, rb_splint]))
            rec["pos_thigh_w"].append(grab(robot.data.body_pos_w[:, rb_thigh]))
            rec["gt_leg"].append(base._peg_leg_index.detach().cpu().numpy().astype(np.int8))
            rec["gt_L"].append(grab(base._peg_leg_splint_length))
            rec["gt_mu"].append(grab(base._peg_leg_foot_friction))
            rec["lock_active"].append(base._peg_leg_lock_active.detach().cpu().numpy())
            rec["dones"].append(dones.detach().cpu().numpy().astype(bool).reshape(-1))
            if pain_idx is not None:
                rec["pain"].append(grab(rm._step_reward[:, pain_idx] / pain_weight))
            else:
                rec["pain"].append(np.zeros(N, dtype=np.float32))

    arrays = {k: np.stack(v) for k, v in rec.items()}  # (T, N, ...)
    # env 별 체중 mg [N] — penalty_pain 의 θ = threshold_bw·mg, ρ = scale_bw·mg 재계산용
    try:
        from go1_lab.tasks.manager_based.go1_lab.mdp.rewards import _body_weight_tensor
        arrays["body_weight_n"] = _body_weight_tensor(base).detach().cpu().numpy()  # (N,)
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] body_weight_n 계산 실패 ({exc}) — pain_report 는 --body_weight_n 사용")
    meta = {
        "checkpoint": str(checkpoint_path),
        "phase": config.phase,
        "phase_config": str(Path(args.phase_config_path).expanduser().resolve()),
        "condition": eval_mode,
        "num_envs": N,
        "steps": T,
        "warmup": args.warmup,
        "seed": seed,
        "step_dt": float(base.step_dt),
        "fixed_x": args.fixed_x,
        "fixed_mu": args.fixed_mu,
        "l_obs_fixed": args.l_obs_fixed,
        "l_obs_offset": args.l_obs_offset,
        "l_obs_note": "정책 입력의 privileged L 채널만 조작 (부상 env 한정); 물리 부목 길이 = gt_L 은 무손상",
        "joint_names": joint_names,
        "leg_joint_names": leg_joint_names,
        "legs": list(LEGS),
        "contact_body_order": "LEGS 순서 (FL, FR, RL, RR), net_forces_w [N] xyz",
        "gt_leg_convention": "-1=정상, 0=FL, 1=FR, 2=RL, 3=RR",
        "obs_privileged_layout": "L(1) lin_vel(3)",
        "pain_weight": pain_weight,
        "pain_params": pain_params,
        "pain_note": "penalty_pain 이 해당 step 에 반환한 원시 C_pain (가중치·dt 제거, 정상 env 는 0)",
    }
    np.savez_compressed(out_path, **arrays, meta=json.dumps(meta))

    sizes = {k: list(v.shape) for k, v in arrays.items()}
    log(f"[INFO] 저장: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
        f"{time.time() - t0:.0f}s 소요)")
    for k in ("obs_policy", "applied_torque_leg", "contact_splint", "gt_L", "pain"):
        log(f"       {k}: {sizes[k]}")
    inj = arrays["gt_leg"][-1] >= 0
    L_inj = arrays["gt_L"][arrays["gt_leg"] >= 0]
    log(f"[INFO] 마지막 스텝 조건: 부상 {int(inj.sum())}/{N} env, "
        f"L 범위 [{L_inj.min():.3f}, {L_inj.max():.3f}] m")
    if pain_idx is not None:
        p_inj = arrays["pain"][arrays["gt_leg"] >= 0]
        log(f"[INFO] C_pain (부상 env·step): mean={p_inj.mean():.4f}, "
            f"P(>0)={(p_inj > 0).mean():.3f}, max={p_inj.max():.2f}")

    env.close()


if __name__ == "__main__":
    # simulation_app.close() 는 종료 시 세그폴트로 원래 예외를 삼킬 수 있어
    # (실측) 다른 test 스크립트처럼 traceback 출력 후 os._exit 로 끝낸다.
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
