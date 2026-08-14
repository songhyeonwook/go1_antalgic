"""학습된 정책의 롤아웃을 .npz 로 저장하는 덤퍼 (phase 2/3 분석용).

저장된 데이터는 학습 루프 밖에서 다음 분석에 쓴다:
  [1] 오프라인 추정기 bake-off — 같은 로그 위에서 RLS(착지 등식 + 토크 잔차
      게이트) / MLP / LSTM 회귀를 돌려 GT L 대비 수렴 곡선 비교
  [2] 토크 잔차 접촉 감지기 채점 — GT 접촉력(sim 전용)으로 정밀도/재현율 측정
  [3] 보행 형태 분석 — 부목 duty factor, 착지 규칙성 (RLS 등식 공급량)
  [4] L 식별 가능성 — 관절각만으로 L 이 새는지 재확인

사용 (phase 2 종료 후):
    cd /home/shw/go1_lod/test
    PYTHONPATH=/home/shw/go1_lod/source/go1_lab python3 rollout_dump.py \
        --phase 2 --checkpoint <model_*.pt 경로> \
        --num_envs 40 --steps 2500 --out dumps/p2_balanced.npz

    # 조건: balanced(기본) = env_id 고정 1:1:1:1:1 (Normal/FL/FR/RL/RR)
    #       train          = 학습과 동일한 env_fixed 배정 (정상 50%)

⚠️ 평가용 덤프이므로 peg-leg 커리큘럼은 비활성화한다 — 켜두면 step counter가
0 이라 부목 길이가 초기 좁은 범위 [0.33, 0.36] 로 고정되어 L 커버리지가 죽는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
RSL_DIR = SCRIPT_DIR.parent / "scripts" / "rsl_rl"
sys.path.insert(0, str(RSL_DIR))

from utils.config_builder import load_experiment_config  # noqa: E402

parser = argparse.ArgumentParser(description="정책 롤아웃을 npz 로 덤프")
parser.add_argument("--phase", type=int, choices=(2, 3), default=2)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=40)
parser.add_argument("--steps", type=int, default=2500, help="50 Hz 기준 2500 step = 50 s")
parser.add_argument("--warmup", type=int, default=100, help="기록 전 버리는 초기 step")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--condition", choices=("balanced", "train"), default="balanced",
                    help="balanced: env_id 고정 1:1:1:1:1 / train: 학습과 동일(env_fixed)")
parser.add_argument("--fixed_x", type=float, default=None, help="전진 명령 고정 (기본: 샘플링)")
parser.add_argument("--out", type=str, default=None, help="저장 경로 (.npz). 기본: dumps/<체크포인트명>.npz")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True

phase_config_path = RSL_DIR / "configs" / "phase" / f"phase{args.phase}.yaml"
config = load_experiment_config(
    phase_path=phase_config_path,
    common_path=str(RSL_DIR / "configs" / "common.yaml"),
)

sys.argv = [sys.argv[0], *hydra_args,
            "hydra/job_logging=disabled", "hydra.output_subdir=null", "hydra.run.dir=."]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac Sim 시작 이후 import ──────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import go1_lab.tasks  # noqa: F401, E402

LEGS = ("FL", "FR", "RL", "RR")


def patch_rsl_rl_agent_cfg(agent_cfg_dict: dict) -> dict:
    policy_cfg = agent_cfg_dict.get("policy")
    if isinstance(policy_cfg, dict):
        for name in ("actor", "critic", "student", "teacher"):
            if isinstance(policy_cfg.get(name), dict):
                policy_cfg[name].setdefault("class_name", "MLP")
    algorithm_cfg = agent_cfg_dict.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        for key in ("optimizer", "config_class", "share_cnn_encoders"):
            algorithm_cfg.pop(key, None)
    return agent_cfg_dict


@hydra_task_config(config.train.task, config.train.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint 없음: {checkpoint_path}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else SCRIPT_DIR / "dumps" / f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_{args.condition}.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else config.train.seed
    device = config.common.get("device", "cuda:0")

    agent_cfg.seed = seed
    agent_cfg.device = device
    exploration = config.exploration
    agent_cfg.policy.noise_std_type = exploration.noise_std_type
    agent_cfg.policy.init_noise_std = exploration.init_noise_std

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = seed
    env_cfg.apply_environment_settings(
        config.environment.values, int(agent_cfg.num_steps_per_env)
    )

    # ── 평가 조건 설정 ──
    peg_event = env_cfg.events.randomize_peg_leg_actuation
    if peg_event is None:
        raise RuntimeError("peg_leg.enabled=true 환경에서만 덤프할 수 있습니다.")
    if args.condition == "balanced":
        # env_id 고정: env0=Normal, env1..4=FL/FR/RL/RR, env5=Normal, ... (주기 5)
        peg_event.params["target_leg"] = "balanced_env"
        peg_event.params["healthy_slots"] = 1
    # "train" 은 yaml 그대로 (env_fixed, healthy_slots=4)

    # 커리큘럼 제거 — 평가에서는 전체 L 범위 [0.33, 0.45] 를 그대로 샘플해야 한다
    if getattr(env_cfg.curriculum, "peg_leg_difficulty", None) is not None:
        env_cfg.curriculum.peg_leg_difficulty = None

    if args.fixed_x is not None:
        ranges = env_cfg.commands.base_velocity.ranges
        ranges.lin_vel_x = (args.fixed_x, args.fixed_x)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)

    env = gym.make(config.train.task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_cfg_dict = patch_rsl_rl_agent_cfg(agent_cfg.to_dict())
    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env=env, train_cfg=agent_cfg_dict, log_dir=None, device=device)
    runner.load(str(checkpoint_path), load_optimizer=False, map_location=device)
    policy = runner.get_inference_policy(device=base.device)

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

    N, T = args.num_envs, args.steps
    rec: dict[str, list] = {k: [] for k in (
        "obs_policy", "obs_privileged", "action",
        "joint_pos", "joint_vel", "applied_torque_leg",
        "root_state", "projected_gravity", "commands",
        "contact_splint", "contact_foot", "contact_calf",
        "gt_leg", "gt_L", "gt_mu", "lock_active", "dones",
        "pos_feet_w", "pos_splint_w", "pos_thigh_w",
    )}

    def grab(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    obs = env.get_observations()
    print(f"[INFO] 덤프 시작: N={N}, steps={T} (+warmup {args.warmup}), 조건={args.condition}")
    print(f"[INFO] 체크포인트: {checkpoint_path}")
    t0 = time.time()

    with torch.inference_mode():
        for step in range(args.warmup + T):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if getattr(runner.alg.policy, "is_recurrent", False):
                runner.alg.policy.reset(dones)

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

    arrays = {k: np.stack(v) for k, v in rec.items()}  # (T, N, ...)
    meta = {
        "checkpoint": str(checkpoint_path),
        "phase": args.phase,
        "condition": args.condition,
        "num_envs": N,
        "steps": T,
        "warmup": args.warmup,
        "seed": seed,
        "step_dt": float(base.step_dt),
        "fixed_x": args.fixed_x,
        "joint_names": joint_names,
        "leg_joint_names": leg_joint_names,
        "legs": list(LEGS),
        "contact_body_order": "LEGS 순서 (FL, FR, RL, RR), net_forces_w [N] xyz",
        "gt_leg_convention": "-1=정상, 0=FL, 1=FR, 2=RL, 3=RR",
        "obs_policy_layout": "ang_vel(3) gravity(3) cmd(3) jpos(12) jvel(12) act(12) calf_nom(4) rls(2)",
        "obs_privileged_layout": "onehot(5) L(1) mu(1) lin_vel(3)",
    }
    np.savez_compressed(out_path, **arrays, meta=json.dumps(meta))

    sizes = {k: list(v.shape) for k, v in arrays.items()}
    print(f"[INFO] 저장: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s 소요)")
    for k in ("obs_policy", "applied_torque_leg", "contact_splint", "gt_L"):
        print(f"       {k}: {sizes[k]}")
    inj = arrays["gt_leg"][-1] >= 0
    print(f"[INFO] 마지막 스텝 조건: 부상 {int(inj.sum())}/{N} env, "
          f"L 범위 [{arrays['gt_L'][arrays['gt_L'] > 0].min():.3f}, "
          f"{arrays['gt_L'].max():.3f}] m")

    env.close()


if __name__ == "__main__":
    # simulation_app.close() 는 종료 시 세그폴트로 원래 예외를 삼킬 수 있어
    # (실측) 다른 test 스크립트처럼 traceback 출력 후 os._exit 로 끝낸다.
    import os
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
