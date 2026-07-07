#!/usr/bin/env python3
"""Student 정책 분석: (1) 건강 vs 부상 충격량 비교  (2) LSTM 부상 파라미터 추정 정확도.

Usage:
  GO1_PHASE=student python3 analyze_student.py \
    --checkpoint <path_to_student_model.pt> \
    --num_envs 30 --steps 1000
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

from isaaclab.app import AppLauncher

import cli_args
from peg_leg_action_wrapper import PegLegActionMaskWrapper

parser = argparse.ArgumentParser(
    description="Analyze policy: shock comparison + LSTM estimation."
)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--task", type=str, default="Template-Go1-Lab-v0")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument(
    "--flat",
    action="store_true",
    default=False,
    help="Use flat terrain for cleaner comparison.",
)
parser.add_argument("--contact_sensor", type=str, default="contact_forces")
parser.add_argument("--contact_threshold", type=float, default=1.0)
parser.add_argument(
    "--load_contact_threshold",
    type=float,
    default=10.0,
    help=(
        "Threshold for load-bearing duty factor. The lower contact_threshold "
        "detects toe-touch/drag contact; this threshold detects meaningful support."
    ),
)
parser.add_argument(
    "--target_vx",
    type=float,
    default=1.0,
    help="Commanded forward velocity (m/s) for analysis. Go1 natural trot: 0.7-1.0 m/s. "
    "Must match the value used in analyze_healthy.py for consistent Phase1↔Phase3 comparison.",
)
parser.add_argument(
    "--contact_use_z_only",
    action="store_true",
    default=False,
    help="Use |Fz| (world z) per foot instead of ||F|| for contact force tables and duty factor.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--balance_conditions",
    action="store_true",
    default=False,
    help="Equalize per-condition sample counts (Normal/FL/FR/RL/RR) for fair comparison.",
)
parser.add_argument(
    "--balanced_envs",
    "--balance",
    dest="balanced_envs",
    action="store_true",
    default=False,
    help=(
        "Evaluate with environment-level 1:1:1:1:1 assignment "
        "(Normal/FL/FR/RL/RR). Also disables peg-leg curriculum."
    ),
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument(
    "--metrics_json",
    type=str,
    default=None,
    help="Optional path for machine-readable antalgic validation metrics.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument(
    "--pretrained_task",
    type=str,
    default=None,
    help=(
        "Task id used only for resolving Isaac Lab's published pretrained checkpoint. "
        "Useful when evaluating a published policy inside a custom analysis env."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.balanced_envs:
    os.environ["GO1_EVAL_MODE"] = "balanced"
    os.environ["GO1_USE_PEG_LEG_CURRICULUM"] = "0"
    os.environ.setdefault("GO1_SPLINT_LENGTH_MIN", "0.20")
    os.environ.setdefault("GO1_SPLINT_LENGTH_MAX", "0.30")
    args_cli.balance_conditions = True
    print(
        "[INFO] Balanced validation enabled: "
        "GO1_EVAL_MODE=balanced, GO1_USE_PEG_LEG_CURRICULUM=0, "
        f"GO1_BALANCED_TARGET_MODE={os.getenv('GO1_BALANCED_TARGET_MODE', 'balanced_random')}"
    )

# Phase 자동 감지 → agent 기본값 결정
_phase = os.getenv("GO1_PHASE", "healthy").strip().lower()
_AGENT_DEFAULTS = {
    "healthy": "rsl_rl_cfg_entry_point",
    "teacher": "rsl_rl_teacher_cfg_entry_point",
    "student": "rsl_rl_distill_cfg_entry_point",
}
if args_cli.agent is None:
    args_cli.agent = _AGENT_DEFAULTS.get(_phase, "rsl_rl_cfg_entry_point")
    print(f"[INFO] GO1_PHASE={_phase} → agent={args_cli.agent}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401
import go1_lab.tasks  # noqa: F401
from go1_lab.tasks.manager_based.go1_lab.mdp.events import _get_peg_leg_per_env

LEG_NAMES = ["FL", "FR", "RL", "RR"]
CONTRA_LEG = {0: 1, 1: 0, 2: 3, 3: 2}  # FL<->FR, RL<->RR


def _find_foot_indices(sensor):
    """ContactSensor body 목록에서 FL/FR/RL/RR_foot 인덱스를 반환."""
    targets = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    indices = []
    for foot in targets:
        found = False
        for idx, body_name in enumerate(sensor.body_names):
            if foot in body_name:
                indices.append(idx)
                found = True
                break
        if not found:
            indices.append(0)
    return indices


def _format_env_ids(env_ids: list[int], limit: int = 24) -> str:
    """Long env-id lists make analysis logs hard to read."""
    if len(env_ids) <= limit:
        return str(env_ids)
    head = ", ".join(str(i) for i in env_ids[:limit])
    return f"[{head}, ...] ({len(env_ids)} total)"


def _collect_lstm_hook(module, input_, output, storage: list):
    """memory_s forward hook — LSTM 출력을 캡처."""
    storage.append(output.detach().cpu())


def _safe_step_dt(env, env_cfg) -> float:
    """Best-effort step dt for stance duration metrics."""
    if hasattr(env, "step_dt"):
        try:
            return float(env.step_dt)
        except Exception:
            pass
    try:
        sim_dt = float(env_cfg.sim.dt)
        decimation = float(getattr(env_cfg, "decimation", 1))
        return sim_dt * decimation
    except Exception:
        return 1.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # ── 평지 환경 설정 ──
    if args_cli.flat:
        print("[INFO] 평지(flat) 환경으로 전환합니다.")
        if hasattr(env_cfg.scene, "terrain"):
            terrain = env_cfg.scene.terrain
            if hasattr(terrain, "terrain_type"):
                terrain.terrain_type = "plane"
            if hasattr(terrain, "terrain_generator"):
                terrain.terrain_generator = None
        if hasattr(env_cfg, "curriculum") and hasattr(
            env_cfg.curriculum, "terrain_levels"
        ):
            env_cfg.curriculum.terrain_levels = None
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            obs_policy = env_cfg.observations.policy
            if hasattr(obs_policy, "enable_corruption"):
                obs_policy.enable_corruption = False

    # ── 속도 명령 고정 (동일 시나리오 보장) ──
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        vel_cmd = env_cfg.commands.base_velocity
        _vx = float(args_cli.target_vx)
        vel_cmd.ranges.lin_vel_x = (_vx, _vx)
        vel_cmd.ranges.lin_vel_y = (0.0, 0.0)
        vel_cmd.ranges.ang_vel_z = (0.0, 0.0)
        vel_cmd.ranges.heading = (0.0, 0.0)
        vel_cmd.heading_command = False
        print(f"[INFO] 속도 명령 고정: vx={_vx:.2f} m/s, vy=0, yaw=0 (동일 시나리오)")

    # 학습(train.py)과 동일하게 저장 경로는 프로젝트 루트 기준 logs/rsl_rl/...
    # (train을 다른 디렉터리에서 실행했다면 GO1_PROJECT_ROOT로 그 루트를 지정)
    _script_dir = Path(__file__).resolve().parent
    _env_root = os.environ.get("GO1_PROJECT_ROOT", "").strip()
    _project_root = (
        str(Path(_env_root).expanduser().resolve())
        if _env_root
        else str(_script_dir.parent.parent)
    )
    log_root_path = cli_args.rsl_rl_experiment_log_dir(
        agent_cfg.experiment_name, root=_project_root
    )

    def _latest_model_in_dir(run_dir: str) -> str | None:
        if not os.path.isdir(run_dir):
            return None
        files = [f for f in os.listdir(run_dir) if re.match(r"model_.*\.pt$", f)]
        if not files:
            return None
        files.sort(key=lambda m: f"{m:0>15}")
        return os.path.join(run_dir, files[-1])

    def _resolve_student_checkpoint() -> str:
        """--checkpoint가 상대경로일 때 cwd·프로젝트 logs/rsl_rl·실험 폴더 순으로 탐색."""
        if args_cli.use_pretrained_checkpoint:
            pretrained_task = args_cli.pretrained_task or args_cli.task
            pretrained_task = pretrained_task.split(":")[-1].replace("-Play", "")
            pretrained_path = get_published_pretrained_checkpoint(
                "rsl_rl", pretrained_task
            )
            if not pretrained_path:
                raise FileNotFoundError(
                    "Isaac Lab published pretrained checkpoint is unavailable for "
                    f"task={pretrained_task!r}."
                )
            return pretrained_path
        ckpt = (args_cli.checkpoint or "").strip()
        if not ckpt:
            return get_checkpoint_path(
                log_root_path, agent_cfg.load_run or ".*", agent_cfg.load_checkpoint
            )
        if os.path.isabs(ckpt) and os.path.isfile(ckpt):
            return ckpt
        candidates = [
            os.path.abspath(ckpt),
            os.path.join(os.getcwd(), "logs", "rsl_rl", ckpt),
            os.path.join(_project_root, "logs", "rsl_rl", ckpt),
            os.path.join(log_root_path, ckpt),
        ]
        # 후보 폴더의 모든 하위 폴더에서 파일 탐색 (날짜/시간 폴더 대응)
        for search_root in [
            log_root_path,
            os.path.join(os.getcwd(), "logs", "rsl_rl"),
            os.path.join(_project_root, "logs", "rsl_rl"),
        ]:
            if os.path.isdir(search_root):
                for r, dirs, files in os.walk(search_root):
                    if ckpt in files:
                        candidates.append(os.path.join(r, ckpt))
        for c in candidates:
            c = os.path.normpath(os.path.abspath(c))
            if os.path.isfile(c):
                return c
        # 마지막 후보(일반적으로 train 로그 위치)
        return os.path.normpath(
            os.path.abspath(os.path.join(_project_root, "logs", "rsl_rl", ckpt))
        )

    resume_path = _resolve_student_checkpoint()
    if not os.path.isfile(resume_path):
        run_dir = os.path.dirname(resume_path)
        alt = _latest_model_in_dir(run_dir)
        if alt is not None:
            print(
                f"[WARN] 요청한 체크포인트가 없습니다: {resume_path}\n"
                f"[WARN] 같은 실행 폴더에서 가장 최근 model_*.pt 로 대체합니다: {alt}"
            )
            resume_path = alt
        else:
            hint = (
                f"체크포인트를 찾을 수 없습니다: {resume_path}\n"
                f"  - 해당 Student 학습이 저장한 run 폴더에 model_*.pt 가 있는지 확인하세요.\n"
                f"  - train 을 다른 작업 디렉터리에서 돌렸다면 "
                f"GO1_PROJECT_ROOT=<그 디렉터리> 를 설정하세요 (현재 프로젝트 루트: {_project_root})."
            )
            raise FileNotFoundError(hint)
    print(f"[INFO] Loading student checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = PegLegActionMaskWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Runner 타입 자동 분기 ──
    phase = os.getenv("GO1_PHASE", "healthy").strip().lower()
    use_distillation = phase == "student"

    _agent_dict = agent_cfg.to_dict()
    # RSL-RL 3.0.1 호환성 패치
    if "policy" in _agent_dict and "class_name" not in _agent_dict["policy"]:
        _agent_dict["policy"]["class_name"] = "ActorCritic"
    if "algorithm" in _agent_dict:
        _agent_dict["algorithm"].pop("optimizer", None)
        _agent_dict["algorithm"].pop("config_class", None)
        _agent_dict["algorithm"].pop("share_cnn_encoders", None)
    if "policy" in _agent_dict:
        _agent_dict["policy"].pop("share_cnn_encoders", None)
        _agent_dict["policy"].pop("config_class", None)

    if use_distillation:
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(
            env, _agent_dict, log_dir=None, device=agent_cfg.device
        )
        runner.load(resume_path)
        policy_nn = runner.alg.policy
    else:
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, _agent_dict, log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy_nn = getattr(runner.alg, "policy", None) or getattr(
            runner.alg, "actor_critic", None
        )
        if policy_nn is None:
            raise RuntimeError(
                "runner.alg 에서 policy 또는 actor_critic 속성을 찾을 수 없습니다."
            )

    policy_nn.eval()
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(
        f"[INFO] Runner: {'DistillationRunner' if use_distillation else 'OnPolicyRunner'} (phase={phase})"
    )

    # ── LSTM hidden-state hook 등록 (Student 모델만) ──
    lstm_outputs: list[torch.Tensor] = []
    hook_handle = None
    if use_distillation and hasattr(policy_nn, "memory_s"):
        hook_handle = policy_nn.memory_s.register_forward_hook(
            lambda m, i, o: _collect_lstm_hook(m, i, o, lstm_outputs)
        )

    obs, _ = env.reset()
    base_env = env.unwrapped
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # ── 실제 부상 상태로 그룹 분류 (env_id % 5가 아닌 _peg_leg_index 기반) ──
    def _build_groups():
        grp = {0: [], 1: [], 2: [], 3: [], 4: []}
        if hasattr(base_env, "_peg_leg_index"):
            peg_idx = base_env._peg_leg_index.cpu().numpy()
            for eid in range(env.num_envs):
                idx = int(peg_idx[eid])
                if idx < 0:
                    grp[0].append(eid)
                else:
                    grp[idx + 1].append(eid)
        else:
            peg_info = _get_peg_leg_per_env(base_env, env_ids)
            for eid, leg_idx in peg_info.items():
                if leg_idx is None:
                    grp[0].append(eid)
                else:
                    grp[leg_idx + 1].append(eid)
        return grp

    groups = _build_groups()

    print("\n" + "=" * 80)
    print("Environment Configuration (grouped by ACTUAL injury)")
    print("=" * 80)
    for gid in range(5):
        label = "Normal" if gid == 0 else f"{LEG_NAMES[gid-1]} Peg"
        print(f"  {label}: {len(groups[gid])} envs {_format_env_ids(groups[gid])}")
    if args_cli.balanced_envs:
        counts = [len(groups[gid]) for gid in range(5)]
        if max(counts) - min(counts) <= 1:
            print(f"  [BALANCED] initial env counts are 1:1:1:1:1 within rounding: {counts}")
        else:
            print(f"  [WARNING] balanced_envs requested but initial counts are uneven: {counts}")
    empty = [LEG_NAMES[g - 1] for g in range(1, 5) if len(groups[g]) == 0]
    if empty:
        print(
            f"  [WARNING] No envs for: {', '.join(empty)}. Increase --num_envs (recommend 100+)."
        )
    print("=" * 80)

    # ── 데이터 수집 ──
    contact_hist = []  # (steps, num_envs, 4)
    gt_index_hist = []  # (steps, num_envs)
    gt_splint_hist = []  # (steps, num_envs)
    root_y_hist = []  # (steps, num_envs)
    foot_z_hist = []  # (steps, num_envs, 4) foot heights for gait-phase/diagonal-coupling
    base_pos_hist = []  # (steps, num_envs, 3) base xyz for CoM displacement
    base_quat_hist = []  # (steps, num_envs, 4) base orientation
    torque_l2_hist = []  # (steps, num_envs)
    mech_power_abs_hist = []  # (steps, num_envs)

    sensor = base_env.scene.sensors[args_cli.contact_sensor]
    robot = None
    try:
        robot = base_env.scene["robot"]
    except Exception:
        robot = None
    foot_indices = _find_foot_indices(sensor)
    print(f"[INFO] foot indices: {foot_indices}")
    # robot body indices for the four feet in FL,FR,RL,RR order (for foot-height
    # time-series used by gait-phase / diagonal-coupling biomechanics metrics)
    robot_foot_idx = None
    if robot is not None and hasattr(robot.data, "body_names"):
        _bn = list(robot.data.body_names)
        _order = []
        for _leg in ["FL", "FR", "RL", "RR"]:
            _m = [i for i, n in enumerate(_bn) if n.startswith(_leg) and "foot" in n.lower()]
            if _m:
                _order.append(_m[0])
        if len(_order) == 4:
            robot_foot_idx = torch.tensor(_order, device=env.device, dtype=torch.long)
    print(f"[INFO] robot foot body idx (FL,FR,RL,RR): {robot_foot_idx.tolist() if robot_foot_idx is not None else None}")
    print(
        f"[INFO] contact force: {'|Fz| only (approx. vertical GRF)' if args_cli.contact_use_z_only else '||F|| (3D norm)'}"
    )
    print(f"[INFO] 데이터 수집 시작 ({args_cli.steps} 스텝) ...")

    # ── Left/right canonicalization (GO1_CANONICALIZE=1) ──
    # Fold the bilaterally-symmetric env along its sagittal axis: right-side
    # injuries (FR, RR) are mirrored to the left frame, the policy is run, and
    # the action is mirrored back. This makes FR-injury an EXACT mirror of
    # FL-injury and RR an exact mirror of RL (zero L-R deviation by construction)
    # while the antalgic loading/gait is whatever the policy learned on the left.
    _canon = os.getenv("GO1_CANONICALIZE", "0").strip().lower() in {"1", "true", "yes", "on"}
    if _canon:
        from go1_lab.tasks.manager_based.go1_lab.mdp import mirror as _mir
        print("[INFO] left/right CANONICALIZATION enabled (FR->FL, RR->RL mirror)")

        def _canon_mirror(o, right):
            if isinstance(o, torch.Tensor):
                oc = o.clone()
                oc[right] = _mir.mirror_full_obs(o[right])
                return oc
            oc = o.clone() if hasattr(o, "clone") else dict(o)
            for k in o.keys():
                v = o[k].clone()
                if k == "policy":
                    # "policy" may be concatenated [proprio(48)+privileged(3)] or
                    # proprio-only; mirror_full_obs mirrors the privileged tail by
                    # dim detection (so the injury index FR->FL gets mirrored).
                    v[right] = _mir.mirror_full_obs(o[k][right])
                elif k in ("privileged_obs", "privileged"):
                    v[right] = _mir.mirror_privileged_obs(o[k][right])
                oc[k] = v
            return oc

    if _canon:
        _ks = {k: tuple(obs[k].shape) for k in obs.keys()} if hasattr(obs, "keys") else "tensor"
        print(f"[CANON-DEBUG] _canon={_canon} obs_type={type(obs).__name__} keys/shapes={_ks}", flush=True)
    for step in range(args_cli.steps):
        with torch.inference_mode():
            if _canon and hasattr(base_env, "_peg_leg_index"):
                _pidx = base_env._peg_leg_index
                _right = (_pidx == 1) | (_pidx == 3)  # FR, RR (internal idx)
                if step == 0:
                    print(f"[CANON-DEBUG] right-injured envs={int(_right.sum())}/{_right.numel()} "
                          f"pidx_unique={torch.unique(_pidx).tolist()}", flush=True)
                if bool(_right.any()):
                    actions = policy(_canon_mirror(obs, _right))
                    actions[_right] = _mir.mirror_action(actions[_right])
                else:
                    actions = policy(obs)
            else:
                actions = policy(obs)
            ret = env.step(actions)
            obs = ret[0]

        forces = sensor.data.net_forces_w
        if args_cli.contact_use_z_only:
            fi = torch.tensor(foot_indices, device=forces.device, dtype=torch.long)
            feet_forces = torch.abs(forces[:, fi, 2])
        else:
            force_mag = torch.norm(forces, dim=-1)
            feet_forces = force_mag[:, foot_indices]
        contact_hist.append(feet_forces.cpu().numpy())

        gt_idx = (
            base_env._peg_leg_index.float() + 1.0
            if hasattr(base_env, "_peg_leg_index")
            else torch.zeros(env.num_envs, device=env.device)
        )
        gt_spl = (
            base_env._peg_leg_splint_length
            if hasattr(base_env, "_peg_leg_splint_length")
            else torch.zeros(env.num_envs, device=env.device)
        )
        gt_index_hist.append(gt_idx.cpu().numpy())
        gt_splint_hist.append(gt_spl.cpu().numpy())
        if robot is not None and hasattr(robot.data, "root_pos_w"):
            root_y_hist.append(robot.data.root_pos_w[:, 1].detach().cpu().numpy())
            base_pos_hist.append(robot.data.root_pos_w.detach().cpu().numpy())
        else:
            root_y_hist.append(np.zeros(env.num_envs, dtype=np.float32))
            base_pos_hist.append(np.zeros((env.num_envs, 3), dtype=np.float32))
        if robot is not None and hasattr(robot.data, "root_quat_w"):
            base_quat_hist.append(robot.data.root_quat_w.detach().cpu().numpy())
        else:
            base_quat_hist.append(np.zeros((env.num_envs, 4), dtype=np.float32))
        if robot_foot_idx is not None and hasattr(robot.data, "body_pos_w"):
            foot_z_hist.append(robot.data.body_pos_w[:, robot_foot_idx, 2].detach().cpu().numpy())
        else:
            foot_z_hist.append(np.zeros((env.num_envs, 4), dtype=np.float32))

        if (
            robot is not None
            and hasattr(robot.data, "applied_torque")
            and hasattr(robot.data, "joint_vel")
        ):
            torque = robot.data.applied_torque.detach()
            joint_vel = robot.data.joint_vel.detach()
            torque_l2_hist.append(torch.sum(torch.square(torque), dim=1).cpu().numpy())
            mech_power_abs_hist.append(
                torch.sum(torch.abs(torque * joint_vel), dim=1).cpu().numpy()
            )
        else:
            torque_l2_hist.append(np.zeros(env.num_envs, dtype=np.float32))
            mech_power_abs_hist.append(np.zeros(env.num_envs, dtype=np.float32))

        if step % 200 == 0:
            print(f"  step {step}/{args_cli.steps}")

    if hook_handle is not None:
        hook_handle.remove()

    contact_hist = np.array(contact_hist)  # (S, E, 4)
    gt_index_hist = np.array(gt_index_hist)  # (S, E)
    gt_splint_hist = np.array(gt_splint_hist)  # (S, E)
    root_y_hist = np.array(root_y_hist)  # (S, E)
    torque_l2_hist = np.array(torque_l2_hist)  # (S, E)
    mech_power_abs_hist = np.array(mech_power_abs_hist)  # (S, E)
    foot_z_hist = np.array(foot_z_hist)  # (S, E, 4)
    base_pos_hist = np.array(base_pos_hist)  # (S, E, 3)
    base_quat_hist = np.array(base_quat_hist)  # (S, E, 4)

    # Raw time-series dump for offline biomechanics analysis (§2.4/2.5)
    _biodump = os.getenv("GO1_BIOMECH_DUMP", "")
    if _biodump:
        np.savez_compressed(
            _biodump,
            forces=contact_hist,
            foot_z=foot_z_hist,
            base_pos=base_pos_hist,
            base_quat=base_quat_hist,
            injury_idx=gt_index_hist,
            splint=gt_splint_hist,
        )
        print(f"[BIOMECH] dumped raw time-series ({contact_hist.shape}) -> {_biodump}", flush=True)

    if lstm_outputs:
        lstm_arr = torch.cat(lstm_outputs, dim=0).squeeze(1).numpy()
        print(f"[INFO] 수집 완료: contact {contact_hist.shape}, lstm {lstm_arr.shape}")
    else:
        lstm_arr = np.zeros((0, 1))
        print(
            f"[INFO] 수집 완료: contact {contact_hist.shape} (LSTM probing 비활성 — Phase {phase})"
        )

    # lstm_arr 이 (S, E, D) 혹은 (S*E, D) 인지 확인
    n_steps = args_cli.steps
    n_envs = env.num_envs
    lstm_3d = None
    if lstm_arr.size > 0 and lstm_arr.shape[-1] > 1:
        if lstm_arr.ndim == 2 and lstm_arr.shape[0] == n_steps * n_envs:
            lstm_3d = lstm_arr.reshape(n_steps, n_envs, -1)
        elif lstm_arr.ndim == 3:
            lstm_3d = lstm_arr
        elif lstm_arr.ndim == 2 and lstm_arr.shape[0] == n_steps:
            lstm_3d = np.repeat(lstm_arr[:, np.newaxis, :], n_envs, axis=1)
        else:
            lstm_3d = (
                lstm_arr.reshape(n_steps, n_envs, -1)
                if lstm_arr.size == n_steps * n_envs * lstm_arr.shape[-1]
                else None
            )

    # ── LSTM hidden-state dump (clean offline t-SNE for Fig.1) ──
    _lstm_dump = os.getenv("GO1_LSTM_DUMP", "")
    if _lstm_dump and lstm_3d is not None:
        np.savez_compressed(_lstm_dump, hidden=lstm_3d, injury_idx=gt_index_hist)
        print(f"[LSTM] dumped hidden {lstm_3d.shape} -> {_lstm_dump}", flush=True)

    # ── Normal/부상 조건별 env 샘플 수 출력 ──
    # (환경이 step 중간에 reset되면 조건이 바뀔 수 있어서, "초기 env 수"뿐 아니라 "수집 기간 평균 env 수"도 같이 봅니다.)
    flat_idx = gt_index_hist.reshape(-1).astype(np.int64)  # (S*E,) 0..4
    sample_counts = np.bincount(
        flat_idx, minlength=5
    )  # total samples per condition over all steps
    avg_envs_per_step = sample_counts / float(
        n_steps
    )  # approximate env count per condition

    print("\n" + "=" * 80)
    print("Condition Counts (collected over steps)")
    print("=" * 80)
    print(
        f"  Normal (0): total_samples={sample_counts[0]}, avg_envs_per_step={avg_envs_per_step[0]:.2f}"
    )
    for gid in range(1, 5):
        key = f"{LEG_NAMES[gid-1]} Peg"
        print(
            f"  {key} ({gid}): total_samples={sample_counts[gid]}, avg_envs_per_step={avg_envs_per_step[gid]:.2f}"
        )
    injured_total = int(
        sample_counts[1] + sample_counts[2] + sample_counts[3] + sample_counts[4]
    )
    injured_avg = float(
        avg_envs_per_step[1]
        + avg_envs_per_step[2]
        + avg_envs_per_step[3]
        + avg_envs_per_step[4]
    )
    print(
        f"  Injured total (1..4): total_samples={injured_total}, avg_envs_per_step={injured_avg:.2f}"
    )

    # =====================================================================
    #  Part 1: Contact Force — per-step grouping by actual injury
    # =====================================================================
    print("\n" + "=" * 80)
    _cf_label = "|Fz| (N)" if args_cli.contact_use_z_only else "||F|| (N)"
    print(f"Part 1: Contact Force Comparison (per-step injury grouping) — {_cf_label}")
    print("=" * 80)

    # group every (step, env) sample by its gt_index at that step
    flat_forces = contact_hist.reshape(-1, 4)  # (S*E, 4)
    flat_idx = gt_index_hist.reshape(-1)  # (S*E,) — 0=Normal,1=FL,2=FR,3=RL,4=RR

    force_stats = {}
    # df_stats is load-bearing duty, used for the main duty plot.
    df_stats = {}
    contact_df_stats = {}
    flat_contact_duty = (contact_hist > args_cli.contact_threshold).reshape(-1, 4).astype(float)
    flat_load_duty = (contact_hist > args_cli.load_contact_threshold).reshape(-1, 4).astype(float)
    severity_plot_rows = []

    # 분석 결과만 공정 비교용: 조건별 샘플 수를 동일하게 서브샘플링
    balanced_mask_part1 = None
    if args_cli.balance_conditions:
        rng = np.random.RandomState(1000)
        counts = [int(np.sum(flat_idx == gid)) for gid in range(5)]
        nonzero = [c for c in counts if c > 0]
        if len(nonzero) < 5:
            missing = [LEG_NAMES[g - 1] for g in range(1, 5) if counts[g] == 0]
            print(
                f"[WARNING] balance_conditions enabled but some conditions have 0 samples. Missing: {missing}"
            )
        N = min(nonzero) if nonzero else 0
        if N > 0:
            balanced_mask_part1 = np.zeros_like(flat_idx, dtype=bool)
            for gid in range(5):
                idxs = np.flatnonzero(flat_idx == gid)
                if len(idxs) == 0:
                    continue
                chosen = rng.choice(idxs, size=N, replace=False)
                balanced_mask_part1[chosen] = True
            print(f"[BALANCE] Part1: using N={N} samples per present condition")
        else:
            print("[WARNING] balance_conditions: no samples found to balance")

    for gid in range(5):
        mask = flat_idx == gid
        if balanced_mask_part1 is not None:
            mask = mask & balanced_mask_part1
        count = int(mask.sum())
        if count == 0:
            continue
        key = "Normal" if gid == 0 else f"{LEG_NAMES[gid-1]} Peg"
        force_stats[key] = flat_forces[mask].mean(axis=0)
        contact_df_stats[key] = flat_contact_duty[mask].mean(axis=0)
        df_stats[key] = flat_load_duty[mask].mean(axis=0)
        print(f"  {key}: {count} samples")

    print(f"\n{'':20s} | {'FL':>8s} | {'FR':>8s} | {'RL':>8s} | {'RR':>8s}")
    print("-" * 65)
    for key, vals in force_stats.items():
        row = f"{key:20s} |"
        for v in vals:
            row += f" {v:8.2f} |"
        print(row)

    if "Normal" in force_stats:
        normal_f = force_stats["Normal"]
        print("\nForce reduction on injured leg:")
        for i in range(4):
            pkey = f"{LEG_NAMES[i]} Peg"
            if pkey in force_stats:
                peg_f = force_stats[pkey][i]
                reduction = (1.0 - peg_f / max(normal_f[i], 1e-6)) * 100
                print(
                    f"  {LEG_NAMES[i]}: Normal {normal_f[i]:.2f}N -> Peg {peg_f:.2f}N  ({reduction:+.1f}% reduction)"
                )

    if "Normal" in df_stats:
        print(
            f"\nDuty factors: contact >{args_cli.contact_threshold:g}N vs load-bearing >{args_cli.load_contact_threshold:g}N"
        )
        print(f"{'':20s} | {'FL':>8s} | {'FR':>8s} | {'RL':>8s} | {'RR':>8s}")
        print("-" * 65)
        for key in force_stats:
            if key not in contact_df_stats or key not in df_stats:
                continue
            row_contact = f"{key + ' contact':20s} |"
            row_load = f"{key + ' load':20s} |"
            for v in contact_df_stats[key]:
                row_contact += f" {v:8.3f} |"
            for v in df_stats[key]:
                row_load += f" {v:8.3f} |"
            print(row_contact)
            print(row_load)

    # Vertical impulse proxy: mean load magnitude multiplied by load-bearing
    # duty. This is more defensible for antalgic gait than duty alone because
    # injured animals can reduce limb load while keeping or extending stance.
    impulse_proxy_stats = {}
    injured_impulse_reductions = {}
    if "Normal" in force_stats and "Normal" in df_stats:
        for key in force_stats:
            if key in df_stats:
                impulse_proxy_stats[key] = force_stats[key] * df_stats[key]

        print("\nVertical impulse proxy on injured leg (mean |Fz| x load duty)")
        for i in range(4):
            pkey = f"{LEG_NAMES[i]} Peg"
            if pkey not in impulse_proxy_stats:
                continue
            normal_impulse = float(impulse_proxy_stats["Normal"][i])
            peg_impulse = float(impulse_proxy_stats[pkey][i])
            reduction = (1.0 - peg_impulse / max(normal_impulse, 1e-6)) * 100.0
            injured_impulse_reductions[LEG_NAMES[i]] = reduction
            print(
                f"  {LEG_NAMES[i]}: Normal {normal_impulse:.2f} -> "
                f"Peg {peg_impulse:.2f}  ({reduction:+.1f}% reduction)"
            )

    if "Normal" in force_stats:
        flat_splint = gt_splint_hist.reshape(-1)
        injured_mask = flat_idx > 0
        valid_splint = injured_mask & np.isfinite(flat_splint) & (flat_splint > 0.0)
        if np.any(valid_splint):
            spl_values = flat_splint[valid_splint]
            q1, q2 = np.quantile(spl_values, [1.0 / 3.0, 2.0 / 3.0])
            severity_bins = [
                ("short/severe", valid_splint & (flat_splint <= q1)),
                ("mid/moderate", valid_splint & (flat_splint > q1) & (flat_splint <= q2)),
                ("long/mild", valid_splint & (flat_splint > q2)),
            ]
            print("\nSeverity-stratified injured-limb metrics")
            print(
                f"  bins by observed splint tertiles: short<= {q1:.3f}m, "
                f"mid<= {q2:.3f}m, long>{q2:.3f}m"
            )
            print(
                f"{'Severity':16s} | {'N':>8s} | {'ForceRed(%)':>11s} | "
                f"{'ContactDF':>9s} | {'LoadDF':>7s}"
            )
            print("-" * 64)
            normal_f = force_stats["Normal"]
            for name, mask in severity_bins:
                if not np.any(mask):
                    continue
                gids = flat_idx[mask].astype(int)
                aff_legs = gids - 1
                sample_rows = np.flatnonzero(mask)
                injured_forces = flat_forces[sample_rows, aff_legs]
                normal_refs = np.maximum(normal_f[aff_legs], 1e-6)
                force_red = float(np.mean((1.0 - injured_forces / normal_refs) * 100.0))
                contact_df = float(
                    np.mean(flat_contact_duty[sample_rows, aff_legs])
                )
                load_df = float(np.mean(flat_load_duty[sample_rows, aff_legs]))
                severity_plot_rows.append(
                    {
                        "name": name,
                        "force_red": force_red,
                        "contact_df": contact_df,
                        "load_df": load_df,
                        "count": int(mask.sum()),
                    }
                )
                print(
                    f"{name:16s} | {int(mask.sum()):8d} | {force_red:11.1f} | "
                    f"{contact_df:9.3f} | {load_df:7.3f}"
                )

    # =====================================================================
    #  Part 1-B: Requested gait/biomechanics metrics
    # =====================================================================
    print("\n" + "=" * 80)
    print("Part 1-B: Additional Biomechanics Metrics")
    print("=" * 80)
    print("Definitions:")
    print(
        "  - affected-side stance duration = mean contiguous load-bearing run length * step_dt"
    )
    print("  - affected-side peak GRF = max(vertical/contact force) on affected leg")
    print("  - symmetry index (SI, %) = |A - C| / (0.5*(A + C)) * 100")
    print(
        "  - CoM lateral sway (m) = detrended peak-to-peak lateral motion during condition"
    )
    print(
        f"  - contact duty uses >{args_cli.contact_threshold:g}N; "
        f"load-bearing duty uses >{args_cli.load_contact_threshold:g}N"
    )
    print("  - duty factor asymmetry = |load_DF_affected - load_DF_contralateral|")

    step_dt = _safe_step_dt(base_env, env_cfg)

    def _mean_true_run_duration_sec(
        x_bool: np.ndarray,
        dt: float,
        min_run_steps: int = 2,
        max_run_steps: int = 40,
    ) -> float:
        """
        Average duration of contiguous True-runs in seconds.
        Runs longer than max_run_steps are treated as non-gait contacts
        (e.g., fall/drag/static contact) and excluded.
        """
        if x_bool.size == 0:
            return 0.0
        x = x_bool.astype(np.int32)
        padded = np.pad(x, (1, 1), mode="constant", constant_values=0)
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        if len(starts) == 0:
            return 0.0
        run_lengths = ends - starts
        valid_runs = run_lengths[
            (run_lengths >= int(min_run_steps)) & (run_lengths <= int(max_run_steps))
        ]
        if len(valid_runs) == 0:
            return 0.0
        return float(np.mean(valid_runs) * dt)

    def _windowed_peak_to_peak(y: np.ndarray, window_size: int = 25) -> float:
        """Windowed detrended peak-to-peak lateral sway."""
        if y.size < window_size:
            return 0.0
        sways = []
        for i in range(0, y.size - window_size + 1, window_size):
            seg = y[i : i + window_size]
            t = np.arange(window_size, dtype=np.float64)
            coef = np.polyfit(t, seg.astype(np.float64), deg=1)
            trend = coef[0] * t + coef[1]
            res = seg - trend
            sways.append(np.max(res) - np.min(res))
        return float(np.mean(sways)) if sways else 0.0

    print(f"\nUsing step_dt={step_dt:.5f} s")
    print(
        f"\n{'Condition':18s} | {'Stance(s)':>9s} | {'PeakGRF(N)':>10s} | {'SI(%)':>8s} | {'CoMsway(m)':>10s} | {'DF asym':>8s}"
    )
    print("-" * 82)

    biomech_rows = []
    energy_by_condition = {}

    # 0(Normal)부터 4(RR Peg)까지 모두 순회합니다.
    for gid in range(5):
        if gid == 0:
            cond_name = "Normal"
            # 정상 상태는 환측/건측이 없으므로, 임의로 FL(0)과 FR(1)을 비교하여 완벽한 대칭성을 증명합니다.
            aff = 0
            ctr = 1
        else:
            aff = gid - 1
            ctr = CONTRA_LEG[aff]
            cond_name = f"{LEG_NAMES[aff]} Peg"

        # 조건에 맞는 롤아웃 데이터 마스크
        cond_mask_2d = gt_index_hist == gid  # (S, E)
        if not np.any(cond_mask_2d):
            continue

        # Load-bearing duty factor. Raw contact duty is printed separately above.
        cond_contact_aff = (
            contact_hist[:, :, aff] > args_cli.load_contact_threshold
        ) & cond_mask_2d
        cond_contact_ctr = (
            contact_hist[:, :, ctr] > args_cli.load_contact_threshold
        ) & cond_mask_2d
        df_aff = float(cond_contact_aff.mean())
        df_ctr = float(cond_contact_ctr.mean())

        # Stance duration: stopwatch-style contiguous contact run (수정된 안전한 함수 사용)
        stance_runs = []
        for e in range(n_envs):
            stance_runs.append(
                _mean_true_run_duration_sec(
                    cond_contact_aff[:, e],
                    step_dt,
                    min_run_steps=2,
                    max_run_steps=40,
                )
            )
        stance_runs = [v for v in stance_runs if v > 0.0]
        stance_duration = float(np.mean(stance_runs)) if stance_runs else 0.0

        mask = flat_idx == gid
        if not np.any(mask):
            continue

        # Peak GRF
        peak_grf = float(np.max(flat_forces[mask, aff]))

        # Symmetry Index (SI)
        mean_aff = float(flat_forces[mask, aff].mean())
        mean_ctr = float(flat_forces[mask, ctr].mean())
        si = abs(mean_aff - mean_ctr) / max(0.5 * (mean_aff + mean_ctr), 1e-6) * 100.0

        # CoM lateral sway: windowed detrended peak-to-peak (수정된 안전한 함수 사용)
        sway_vals = []
        for e in range(n_envs):
            y_seg = root_y_hist[cond_mask_2d[:, e], e]
            if y_seg.size < 25:
                continue
            sway_vals.append(_windowed_peak_to_peak(y_seg, window_size=25))
        com_shift = float(np.mean(sway_vals)) if sway_vals else 0.0

        # Duty Factor Asymmetry
        df_asym = abs(df_aff - df_ctr)

        torque_l2_mean = float(torque_l2_hist.reshape(-1)[mask].mean())
        mech_power_abs_mean = float(mech_power_abs_hist.reshape(-1)[mask].mean())
        transport_power_proxy = mech_power_abs_mean / max(float(args_cli.target_vx), 1e-6)
        energy_by_condition[cond_name] = {
            "torque_l2_mean": torque_l2_mean,
            "abs_mechanical_power_w": mech_power_abs_mean,
            "abs_mechanical_work_per_meter_proxy_j_per_m": transport_power_proxy,
        }

        print(
            f"{cond_name:18s} | {stance_duration:9.4f} | {peak_grf:10.2f} | {si:8.2f} | {com_shift:10.4f} | {df_asym:8.4f}"
        )
        biomech_rows.append(
            {
                "condition": cond_name,
                "stance_duration": stance_duration,
                "peak_grf": peak_grf,
                "si": si,
                "com_sway": com_shift,
                "df_asym": df_asym,
                "torque_l2_mean": torque_l2_mean,
                "abs_mechanical_power_w": mech_power_abs_mean,
                "abs_mechanical_work_per_meter_proxy_j_per_m": transport_power_proxy,
            }
        )

    if energy_by_condition:
        print("\nEnergy / effort proxy")
        print(
            f"{'Condition':18s} | {'TorqueL2':>10s} | {'AbsPower(W)':>11s} | {'J/m proxy':>10s} | {'vsNormal':>9s}"
        )
        print("-" * 72)
        normal_power = max(
            float(energy_by_condition.get("Normal", {}).get("abs_mechanical_power_w", 0.0)),
            1e-6,
        )
        for cond_name, vals in energy_by_condition.items():
            rel = vals["abs_mechanical_power_w"] / normal_power
            print(
                f"{cond_name:18s} | {vals['torque_l2_mean']:10.2f} | "
                f"{vals['abs_mechanical_power_w']:11.2f} | "
                f"{vals['abs_mechanical_work_per_meter_proxy_j_per_m']:10.2f} | "
                f"{rel:9.2f}x"
            )

    # =====================================================================
    #  Part 2: LSTM 부상 파라미터 추정
    # =====================================================================
    print("\n" + "=" * 80)
    print("Part 2: Student LSTM Injury Parameter Estimation")
    print("=" * 80)

    probe_results = {}
    if lstm_3d is not None:
        X = lstm_3d.reshape(-1, lstm_3d.shape[-1])  # (S*E, D)
        y_idx = gt_index_hist.reshape(-1).astype(int)  # (S*E,)
        y_spl = gt_splint_hist.reshape(-1)  # (S*E,)

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y_idx) & np.isfinite(y_spl)
        X, y_idx, y_spl = X[valid], y_idx[valid], y_spl[valid]

        if args_cli.balance_conditions and len(X) > 0:
            rng = np.random.RandomState(1000)
            counts = [int(np.sum(y_idx == gid)) for gid in range(5)]
            nonzero = [c for c in counts if c > 0]
            N = min(nonzero) if nonzero else 0
            if N > 0:
                balanced_indices = []
                for gid in range(5):
                    idxs = np.flatnonzero(y_idx == gid)
                    if len(idxs) == 0:
                        continue
                    chosen = rng.choice(idxs, size=N, replace=False)
                    balanced_indices.append(chosen)
                if balanced_indices:
                    balanced_indices = np.concatenate(balanced_indices)
                    X = X[balanced_indices]
                    y_idx = y_idx[balanced_indices]
                    y_spl = y_spl[balanced_indices]
                    print(
                        f"[BALANCE] Part2: using N={N} samples per present condition (after valid filter)"
                    )

        n = len(X)
        split = int(n * 0.7)
        perm = np.random.RandomState(42).permutation(n)
        tr, te = perm[:split], perm[split:]

        # ── 2-A: Injury Leg Classification (Logistic Regression) ──
        print("\n[2-A] Injury Leg Classification (0=Normal, 1=FL, 2=FR, 3=RL, 4=RR)")
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y_idx[tr])
        pred_idx = clf.predict(X[te])
        acc = accuracy_score(y_idx[te], pred_idx) * 100
        print(f"  Classification Accuracy: {acc:.1f}%")
        cm = confusion_matrix(y_idx[te], pred_idx, labels=[0, 1, 2, 3, 4])
        labels = ["Normal", "FL", "FR", "RL", "RR"]
        print(f"  {'':>8s}", end="")
        for l in labels:
            print(f" {l:>6s}", end="")
        print()
        for i, row in enumerate(cm):
            print(f"  {labels[i]:>8s}", end="")
            for v in row:
                print(f" {v:6d}", end="")
            print()
        probe_results["idx_acc"] = acc
        probe_results["idx_cm"] = cm
        probe_results["idx_pred"] = pred_idx
        probe_results["idx_true"] = y_idx[te]

        # ── 2-B: Splint Length Regression (Linear Regression) ──
        print("\n[2-B] Splint Equivalent Length Regression (m)")
        injured_mask_tr = y_idx[tr] > 0
        injured_mask_te = y_idx[te] > 0

        if np.sum(injured_mask_tr) > 50 and np.sum(injured_mask_te) > 50:
            reg = LinearRegression()
            reg.fit(X[tr][injured_mask_tr], y_spl[tr][injured_mask_tr])
            pred_spl = reg.predict(X[te][injured_mask_te])
            true_spl = y_spl[te][injured_mask_te]
            r2 = r2_score(true_spl, pred_spl)
            mae = np.mean(np.abs(pred_spl - true_spl))
            print(f"  R² score: {r2:.4f}")
            print(f"  MAE: {mae:.4f} m")
            print(
                f"  Splint length range: {true_spl.min():.3f} ~ {true_spl.max():.3f} m"
            )
            probe_results["spl_r2"] = r2
            probe_results["spl_mae"] = mae
            probe_results["spl_pred"] = pred_spl
            probe_results["spl_true"] = true_spl
        else:
            print("  Insufficient injured samples -- skipping regression")
    else:
        print("  LSTM output shape mismatch -- skipping analysis")

    # =====================================================================
    #  시각화
    # =====================================================================
    save_dir = os.path.dirname(resume_path)
    fig = plt.figure(figsize=(20, 14))
    terrain_label = "Flat Terrain" if args_cli.flat else "Rough Terrain"
    _force_mode = "|Fz|" if args_cli.contact_use_z_only else "||F||"
    fig.suptitle(
        f"Student Policy Analysis — {terrain_label}, vx={float(args_cli.target_vx):.2f} m/s ({_force_mode})",
        fontsize=16,
        fontweight="bold",
    )

    # ── (1) Contact Force: Normal vs Peg ──
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(4)
    w = 0.35
    if "Normal" in force_stats:
        ax1.bar(x - w / 2, force_stats["Normal"], w, label="Normal", color="#1f77b4")
        peg_vals = []
        for i in range(4):
            pkey = f"{LEG_NAMES[i]} Peg"
            peg_vals.append(force_stats.get(pkey, np.zeros(4))[i])
        ax1.bar(
            x + w / 2,
            peg_vals,
            w,
            label="Peg Leg (injured)",
            color="#d62728",
            alpha=0.8,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(LEG_NAMES)
    ax1.set_ylabel("Avg |Fz| (N)" if args_cli.contact_use_z_only else "Avg ||F|| (N)")
    ax1.set_title("Injured Leg Contact Force\n(Normal vs Peg-leg)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # ── (2) Force Reduction % ──
    ax2 = fig.add_subplot(2, 3, 2)
    if "Normal" in force_stats:
        reductions = []
        for i in range(4):
            pkey = f"{LEG_NAMES[i]} Peg"
            if pkey in force_stats and force_stats["Normal"][i] > 1e-6:
                reductions.append(
                    (1.0 - force_stats[pkey][i] / force_stats["Normal"][i]) * 100
                )
            else:
                reductions.append(0)
        colors = ["green" if r > 0 else "red" for r in reductions]
        ax2.bar(LEG_NAMES, reductions, color=colors, alpha=0.8)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_ylabel("Force Reduction (%)")
        ax2.set_title("Injured Leg Force Reduction\n(positive = protection)")
        ax2.grid(axis="y", alpha=0.3)

    # ── (3) Vertical Impulse Proxy ──
    ax3 = fig.add_subplot(2, 3, 3)
    if "Normal" in impulse_proxy_stats:
        ax3.bar(
            x - w / 2,
            impulse_proxy_stats["Normal"],
            w,
            label="Normal",
            color="#1f77b4",
        )
        peg_impulse = []
        for i in range(4):
            pkey = f"{LEG_NAMES[i]} Peg"
            peg_impulse.append(impulse_proxy_stats.get(pkey, np.zeros(4))[i])
        ax3.bar(
            x + w / 2,
            peg_impulse,
            w,
            label="Peg Leg (injured)",
            color="#d62728",
            alpha=0.8,
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(LEG_NAMES)
    ax3.set_ylabel("Mean |Fz| x Load Duty")
    ax3.set_title("Injured-Limb Vertical Impulse Proxy\n(lower = reduced painful loading)")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # ── (4) Confusion Matrix ──
    ax4 = fig.add_subplot(2, 3, 4)
    if "idx_cm" in probe_results:
        cm = probe_results["idx_cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1e-8)
        im = ax4.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        labels = ["Normal", "FL", "FR", "RL", "RR"]
        ax4.set_xticks(range(5))
        ax4.set_xticklabels(labels, fontsize=9)
        ax4.set_yticks(range(5))
        ax4.set_yticklabels(labels, fontsize=9)
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("True")
        ax4.set_title(
            f"LSTM Injury Classification\n(Accuracy: {probe_results['idx_acc']:.1f}%)"
        )
        for i in range(5):
            for j in range(5):
                ax4.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    else:
        if severity_plot_rows:
            labels = [row["name"].replace("/", "\n") for row in severity_plot_rows]
            vals = [row["force_red"] for row in severity_plot_rows]
            ax4.bar(labels, vals, color="#2ca02c", alpha=0.8)
            ax4.axhspan(30, 80, color="#2ca02c", alpha=0.08, label="target band")
            ax4.axhline(95, color="#d62728", linestyle="--", linewidth=1, label="near non-use")
            ax4.set_ylim(0, 105)
            ax4.set_ylabel("Force Reduction (%)")
            ax4.set_title("Severity-Stratified Force Reduction\n(shorter splint = more severe)")
            ax4.legend(fontsize=8)
            ax4.grid(axis="y", alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Severity-Stratified Force Reduction")

    # ── (5) Splint Length Regression ──
    ax5 = fig.add_subplot(2, 3, 5)
    if "spl_pred" in probe_results:
        true_s = probe_results["spl_true"]
        pred_s = probe_results["spl_pred"]
        ax5.scatter(true_s, pred_s, alpha=0.15, s=4, c="#2ca02c")
        lims = [
            min(true_s.min(), pred_s.min()) - 0.02,
            max(true_s.max(), pred_s.max()) + 0.02,
        ]
        ax5.plot(lims, lims, "k--", linewidth=1, label="ideal")
        ax5.set_xlabel("True Splint Length (m)")
        ax5.set_ylabel("Predicted Splint Length (m)")
        ax5.set_title(
            f"LSTM Splint Length Estimation\n(R2={probe_results['spl_r2']:.3f}, MAE={probe_results['spl_mae']:.4f}m)"
        )
        ax5.legend()
        ax5.set_xlim(lims)
        ax5.set_ylim(lims)
        ax5.grid(alpha=0.3)
    else:
        if severity_plot_rows:
            labels = [row["name"].replace("/", "\n") for row in severity_plot_rows]
            x_sev = np.arange(len(labels))
            contact_vals = [row["contact_df"] for row in severity_plot_rows]
            load_vals = [row["load_df"] for row in severity_plot_rows]
            ax5.plot(x_sev, contact_vals, marker="o", label="contact duty >1N", color="#ff7f0e")
            ax5.plot(x_sev, load_vals, marker="o", label=f"load duty >{args_cli.load_contact_threshold:g}N", color="#d62728")
            ax5.axhspan(0.10, 0.45, color="#2ca02c", alpha=0.08, label="partial-use band")
            ax5.set_xticks(x_sev)
            ax5.set_xticklabels(labels)
            ax5.set_ylim(0, 1.0)
            ax5.set_ylabel("Duty Factor")
            ax5.set_title("Severity-Stratified Injured-Limb Duty\n(contact vs load-bearing)")
            ax5.legend(fontsize=8)
            ax5.grid(axis="y", alpha=0.3)
        else:
            ax5.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax5.transAxes,
            )
            ax5.set_title("Severity-Stratified Injured-Limb Duty")

    # ── (6) LSTM t-SNE ──
    ax6 = fig.add_subplot(2, 3, 6)
    if lstm_3d is not None:
        try:
            from sklearn.manifold import TSNE

            X_all = lstm_3d.reshape(-1, lstm_3d.shape[-1])
            y_all = gt_index_hist.reshape(-1).astype(int)
            subsample = min(5000, len(X_all))
            rng = np.random.RandomState(0)
            idx_sub = rng.choice(len(X_all), subsample, replace=False)
            emb = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(
                X_all[idx_sub]
            )
            y_sub = y_all[idx_sub]
            colors_map = {
                0: "#1f77b4",
                1: "#ff7f0e",
                2: "#2ca02c",
                3: "#d62728",
                4: "#9467bd",
            }
            for cls in range(5):
                mask = y_sub == cls
                label = "Normal" if cls == 0 else f"{LEG_NAMES[cls-1]} Peg"
                ax6.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    s=6,
                    alpha=0.5,
                    c=colors_map[cls],
                    label=label,
                )
            ax6.legend(fontsize=8, markerscale=3)
            ax6.set_title("LSTM Hidden State t-SNE\n(per injury condition)")
            ax6.set_xticks([])
            ax6.set_yticks([])
        except Exception as e:
            ax6.text(
                0.5,
                0.5,
                f"t-SNE failed: {e}",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=8,
            )
            ax6.set_title("LSTM t-SNE")
    else:
        if "Normal" in force_stats and "Normal" in df_stats:
            ax6.axvspan(30, 85, color="#2ca02c", alpha=0.08)
            ax6.axhspan(0.10, 0.45, color="#1f77b4", alpha=0.06)
            for i, leg in enumerate(LEG_NAMES):
                pkey = f"{leg} Peg"
                if pkey not in force_stats or pkey not in df_stats:
                    continue
                reduction = (1.0 - force_stats[pkey][i] / max(force_stats["Normal"][i], 1e-6)) * 100
                load_df = df_stats[pkey][i]
                ax6.scatter(reduction, load_df, s=80, label=leg)
                ax6.text(reduction + 1.0, load_df + 0.01, leg, fontsize=9)
            ax6.axvline(95, color="#d62728", linestyle="--", linewidth=1)
            ax6.set_xlim(0, 105)
            ax6.set_ylim(0, 0.8)
            ax6.set_xlabel("Injured-Limb Force Reduction (%)")
            ax6.set_ylabel("Injured-Limb Load-Bearing Duty")
            ax6.set_title("Antalgic Validity Map\n(target: protected but not unused)")
            ax6.grid(alpha=0.3)
        else:
            ax6.text(
                0.5, 0.5, "No LSTM data", ha="center", va="center", transform=ax6.transAxes
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, "student_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[SAVED] {save_path}")
    plt.close()

    # ── Summary ──
    metrics = {
        "checkpoint": resume_path,
        "figure": save_path,
        "target_vx": float(args_cli.target_vx),
        "contact_use_z_only": bool(args_cli.contact_use_z_only),
        "contact_threshold_n": float(args_cli.contact_threshold),
        "load_contact_threshold_n": float(args_cli.load_contact_threshold),
        "force_by_condition": {
            key: [float(v) for v in vals] for key, vals in force_stats.items()
        },
        "contact_duty_by_condition": {
            key: [float(v) for v in vals] for key, vals in contact_df_stats.items()
        },
        "load_duty_by_condition": {
            key: [float(v) for v in vals] for key, vals in df_stats.items()
        },
        "vertical_impulse_proxy_by_condition": {
            key: [float(v) for v in vals] for key, vals in impulse_proxy_stats.items()
        },
        "injured_vertical_impulse_reduction_by_leg_pct": {
            key: float(val) for key, val in injured_impulse_reductions.items()
        },
        "severity_rows": severity_plot_rows,
        "biomech_rows": biomech_rows,
        "energy_by_condition": energy_by_condition,
    }

    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)
    if "Normal" in force_stats:
        print("\n[Contact Force Comparison]")
        force_reductions = []
        duty_reductions = []
        for i in range(4):
            pkey = f"{LEG_NAMES[i]} Peg"
            if pkey in force_stats and force_stats["Normal"][i] > 1e-6:
                r = (1.0 - force_stats[pkey][i] / force_stats["Normal"][i]) * 100
                force_reductions.append(r)
                print(
                    f"  {LEG_NAMES[i]}: {r:+.1f}% reduction (Normal {force_stats['Normal'][i]:.1f}N -> Peg {force_stats[pkey][i]:.1f}N)"
                )
                if pkey in df_stats and "Normal" in df_stats:
                    d = (df_stats["Normal"][i] - df_stats[pkey][i]) * 100
                    duty_reductions.append(d)
        if duty_reductions:
            print("\n[Duty Factor Comparison]")
            for i in range(4):
                pkey = f"{LEG_NAMES[i]} Peg"
                if pkey in df_stats and "Normal" in df_stats:
                    d = (df_stats["Normal"][i] - df_stats[pkey][i]) * 100
                    print(
                        f"  {LEG_NAMES[i]}: {d:+.1f} pp reduction "
                        f"(Normal {df_stats['Normal'][i]:.3f} -> Peg {df_stats[pkey][i]:.3f})"
                    )
        if injured_impulse_reductions:
            print("\n[Vertical Impulse Proxy Comparison]")
            for leg in LEG_NAMES:
                if leg in injured_impulse_reductions:
                    print(
                        f"  {leg}: {injured_impulse_reductions[leg]:+.1f}% reduction "
                        "(mean force x load-bearing duty)"
                    )
        if force_reductions:
            protected_force = sum(r > 0.0 for r in force_reductions)
            protected_duty = sum(d > 0.0 for d in duty_reductions) if duty_reductions else 0
            protected_impulse = sum(
                r > 0.0 for r in injured_impulse_reductions.values()
            )
            impulse_reduction_min = (
                float(np.min(list(injured_impulse_reductions.values())))
                if injured_impulse_reductions
                else float("nan")
            )
            injured_duties = [
                float(df_stats[f"{LEG_NAMES[i]} Peg"][i])
                for i in range(4)
                if f"{LEG_NAMES[i]} Peg" in df_stats
            ]
            injured_force_reductions = {
                LEG_NAMES[i]: float(force_reductions[i])
                for i in range(min(4, len(force_reductions)))
            }
            injured_duty_by_leg = {
                LEG_NAMES[i]: float(df_stats[f"{LEG_NAMES[i]} Peg"][i])
                for i in range(4)
                if f"{LEG_NAMES[i]} Peg" in df_stats
            }
            injured_contact_duty_by_leg = {
                LEG_NAMES[i]: float(contact_df_stats[f"{LEG_NAMES[i]} Peg"][i])
                for i in range(4)
                if f"{LEG_NAMES[i]} Peg" in contact_df_stats
            }
            force_lr_gap_front = abs(
                injured_force_reductions.get("FL", float("nan"))
                - injured_force_reductions.get("FR", float("nan"))
            )
            force_lr_gap_rear = abs(
                injured_force_reductions.get("RL", float("nan"))
                - injured_force_reductions.get("RR", float("nan"))
            )
            duty_lr_gap_front = abs(
                injured_duty_by_leg.get("FL", float("nan"))
                - injured_duty_by_leg.get("FR", float("nan"))
            )
            duty_lr_gap_rear = abs(
                injured_duty_by_leg.get("RL", float("nan"))
                - injured_duty_by_leg.get("RR", float("nan"))
            )
            impulse_lr_gap_front = abs(
                injured_impulse_reductions.get("FL", float("nan"))
                - injured_impulse_reductions.get("FR", float("nan"))
            )
            impulse_lr_gap_rear = abs(
                injured_impulse_reductions.get("RL", float("nan"))
                - injured_impulse_reductions.get("RR", float("nan"))
            )
            mean_injured_duty = float(np.mean(injured_duties)) if injured_duties else float("nan")
            mean_force_reduction = float(np.mean(force_reductions))
            mean_duty_reduction = float(np.mean(duty_reductions)) if duty_reductions else float("nan")
            nonuse_legs = [
                leg
                for leg, reduction in injured_force_reductions.items()
                if reduction > 95.0 or injured_duty_by_leg.get(leg, 1.0) < 0.05
            ]
            near_nonuse_legs = [
                leg
                for leg, reduction in injured_force_reductions.items()
                if reduction > 85.0 and injured_duty_by_leg.get(leg, 1.0) < 0.25
            ]
            overuse_legs = [
                leg
                for leg, reduction in injured_force_reductions.items()
                if reduction < 20.0
            ]
            light_touch_legs = [
                leg
                for leg, contact_duty in injured_contact_duty_by_leg.items()
                if contact_duty > contact_df_stats["Normal"][LEG_NAMES.index(leg)]
                and injured_duty_by_leg.get(leg, 0.0) < df_stats["Normal"][LEG_NAMES.index(leg)]
            ]
            drag_like_legs = [
                leg
                for leg, contact_duty in injured_contact_duty_by_leg.items()
                if (
                    contact_duty - injured_duty_by_leg.get(leg, 0.0) > 0.65
                    or (
                        contact_duty > 0.90
                        and injured_duty_by_leg.get(leg, 1.0) < 0.20
                    )
                )
            ]
            reduction_min = float(np.min(force_reductions)) if force_reductions else float("nan")
            reduction_max = float(np.max(force_reductions)) if force_reductions else float("nan")
            duty_min = float(np.min(injured_duties)) if injured_duties else float("nan")
            duty_max = float(np.max(injured_duties)) if injured_duties else float("nan")
            severity_by_name = {row["name"]: row for row in severity_plot_rows}
            severe_row = severity_by_name.get("short/severe", {})
            mild_row = severity_by_name.get("long/mild", {})
            mid_row = severity_by_name.get("mid/moderate", {})
            mild_ok = (
                not mild_row
                or (
                    float(mild_row.get("force_red", 100.0)) <= 85.0
                    and float(mild_row.get("load_df", 0.0)) >= 0.15
                )
            )
            mid_ok = (
                not mid_row
                or (
                    float(mid_row.get("force_red", 100.0)) <= 90.0
                    and float(mid_row.get("load_df", 0.0)) >= 0.10
                )
            )
            severity_trend_ok = True
            if severe_row and mild_row:
                severity_force_gap = float(severe_row.get("force_red", 0.0)) - float(
                    mild_row.get("force_red", 0.0)
                )
                severity_duty_gap = float(mild_row.get("load_df", 0.0)) - float(
                    severe_row.get("load_df", 0.0)
                )
                severity_trend_ok = severity_force_gap >= 5.0 and severity_duty_gap >= 0.04
            # Main paper criterion:
            # Antalgic gait is primarily reduced load/vertical impulse on the
            # painful limb. Duty factor can decrease, stay similar, or even
            # increase slightly if the stance force is much smaller. We reject
            # complete non-use, weak protection, and drag-like light contact,
            # but do not require a prescribed stance-duration pattern. Severity
            # trends are reported as secondary biomechanics, not used as a hard
            # pass/fail gate for the paper-grade teacher checkpoint.
            primary_force_candidate = (
                protected_force >= 4
                and len(nonuse_legs) == 0
                and len(near_nonuse_legs) == 0
                and len(overuse_legs) == 0
                and len(drag_like_legs) == 0
                and reduction_min >= 30.0
                and reduction_max <= 90.0
                and mean_force_reduction >= 45.0
                and duty_min >= 0.12
                and duty_max <= 0.55
                and mild_ok
                and mid_ok
            )
            balanced_four_leg_candidate = (
                primary_force_candidate
                and reduction_min >= 50.0
                and (reduction_max - reduction_min) <= 25.0
                and duty_max <= 0.55
                and (duty_max - duty_min) <= 0.22
                and force_lr_gap_front <= 15.0
                and force_lr_gap_rear <= 15.0
                and duty_lr_gap_front <= 0.16
                and duty_lr_gap_rear <= 0.16
                and (not injured_impulse_reductions or impulse_reduction_min >= 45.0)
                and (not injured_impulse_reductions or impulse_lr_gap_front <= 15.0)
                and (not injured_impulse_reductions or impulse_lr_gap_rear <= 15.0)
            )
            paper_grade_candidate = (
                balanced_four_leg_candidate
            )
            # ── mirror symmetry index ──────────────────────────────────────
            # Measures how closely FL/FR and RL/RR injury responses mirror
            # each other (paper biological claim: antalgic objective is
            # left-right symmetric → identical injury conditions produce
            # mirror-symmetric compensatory gait).
            # MSI = mean of the 4 left-right gap magnitudes, normalised so
            # 0 = perfect mirror, 100 = completely asymmetric.
            _msi_force = (force_lr_gap_front + force_lr_gap_rear) / 2.0
            _msi_duty = (duty_lr_gap_front + duty_lr_gap_rear) / 2.0
            _msi_impulse = (
                (impulse_lr_gap_front + impulse_lr_gap_rear) / 2.0
                if not (np.isnan(impulse_lr_gap_front) or np.isnan(impulse_lr_gap_rear))
                else float("nan")
            )
            mirror_symmetry_ok = (
                force_lr_gap_front <= 10.0
                and force_lr_gap_rear <= 10.0
                and duty_lr_gap_front <= 0.12
                and duty_lr_gap_rear <= 0.12
                and (np.isnan(_msi_impulse) or _msi_impulse <= 10.0)
            )
            metrics["antalgic_validation"] = {
                "force_reduction_by_leg_pct": injured_force_reductions,
                "injured_load_duty_by_leg": injured_duty_by_leg,
                "injured_contact_duty_by_leg": injured_contact_duty_by_leg,
                "vertical_impulse_reduction_by_leg_pct": {
                    key: float(val) for key, val in injured_impulse_reductions.items()
                },
                "mean_force_reduction_pct": mean_force_reduction,
                "mean_load_duty_reduction_pp": mean_duty_reduction,
                "mean_injured_load_duty": mean_injured_duty,
                "protected_force_count": int(protected_force),
                "protected_duty_count": int(protected_duty),
                "protected_impulse_count": int(protected_impulse),
                "vertical_impulse_reduction_min_pct": impulse_reduction_min,
                "left_right_force_reduction_gap_front_pp": force_lr_gap_front,
                "left_right_force_reduction_gap_rear_pp": force_lr_gap_rear,
                "left_right_load_duty_gap_front": duty_lr_gap_front,
                "left_right_load_duty_gap_rear": duty_lr_gap_rear,
                "left_right_impulse_reduction_gap_front_pp": impulse_lr_gap_front,
                "left_right_impulse_reduction_gap_rear_pp": impulse_lr_gap_rear,
                "mirror_symmetry_force_gap_mean_pp": _msi_force,
                "mirror_symmetry_duty_gap_mean": _msi_duty,
                "mirror_symmetry_impulse_gap_mean_pp": _msi_impulse,
                "mirror_symmetry_ok": bool(mirror_symmetry_ok),
                "nonuse_legs": nonuse_legs,
                "near_nonuse_legs": near_nonuse_legs,
                "weak_protection_legs": overuse_legs,
                "light_touch_legs": light_touch_legs,
                "drag_like_legs": drag_like_legs,
                "force_reduction_min_pct": reduction_min,
                "force_reduction_max_pct": reduction_max,
                "injured_load_duty_min": duty_min,
                "injured_load_duty_max": duty_max,
                "mild_severity_ok": bool(mild_ok),
                "mid_severity_ok": bool(mid_ok),
                "severity_trend_ok": bool(severity_trend_ok),
                "primary_force_candidate": bool(primary_force_candidate),
                "balanced_four_leg_candidate": bool(balanced_four_leg_candidate),
                "paper_grade_candidate": bool(paper_grade_candidate),
            }
            print("\n[Antalgic Validation]")
            print(
                f"  Force protection: {protected_force}/4 legs positive, "
                f"mean={mean_force_reduction:+.1f}%"
            )
            if duty_reductions:
                print(
                    f"  Duty protection:  {protected_duty}/4 legs positive, "
                    f"mean={mean_duty_reduction:+.1f} pp"
                )
            if injured_impulse_reductions:
                print(
                    f"  Impulse protection: {protected_impulse}/4 legs positive, "
                    f"min={impulse_reduction_min:+.1f}%"
                )
            _mirror_ok_str = "OK" if mirror_symmetry_ok else "FAIL"
            print(
                f"  Mirror symmetry [{_mirror_ok_str}]: "
                f"force FL/FR={force_lr_gap_front:.1f}pp RL/RR={force_lr_gap_rear:.1f}pp | "
                f"duty FL/FR={duty_lr_gap_front:.3f} RL/RR={duty_lr_gap_rear:.3f} | "
                f"impulse FL/FR={impulse_lr_gap_front:.1f}pp RL/RR={impulse_lr_gap_rear:.1f}pp"
                f"\n    (gates: force≤10pp, duty≤0.12, impulse≤10pp)"
            )
            if injured_duties:
                print(f"  Injured duty mean: {mean_injured_duty:.3f}")
                print(
                    f"  Duty threshold: contact >{args_cli.contact_threshold:g}N, "
                    f"load-bearing >{args_cli.load_contact_threshold:g}N"
                )
            if nonuse_legs:
                print(f"  Non-use legs: {', '.join(nonuse_legs)}")
            if near_nonuse_legs:
                print(
                    f"  Near non-use legs: {', '.join(near_nonuse_legs)} "
                    "(force reduction >85% and load duty <0.25)"
                )
            if light_touch_legs:
                print(
                    "  Light-touch legs: "
                    f"{', '.join(light_touch_legs)} "
                    "(contact duty increased but load-bearing duty decreased)"
                )
            if drag_like_legs:
                print(
                    "  Drag-like light-contact legs: "
                    f"{', '.join(drag_like_legs)} "
                    "(high contact duty but low load-bearing duty)"
                )
            if overuse_legs:
                print(f"  Weak protection legs: {', '.join(overuse_legs)}")
            if len(nonuse_legs) >= 2:
                print("  Verdict: limb non-use / tripod-like gait, not antalgic limping")
            elif nonuse_legs:
                print("  Verdict: mixed result; some legs show non-use, not clean enough for main figure")
            elif near_nonuse_legs:
                print("  Verdict: partial result; near non-use still too strong for the main paper figure")
            elif protected_force >= 3 and (not duty_reductions or protected_duty >= 3):
                print("  Verdict: usable antalgic Phase2 teacher candidate")
            else:
                print("  Verdict: not yet a reliable antalgic teacher")
            print(f"  Paper-grade candidate: {paper_grade_candidate}")
    if "idx_acc" in probe_results:
        print(
            f"\n[LSTM Injury Classification] Accuracy: {probe_results['idx_acc']:.1f}%"
        )
        metrics["lstm_injury_classification_acc_pct"] = float(probe_results["idx_acc"])
    if "spl_r2" in probe_results:
        print(
            f"[LSTM Splint Estimation] R2={probe_results['spl_r2']:.3f}, MAE={probe_results['spl_mae']:.4f}m"
        )
        metrics["lstm_splint_r2"] = float(probe_results["spl_r2"])
        metrics["lstm_splint_mae_m"] = float(probe_results["spl_mae"])
    metrics_path = args_cli.metrics_json or os.path.join(save_dir, "student_analysis_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[SAVED] {metrics_path}")
    print("=" * 80)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
