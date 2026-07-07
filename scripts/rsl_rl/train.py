# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from peg_leg_action_wrapper import PegLegActionMaskWrapper  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate (default: uses env config default, typically 4096).")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Go1-Lab-v0",
    help="Name of the task (default: Template-Go1-Lab-v0).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--teacher_ckpt_path",
    type=str,
    default=None,
    help="Path to teacher checkpoint for distillation (optional).",
)
parser.add_argument(
    "--warmstart_ckpt_path",
    type=str,
    default=None,
    help=(
        "Path to a checkpoint to warmstart the (OnPolicy) runner from, across "
        "experiments. Typically used for Phase 2 Teacher to load Phase 1 "
        "Healthy pretrain weights. Loaded only if --resume is not used and "
        "agent is not distillation."
    ),
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    default=False,
    help="Warm-start OnPolicy training from an Isaac Lab published pretrained checkpoint.",
)
parser.add_argument(
    "--pretrained_task",
    type=str,
    default=None,
    help=(
        "Task id used to resolve the Isaac Lab published checkpoint when "
        "--use_pretrained_checkpoint is set. Defaults to --task."
    ),
)
parser.add_argument(
    "--teacher_experiment_name",
    type=str,
    default="unitree_go1_rough_teacher",
    help="Teacher experiment folder under logs/rsl_rl used when --teacher_ckpt_path is not set.",
)
parser.add_argument(
    "--teacher_load_run",
    type=str,
    default=None,
    help="Teacher run folder regex/name used for checkpoint lookup.",
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    default="model_.*.pt",
    help="Teacher checkpoint regex/name used for checkpoint lookup.",
)
parser.add_argument(
    "--min_action_std",
    type=float,
    default=1.0e-3,
    help="Lower bound for action std to avoid invalid Normal distribution.",
)
parser.add_argument(
    "--use_peg_leg_action_mask",
    action="store_true",
    default=False,
    help="Enable peg-leg calf action masking wrapper.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs is not None and args_cli.num_envs > 32768:
    raise ValueError(
        f"--num_envs={args_cli.num_envs} is too large for a single-PC Isaac Lab run. "
        "Use a value such as 4096, 6144, or 8192. "
        "If this was meant to be 8192 or 4096, pass that value explicitly."
    )

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
from torch.distributions import Normal

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _inject_action_std_safety(policy, min_action_std: float) -> None:
    """Clamp action std to keep Normal distribution valid."""
    if not hasattr(policy, "update_distribution"):
        return

    original_update_distribution = policy.update_distribution

    def safe_update_distribution(obs):
        original_update_distribution(obs)
        if not hasattr(policy, "distribution") or policy.distribution is None:
            return

        mean = policy.distribution.mean
        std = policy.distribution.stddev
        std = torch.nan_to_num(std, nan=float(min_action_std), posinf=1.0, neginf=float(min_action_std))
        std = torch.clamp(std, min=float(min_action_std))
        policy.distribution = Normal(mean, std)

        with torch.no_grad():
            if hasattr(policy, "std"):
                policy.std.data = torch.nan_to_num(
                    policy.std.data,
                    nan=float(min_action_std),
                    posinf=1.0,
                    neginf=float(min_action_std),
                )
                policy.std.data.clamp_(min=float(min_action_std))
            if hasattr(policy, "log_std"):
                min_log_std = float(torch.log(torch.tensor(float(min_action_std))).item())
                policy.log_std.data = torch.nan_to_num(
                    policy.log_std.data,
                    nan=min_log_std,
                    posinf=0.0,
                    neginf=min_log_std,
                )
                policy.log_std.data.clamp_(min=min_log_std)

    policy.update_distribution = safe_update_distribution


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    resume_path = None
    warmstart_path = None
    teacher_resume_path = None
    is_distillation = agent_cfg.class_name == "DistillationRunner"

    # Distillation에서는 teacher checkpoint를 기준으로 시작합니다.
    # (RslRlBaseRunnerCfg 문서상 resume 플래그는 distillation에서 무시됨)
    if is_distillation:
        if args_cli.teacher_ckpt_path is not None:
            teacher_resume_path = os.path.abspath(args_cli.teacher_ckpt_path)
        else:
            teacher_log_root = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment_name))
            teacher_resume_path = get_checkpoint_path(
                teacher_log_root,
                args_cli.teacher_load_run or ".*",
                args_cli.teacher_checkpoint,
            )
    elif agent_cfg.resume:
        # OnPolicyRunner인 경우에만 명시적 resume를 적용합니다.
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.use_pretrained_checkpoint:
        pretrained_task = args_cli.pretrained_task or args_cli.task
        pretrained_task = pretrained_task.split(":")[-1].replace("-Play", "")
        warmstart_path = get_published_pretrained_checkpoint("rsl_rl", pretrained_task)
        if not warmstart_path:
            raise FileNotFoundError(
                "Isaac Lab published pretrained checkpoint is unavailable for "
                f"task={pretrained_task!r}."
            )
    elif args_cli.warmstart_ckpt_path is not None:
        # Cross-experiment warmstart (e.g. Phase1 Healthy → Phase2 Teacher).
        warmstart_path = os.path.abspath(args_cli.warmstart_ckpt_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 필요할 때만 고장 다리 calf action masking 적용
    enable_mask = args_cli.use_peg_leg_action_mask or bool(getattr(env_cfg, "use_peg_leg_action_mask", False))
    if enable_mask:
        env = PegLegActionMaskWrapper(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    agent_cfg_dict = agent_cfg.to_dict()

    # RSL-RL 3.0.1+ 버전 호환성을 위한 Patch
    # 1. policy 구성 요소에 class_name 주입
    if "policy" in agent_cfg_dict:
        policy_cfg = agent_cfg_dict["policy"]
        for component in ["actor", "critic", "student", "teacher"]:
            if component in policy_cfg and isinstance(policy_cfg[component], dict):
                if "class_name" not in policy_cfg[component]:
                    # 기본값으로 MLP 설정 (Isaac Lab 표준)
                    policy_cfg[component]["class_name"] = "MLP"

    # 2. algorithm 설정에서 PPO 클래스가 지원하지 않는 키워드 제거
    if "algorithm" in agent_cfg_dict:
        alg_cfg = agent_cfg_dict["algorithm"]
        # rsl-rl 3.0.1 PPO.__init__에서 제거된 키워드들 (버전 불일치 해결)
        taboo_keys = ["optimizer", "config_class", "share_cnn_encoders"]
        for taboo_key in taboo_keys:
            if taboo_key in alg_cfg:
                alg_cfg.pop(taboo_key)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if (not is_distillation) and agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif (not is_distillation) and warmstart_path is not None:
        # Cross-experiment warmstart (optimizer/state 는 리셋, 가중치만 이식).
        print(f"[INFO]: Warmstarting from checkpoint: {warmstart_path}")
        runner.load(warmstart_path, load_optimizer=False)
    elif is_distillation:
        print(f"[INFO]: Loading teacher checkpoint for distillation from: {teacher_resume_path}")
        # DistillationRunner는 teacher 파라미터가 먼저 로드되어야 학습을 시작할 수 있습니다.
        runner.load(teacher_resume_path, load_optimizer=False)

    # 학습 안정화: 분포 표준편차 하한 강제
    _inject_action_std_safety(runner.alg.policy, min_action_std=args_cli.min_action_std)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
