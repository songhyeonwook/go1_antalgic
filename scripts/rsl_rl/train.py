# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL.

사용 예:
    PYTHONPATH=<repo>/source/go1_lab python train.py --phase 1 --headless --run_tag P1-001
"""

import argparse
import sys
import traceback
from isaaclab.app import AppLauncher

# added
from utils.config_builder import ExperimentConfig, load_experiment_config, read_yaml
from pathlib import Path
from utils.prettyjson import prettyjson
import json
from datetime import datetime
from dataclasses import asdict
from utils.logger import create_logger, StreamToLogger, redirect_python_streams
import logging


def rsl_rl_version_check():
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

def set_log(log_dir, log_config_path, run_name):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_configs = read_yaml(log_config_path)

    app_logger = create_logger(
        name=run_name,
        log_directory=str(log_dir),
        log_cfgs=log_configs,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = StreamToLogger(
        app_logger,
        logging.INFO,
        original_stdout,
    )

    sys.stderr = StreamToLogger(
        app_logger,
        logging.WARNING,
        original_stderr,
    )
    
    return app_logger
    
"""
python3 train.py --phase 1 --run_tag P1-004
"""

current_file = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True, help="Training phase: 1, 2, or 3.") 
parser.add_argument("--common_config_path", type=str, required=False, default=f"{current_file}/configs/common.yaml", help="Path to YAML log config")
parser.add_argument("--log_config_path", type=str, required=False, default=f"{current_file}/configs/logger.yaml" ,help="Path to YAML log config") 
parser.add_argument("--run_tag", type=str, default="", help="실험 구분 이름. 예: z1_air050")
AppLauncher.add_app_launcher_args(parser)

# argparse가 아는 인자와 Hydra 인자를 분리
args, hydra_args = parser.parse_known_args()
# Hydra가 argparse용 인자를 다시 읽지 않도록 Hydra 인자만 남김
sys.argv = [
    sys.argv[0],
    *hydra_args,
    "hydra/job_logging=none", 
    "hydra.output_subdir=null",
    "hydra.run.dir=.",
]

phase_config_path = (
    current_file / "configs" / "phase" / f"phase{args.phase}.yaml"
)


config = load_experiment_config(
    phase_path=phase_config_path,
    common_path=args.common_config_path    
)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

run_tag = args.run_tag.strip()
tag_suffix = f"_{run_tag}" if run_tag else ""
run_name = f"{timestamp}_{config.phase}_s{config.train.seed}{tag_suffix}"

log_dir = (
    current_file
    / "logs"
    / config.train.project_name
    / run_name
)

app_logger = set_log(log_dir, args.log_config_path, run_name)


config_snapshot = asdict(config)
config_snapshot["runtime"] = {
    "run_name": run_name,
    "run_tag": run_tag,
    "phase_config_path": str(phase_config_path),
    "log_dir": str(log_dir),
    "cli_args": vars(args).copy(),
}


with (log_dir / "config.json").open("w", encoding="utf-8") as file:
    json.dump(config_snapshot, file, indent=4, default=str)
    
app_logger.info(
    "Training configuration:\n%s",
    prettyjson(config_snapshot),
)

# 어느 저장소의 go1_lab 을 쓸지는 PYTHONPATH 로 직접 지정한다 (사본 간 혼동 방지).
# Isaac Sim 부팅(수십 초) 후에 ImportError 로 죽지 않도록 여기서 미리 검사한다.
import importlib.util

if importlib.util.find_spec("go1_lab") is None:
    _repo_hint = Path(__file__).resolve().parents[2] / "source" / "go1_lab"
    sys.exit(
        "[train] go1_lab 패키지를 찾을 수 없습니다. 사용할 저장소를 직접 지정해 실행하세요:\n"
        f"  PYTHONPATH={_repo_hint} python train.py --phase N\n"
        "  (다른 저장소 사본을 쓰려면 그 저장소의 source/go1_lab 경로를 지정)"
    )

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

redirect_python_streams(app_logger)

# Isaac Sim 실행 이후 import
import importlib.metadata as metadata
import platform

import gymnasium as gym
import torch
from torch.distributions import Normal
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import go1_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# check minimum supported rsl-rl version
rsl_rl_version_check()

def inject_action_std_safety(policy, min_action_std: float) -> None:
    """Action 표준편차가 YAML의 하한보다 작아지지 않게 한다."""

    if not hasattr(policy, "update_distribution"):
        return

    min_action_std = float(min_action_std)

    if min_action_std <= 0.0:
        raise ValueError(
            "min_action_std must be greater than zero, "
            f"got {min_action_std}"
        )

    original_update_distribution = (policy.update_distribution)

    def safe_update_distribution(obs):
        # scalar 방식에서는 원래 distribution을 만들기 전에
        # std 파라미터를 먼저 양수로 보정
        with torch.no_grad():
            if hasattr(policy, "std"):
                policy.std.data = torch.nan_to_num(
                    policy.std.data,
                    nan=min_action_std,
                    posinf=1.0,
                    neginf=min_action_std,
                )
                policy.std.data.clamp_(
                    min=min_action_std
                )

            if hasattr(policy, "log_std"):
                min_log_std = torch.log(
                    torch.tensor(
                        min_action_std,
                        device=policy.log_std.device,
                    )
                ).item()

                policy.log_std.data = torch.nan_to_num(
                    policy.log_std.data,
                    nan=min_log_std,
                    posinf=0.0,
                    neginf=min_log_std,
                )
                policy.log_std.data.clamp_(
                    min=min_log_std
                )

        # RSL-RL이 Normal distribution 생성
        original_update_distribution(obs)

        if (
            not hasattr(policy, "distribution")
            or policy.distribution is None
        ):
            return

        mean = policy.distribution.mean
        std = policy.distribution.stddev

        safe_std = torch.nan_to_num(
            std,
            nan=min_action_std,
            posinf=1.0,
            neginf=min_action_std,
        )

        safe_std = torch.clamp(
            safe_std,
            min=min_action_std,
        )

        policy.distribution = Normal(
            mean,
            safe_std,
        )

    policy.update_distribution = (safe_update_distribution)
    
def update_agent_cfg(agent_cfg, config: ExperimentConfig, run_name: str):
    train = config.train
    
    
    agent_cfg.seed = train.seed
    agent_cfg.logger = config.rsl_logger
    # TensorBoard 로그 디렉터리와 RSL-RL 내부 run 이름을 동일하게 유지한다.
    # 따라서 --run_tag가 두 위치에 모두 기록된다.
    agent_cfg.run_name = run_name
    agent_cfg.experiment_name = train.project_name
    agent_cfg.max_iterations = train.max_iterations
    
    if config.phase in {"phase1", "phase2"}:
        exploration = config.exploration
        agent_cfg.policy.noise_std_type = (exploration.noise_std_type)
        agent_cfg.policy.init_noise_std = (exploration.init_noise_std)

    return agent_cfg
    
def update_env_cfg(env_cfg, config: ExperimentConfig, log_dir: str, steps_per_iteration: int):
    train = config.train

    steps_per_iteration = int(steps_per_iteration)
    
    env_cfg.scene.num_envs = train.num_envs
    env_cfg.sim.device = config.common["device"]
    env_cfg.seed = train.seed
    env_cfg.log_dir = log_dir

    env_cfg.apply_environment_settings(
        config.environment.values,
        steps_per_iteration
    )

    return env_cfg

def patch_rsl_rl_agent_cfg(agent_cfg_dict: dict) -> dict:
    """RSL-RL 3.0.1+와의 설정 호환성을 위해 agent config를 수정한다.

    처리 내용:
    1. policy의 actor, critic, student, teacher 설정에 class_name이 없으면 기본값 "MLP"를 추가한다.
    2. PPO 생성자가 지원하지 않는 algorithm 키를 제거한다.
    """
    policy_cfg = agent_cfg_dict.get("policy")

    if isinstance(policy_cfg, dict):
        policy_components = (
            "actor",
            "critic",
            "student",
            "teacher",
        )

        for component_name in policy_components:
            component_cfg = policy_cfg.get(component_name)

            if isinstance(component_cfg, dict):
                component_cfg.setdefault(
                    "class_name",
                    "MLP",
                )

    algorithm_cfg = agent_cfg_dict.get("algorithm")

    if isinstance(algorithm_cfg, dict):
        unsupported_keys = (
            "optimizer",
            "config_class",
            "share_cnn_encoders",
        )

        for key in unsupported_keys:
            algorithm_cfg.pop(key, None)

    return agent_cfg_dict    



@hydra_task_config(
    config.train.task,
    config.train.agent,
)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    train_cfg = config.train
    checkpoint_cfg = config.checkpoint

    # YAML 설정을 Hydra가 만든 설정 객체에 반영
    agent_cfg = update_agent_cfg(agent_cfg, config, run_name)
    
    steps_per_iteration = int(agent_cfg.num_steps_per_env)
    
    env_cfg = update_env_cfg(env_cfg, config, str(log_dir), steps_per_iteration)

    app_logger.info("Phase: %s", config.phase)
    app_logger.info("Task: %s", train_cfg.task)
    app_logger.info("Agent entry point: %s", train_cfg.agent)
    app_logger.info("Number of environments: %d", env_cfg.scene.num_envs)
    app_logger.info("Maximum iterations: %d", agent_cfg.max_iterations)
    app_logger.info("Rollout steps per PPO iteration: %d", steps_per_iteration)
    
    # create isaac environment
    # (부상 action mask 는 Go1LabEnv.step() 내부에서 수행 — 별도 래퍼 불필요)
    env = gym.make(train_cfg.task, cfg=env_cfg, render_mode=None)

    # RSL-RL 환경 wrapper
    env = RslRlVecEnvWrapper(env)

    # Agent config 변환
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict = patch_rsl_rl_agent_cfg(agent_cfg_dict)
    
    # runner 생성
    if config.phase in ("phase1", "phase2"):
        runner = OnPolicyRunner(
            env=env,
            train_cfg=agent_cfg_dict,
            log_dir=str(log_dir),
            #logger=app_logger,
            device=agent_cfg.device,
        )

    elif config.phase == "phase3":
        runner = DistillationRunner(
            env=env,
            train_cfg=agent_cfg_dict,
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )

    else:
        raise ValueError(
            f"Unsupported phase: {config.phase}"
        )

    # 체크포인트
    mode = checkpoint_cfg.mode.strip().lower()
    

    if mode == "scratch": # phase 2, phase 3 
        pass
    else:
        if not checkpoint_cfg.teacher:
            raise ValueError(
                f"checkpoint.teacher is required "
                f"when mode={mode!r}"
            )

        checkpoint_path = checkpoint_cfg.teacher.strip()
        
        runner.load(
            checkpoint_path,
            load_optimizer=checkpoint_cfg.load_optimizer,
        )

        if mode == "resume":
            restored_env_steps = int(runner.current_learning_iteration * steps_per_iteration)
            env.unwrapped.common_step_counter = (restored_env_steps)
            env.reset()
            app_logger.info("Resumed iteration: %d",runner.current_learning_iteration,)
            app_logger.info("Resumed environment step counter: %d", restored_env_steps,)
        
        elif checkpoint_cfg.reset_iteration:
            runner.current_learning_iteration = 0
                
    
        inject_action_std_safety(runner.alg.policy, min_action_std=(config.exploration.min_action_std),)

        app_logger.info(
            "PPO exploration: "
            "type=%s, init_std=%g, min_std=%g, enforce=%s",
            config.exploration.noise_std_type,
            config.exploration.init_noise_std,
            config.exploration.min_action_std,
            config.exploration.enforce_min_std,
        )
            
        
    # 학습 시작
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    
if __name__ == "__main__":
    main_error = None

    try:
        main()

    except BaseException as error:
        main_error = error
        error_type = type(error).__name__

        print(
            f"\n[FATAL] main() terminated with {error_type}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)

        app_logger.critical(
            "main() terminated with %s: %s",
            error_type,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )

        raise

    finally:
        try:
            app_logger.info("Closing Isaac Sim application.")
            simulation_app.close()
        except BaseException as close_error:
            print(
                (
                    "\n[FATAL] simulation_app.close() failed with "
                    f"{type(close_error).__name__}: {close_error}"
                ),
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)

            app_logger.critical(
                "simulation_app.close() failed with %s: %s",
                type(close_error).__name__,
                close_error,
                exc_info=(
                    type(close_error),
                    close_error,
                    close_error.__traceback__,
                ),
            )

            # main()이 정상 종료된 경우에는 close 오류를 그대로 전파합니다.
            # main() 예외가 이미 있으면 close 오류가 원래 예외를 덮지 않게 합니다.
            if main_error is None:
                raise
