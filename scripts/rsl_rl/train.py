# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL.

사용 예:
    PYTHONPATH=<repo>/source/go1_lab python train.py --phase 1 --headless --run_tag P1-001
"""
import math
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
parser.add_argument("--phase_config_path", type=str, required=True, help="phase YAML 경로 지정")
parser.add_argument("--common_config_path", type=str, required=False, default=f"{current_file}/configs/common.yaml", help="Path to YAML log config")
parser.add_argument("--log_config_path", type=str, required=False, default=f"{current_file}/configs/logger.yaml" ,help="Path to YAML log config") 
parser.add_argument("--run_tag", type=str, default="", help="실험 구분 이름. 예: z1_air050")
parser.add_argument("--debug_obs", action="store_true", help="학습 전 구간의 정책 입력(raw/normalized)과 출력 action 을 CSV 저장 (log_dir/obs_debug/).",)
parser.add_argument("--debug_obs_envs", type=int, default=4, help="--debug_obs 에서 act() 호출마다 기록할 env 수")
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

phase_config_path = Path(args.phase_config_path).expanduser().resolve()

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
from utils.rsl_rl_compat import resolve_runner_class

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import go1_lab.tasks  # noqa: F401

from go1_lab.tasks.manager_based.go1_lab.mdp.obs_normalizer import (
    install_obs_scaler, obs_scale_summary,
)
from utils.obs_debug_dump import ObsDebugDumper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# check minimum supported rsl-rl version
rsl_rl_version_check()

def inject_action_std_safety(policy, min_action_std: float) -> None:
    """Action 표준편차가 YAML의 하한보다 작아지지 않게 한다."""

    min_action_std = float(min_action_std)

    if min_action_std <= 0.0:
        raise ValueError(
            "min_action_std must be greater than zero, "
            f"got {min_action_std}"
        )

    if not hasattr(policy, "_update_distribution"):       
        raise RuntimeError(
            "update_distribution 계열 메서드를 찾지 못했습니다 — rsl_rl 버전 확인 필요"
        )
    
    original_update_distribution = policy._update_distribution

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

    policy._update_distribution = (safe_update_distribution)
    
def update_agent_cfg(agent_cfg, config: ExperimentConfig, run_name: str):
    train = config.train
    
    agent_cfg.seed = train.seed
    agent_cfg.logger = config.rsl_logger
    # TensorBoard 로그 디렉터리와 RSL-RL 내부 run 이름을 동일하게 유지한다.
    # 따라서 --run_tag가 두 위치에 모두 기록된다.
    agent_cfg.run_name = run_name
    agent_cfg.experiment_name = train.project_name
    agent_cfg.max_iterations = train.max_iterations
    agent_cfg.num_steps_per_env = train.num_steps_per_env

    # 탐색 노이즈는 전 phase 공통. phase 3 에서는 student rollout 의 데이터 수집 노이즈다.
    exploration = config.exploration
    agent_cfg.policy.noise_std_type = exploration.noise_std_type
    agent_cfg.policy.init_noise_std = exploration.init_noise_std


    if config.phase == "phase3":
        agent_cfg.algorithm.gradient_length = train.gradient_length
        # 부목 길이 범위는 env yaml 과 동일하게 (Phase3Distillation 이 검증/정규화에 사용)
        lo, hi = config.environment.values["peg_leg"]["splint_length_range"]
        agent_cfg.algorithm.splint_length_range = (float(lo), float(hi))
        agent_cfg.algorithm.splint_loss_coef = train.splint_loss_coef
        agent_cfg.algorithm.vel_loss_coef = train.vel_loss_coef

        mn = train.mse_norm
        agent_cfg.policy.mse_norm_enable = bool(mn is not None and mn.enable)
        if agent_cfg.policy.mse_norm_enable:
            agent_cfg.policy.action_mean = mn.action_mean
            agent_cfg.policy.action_pstd = mn.action_pstd
            agent_cfg.policy.splint_mean = mn.splint_mean
            agent_cfg.policy.splint_std = mn.splint_std
            agent_cfg.policy.vel_mean = mn.vel_mean
            agent_cfg.policy.vel_pstd = mn.vel_pstd

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
    
    # runner 생성
    # cfg 의 class_name 으로 선택 (OnPolicyRunner / Phase3DistillationRunner)
    runner_cls = resolve_runner_class(agent_cfg.class_name)
    runner = runner_cls(
        env=env,
        train_cfg=agent_cfg_dict,
        log_dir=str(log_dir),
        device=agent_cfg.device,
    )
    
    if config.phase == "phase3":
        app_logger.info("Output norm (train.mse_norm): %s", runner.alg.policy.output_norm_summary())

    nm = config.train.normalize
    if nm.enable:
        replaced = install_obs_scaler(runner.alg.policy, env.get_observations(), nm.obs, nm.priv)
        app_logger.info(
            "Obs scale (train.normalize): enabled — %s | obs %s, priv %s",
            ", ".join(replaced), nm.obs, nm.priv,
        )
    else:
        app_logger.info("Obs scale (train.normalize): disabled — raw obs")
        
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

        if mode in ("warmstart", "distill"):
            # ── 탐색 노이즈 재설정 ────────────────────────────────────
            # log_std / std 는 nn.Parameter 라 runner.load() 가 체크포인트 값으로
            # 덮어쓴다. 그래서 yaml 의 init_noise_std 는 warmstart 에서 그냥 무시된다.
            # 새 과제(부상 보행)를 이전 phase 의 수렴된 좁은 std 로 시작하지 않도록
            # 여기서 다시 써 준다. 네트워크 가중치는 그대로 두고 std 만 되돌린다.
            if checkpoint_cfg.reset_noise_std:
                policy = runner.alg.policy
                init_std = float(config.exploration.init_noise_std)

                with torch.no_grad():
                    if hasattr(policy, "log_std"):
                        policy.log_std.data.fill_(math.log(init_std))
                    elif hasattr(policy, "std"):
                        policy.std.data.fill_(init_std)
                    else:
                        app_logger.warning(
                            "log_std / std 를 찾지 못해 탐색 노이즈를 재설정하지 못했습니다."
                        )

                app_logger.info(
                    "Exploration std reset: checkpoint 값 대신 init_noise_std=%g 로 시작",
                    init_std,
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
    app_logger.info("Obs scale (실제 적용): %s", obs_scale_summary(runner.alg.policy))

    app_logger.info(
        "PPO exploration: "
        "type=%s, init_std=%g, min_std=%g, enforce=%s",
        config.exploration.noise_std_type,
        config.exploration.init_noise_std,
        config.exploration.min_action_std,
        config.exploration.enforce_min_std,
    )
            
    # 학습 전 구간을 CSV 로 흘려쓴다. envs_per_step 이 용량 손잡이다
    # (4 → 약 584MB, 16 → 2.3GB, 64 → 9.3GB @ phase1 5000 iter).
    obs_dumper = ObsDebugDumper(
        runner.alg.policy, log_dir / "obs_debug",
        env=env,
        envs_per_step=args.debug_obs_envs,
        total_calls=agent_cfg.max_iterations * steps_per_iteration,
        logger=app_logger,
    ) if args.debug_obs else None

    # 학습 시작
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    if obs_dumper is not None:
        obs_dumper.close()

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
