from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

from utils.config_builder import load_experiment_config


# ============================================================
"""
python3 test.py --phase 1 --checkpoint /home/unicon/wj/go1_antalgic/scripts/rsl_rl/logs/unitree_go1_antalgic/2026_08_04_09_56_38_phase1_s42_flatori0.5/model_5999.pt --num_envs 1 --seed 42  --fixed_x 0.3
"""
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent 

parser = argparse.ArgumentParser(
    description="Play an RSL-RL checkpoint using the same config pipeline as train.py."
)

parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=1, help="설정을 가져올 phase 번호")
parser.add_argument("--checkpoint", type=str, required=True, help="재생할 model_*.pt 경로",)
parser.add_argument( "--common_config_path", type=str, default=str(SCRIPT_DIR / "configs" / "common.yaml"),)
parser.add_argument( "--num_envs", type=int, default=1,)
parser.add_argument( "--seed", type=int, default=None,)
parser.add_argument( "--real_time", action="store_true", help="실제 시간 속도에 맞춰 재생",)
parser.add_argument( "--video", action="store_true", help="재생 영상을 mp4로 저장",)
parser.add_argument( "--video_length", type=int, default=1000, help="영상 길이. 현재 설정에서 1000 step은 약 20초",)
parser.add_argument( "--fixed_x", type=float, default=None, help="고정 전진 명령. 예: 0.2",)
parser.add_argument( "--fixed_yaw", type=float, default=None, help="고정 yaw 명령. 예: 0.0",)
parser.add_argument( "--clean", action="store_true", help="마찰·질량 랜덤화, push, 관측 노이즈를 끄고 기본 보행만 확인",)
parser.add_argument("--peg_leg", type=str, choices=("normal", "fl", "fr", "rl", "rr", "balanced"), default=None, help="YAML의 eval.peg_leg를 임시로 덮어쓸 평가 조건",)
parser.add_argument(
    "--splint_length",
    type=float,
    nargs="+",
    default=None,
    metavar="L",
    help="부목 길이 지정. 값 1개면 고정(예: 0.4), 2개면 범위(예: 0.33 0.45). 미지정 시 YAML 값 사용",
)
parser.add_argument(
    "--compare_all",
    action="store_true",
    help="Phase 2/3에서 Normal/FL/FR/RL/RR 다섯 조건을 고정 배치하여 한 화면에서 비교",
)

AppLauncher.add_app_launcher_args(parser)

args, hydra_args = parser.parse_known_args()

if args.compare_all:
    if args.phase not in (2, 3):
        parser.error("--compare_all은 --phase 2 또는 --phase 3와 함께 사용해야 합니다.")
    args.num_envs = 5

if args.splint_length is not None and len(args.splint_length) not in (1, 2):
    parser.error("--splint_length는 값 1개(고정) 또는 2개(범위)만 받습니다.")

if args.video:
    args.enable_cameras = True



phase_config_path = (
    SCRIPT_DIR
    / "configs"
    / "phase"
    / f"phase{args.phase}.yaml"
)

load_kwargs = {
    "phase_path": phase_config_path,
    "common_path": args.common_config_path,
}


config = load_experiment_config(**load_kwargs)

eval_peg_leg = (
    "balanced"
    if args.compare_all
    else (
        args.peg_leg.strip().lower()
        if args.peg_leg is not None
        else config.evaluation.peg_leg
    )
)

print(f"[INFO] Phase config : {phase_config_path}")
print(f"[INFO] Env config   : {config.environment.path}")
print(f"[INFO] Task         : {config.train.task}")
print(f"[INFO] Agent        : {config.train.agent}")
print(f"[INFO] Phase        : {config.phase}")


sys.argv = [
    sys.argv[0],
    *hydra_args,
    "hydra/job_logging=disabled",
    "hydra.output_subdir=null",
    "hydra.run.dir=.",
]


app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# Isaac Sim 시작 이후 import
import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_lab.tasks  # noqa: F401
from go1_lab.splint import SPLINT_MAX, SPLINT_MIN

from peg_leg_action_wrapper import PegLegActionMaskWrapper


def patch_rsl_rl_agent_cfg(agent_cfg_dict: dict) -> dict:
    policy_cfg = agent_cfg_dict.get("policy")

    if isinstance(policy_cfg, dict):
        for component_name in (
            "actor",
            "critic",
            "student",
            "teacher",
        ):
            component_cfg = policy_cfg.get(component_name)

            if isinstance(component_cfg, dict):
                component_cfg.setdefault("class_name", "MLP")

    algorithm_cfg = agent_cfg_dict.get("algorithm")

    if isinstance(algorithm_cfg, dict):
        for unsupported_key in (
            "optimizer",
            "config_class",
            "share_cnn_encoders",
        ):
            algorithm_cfg.pop(unsupported_key, None)

    return agent_cfg_dict


@hydra_task_config(
    config.train.task,
    config.train.agent,
)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint를 찾을 수 없습니다: {checkpoint_path}"
        )

    log_dir = checkpoint_path.parent

    seed = args.seed if args.seed is not None else config.train.seed
    device = (
        args.device
        if args.device is not None
        else config.common.get("device", "cuda:0")
    )

    agent_cfg.seed = seed
    agent_cfg.logger = config.rsl_logger
    agent_cfg.run_name = config.phase
    agent_cfg.experiment_name = config.train.project_name
    agent_cfg.device = device

    # train.py와 동일한 policy 모듈을 생성해야 checkpoint의 std/log_std
    # 파라미터 이름과 형상이 일치한다. 평가 policy는 deterministic이라 이 값을
    # 샘플링에 사용하지 않지만, runner 생성 및 checkpoint 로드 전에는 반드시 맞춘다.
    if config.phase in {"phase1", "phase2", "phase3"}:
        exploration = config.exploration
        agent_cfg.policy.noise_std_type = exploration.noise_std_type
        agent_cfg.policy.init_noise_std = exploration.init_noise_std

    steps_per_iteration = int(agent_cfg.num_steps_per_env)
    
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = seed
    env_cfg.log_dir = str(log_dir)

    # 핵심:
    # phase1.yaml이 가리키는 healthy.yaml 전체를 적용한다.
    env_cfg.apply_environment_settings(
        config.environment.values,
        steps_per_iteration,
        eval_peg_leg=eval_peg_leg,
    )

    # 다섯 조건 비교 모드에서는 환경 번호별 조건을 고정한다.
    # balanced_random은 일부 환경만 리셋될 때 조건 구성이 흐트러질 수 있으므로,
    # env-id 기반 배정으로 env0=Normal, env1=FL, env2=FR, env3=RL, env4=RR를 유지한다.
    if args.compare_all:
        peg_event = env_cfg.events.randomize_peg_leg_actuation
        if peg_event is None:
            raise RuntimeError(
                "--compare_all을 사용하려면 peg-leg reset event가 활성화되어야 합니다."
            )
        peg_event.params["target_leg"] = "balanced_env"
        peg_event.params["healthy_slots"] = 1
        peg_event.params["prob_peg_leg"] = 0.8

    # 부목 길이를 CLI에서 지정한 경우 YAML의 splint_length_range를 덮어쓴다.
    # 값 1개면 (L, L) 고정, 2개면 균등분포 범위로 사용한다.
    if args.splint_length is not None:
        peg_event = env_cfg.events.randomize_peg_leg_actuation
        if peg_event is None:
            raise RuntimeError(
                "--splint_length를 사용하려면 peg-leg reset event가 활성화되어야 합니다."
            )
        lo = min(args.splint_length)
        hi = max(args.splint_length)
        if lo < SPLINT_MIN or hi > SPLINT_MAX:
            raise ValueError(
                f"--splint_length {args.splint_length}는 USD 설계 한계 "
                f"[{SPLINT_MIN}, {SPLINT_MAX}] 안에 있어야 합니다."
            )
        peg_event.params["splint_length_range"] = (lo, hi)

    # 평가는 지정한 조건(normal/FL/FR/RL/RR/balanced)을 그대로 유지해야 한다.
    # 학습용 peg-leg curriculum이 남아 있으면 환경 reset 직전에 계산되어
    # 위에서 설정한 평가용 prob_peg_leg를 curriculum 초기값으로 덮어쓸 수 있다.
    # gym.make() 전에 term을 제거하여 reset event가 평가용 설정만 사용하게 한다.
    if (
        hasattr(env_cfg, "curriculum")
        and env_cfg.curriculum is not None
        and hasattr(env_cfg.curriculum, "peg_leg_difficulty")
    ):
        env_cfg.curriculum.peg_leg_difficulty = None

    print(
        "[INFO] Rollout steps per iteration: "
        f"{steps_per_iteration}"
    )
    print("[INFO] Peg-leg curriculum disabled for evaluation")
    
    # apply_environment_settings 이후 시각화 값을 다시 보장
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = device
    env_cfg.log_dir = str(log_dir)

    # --------------------------------------------------------
    # 카메라 설정
    # --------------------------------------------------------

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-3.0, 2.0, 1.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.35)

    if args.compare_all:
        # 다섯 조건은 모두 생성하되, 카메라는 선택한 로봇 한 마리를 가까이 추적한다.
        # IsaacLab > Viewer Settings > Environment Index(1~5)만 바꾸면
        # Normal/FL/FR/RL/RR 사이를 같은 거리와 각도로 전환할 수 있다.
        env_cfg.scene.env_spacing = 2.5

    # --------------------------------------------------------
    # 선택 사항: 명령 속도 고정
    # --------------------------------------------------------

    command_ranges = env_cfg.commands.base_velocity.ranges

    if args.fixed_x is not None:
        command_ranges.lin_vel_x = (
            args.fixed_x,
            args.fixed_x,
        )
        command_ranges.lin_vel_y = (0.0, 0.0)

    if args.fixed_yaw is not None:
        command_ranges.ang_vel_z = (
            args.fixed_yaw,
            args.fixed_yaw,
        )

    # --------------------------------------------------------
    # 선택 사항: 랜덤화 없는 clean 환경
    # --------------------------------------------------------

    if args.clean:
        for event_name in (
            "physics_material",
            "add_base_mass",
            "push_robot",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.enable_corruption = False

        print("[INFO] Clean evaluation enabled")
        print("[INFO] Friction/mass randomization, push, observation noise disabled")

    print(f"[INFO] Checkpoint    : {checkpoint_path}")
    print(f"[INFO] Device        : {device}")
    print(f"[INFO] Seed          : {seed}")
    print(f"[INFO] Environments  : {env_cfg.scene.num_envs}")
    print(f"[INFO] Peg-leg eval  : {eval_peg_leg}")
    if args.splint_length is not None:
        splint_range = env_cfg.events.randomize_peg_leg_actuation.params["splint_length_range"]
        print(f"[INFO] Splint length : {splint_range[0]:.3f} ~ {splint_range[1]:.3f} m (CLI override)")
    if config.phase in {"phase1", "phase2", "phase3"}:
        print(
            "[INFO] Policy noise  : "
            f"type={agent_cfg.policy.noise_std_type}, "
            f"init_std={agent_cfg.policy.init_noise_std}"
        )
    if args.compare_all:
        print("[INFO] Fixed mapping : env0=Normal, env1=FL, env2=FR, env3=RL, env4=RR")
    print(f"[INFO] lin_vel_x     : {command_ranges.lin_vel_x}")
    print(f"[INFO] lin_vel_y     : {command_ranges.lin_vel_y}")
    print(f"[INFO] ang_vel_z     : {command_ranges.ang_vel_z}")


    render_mode = "rgb_array" if args.video else None

    env = gym.make(
        config.train.task,
        cfg=env_cfg,
        render_mode=render_mode,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # --------------------------------------------------------
    # 영상 저장 wrapper
    # --------------------------------------------------------

    if args.video:
        video_folder = log_dir / "videos" / f"{config.phase}_play"

        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }

        print("[INFO] Video recording configuration:")
        print_dict(video_kwargs, nesting=4)

        env = gym.wrappers.RecordVideo(
            env,
            **video_kwargs,
        )

    # # train.py와 동일하게 Phase2/3에서만 action mask 적용
    # if config.phase in ("phase2", "phase3"):
    #     env = PegLegActionMaskWrapper(env)

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    # --------------------------------------------------------
    # RSL-RL 모델 생성 및 checkpoint 로드
    # --------------------------------------------------------

    agent_cfg_dict = patch_rsl_rl_agent_cfg(
        agent_cfg.to_dict()
    )

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env=env,
            train_cfg=agent_cfg_dict,
            log_dir=None,
            device=device,
        )

    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env=env,
            train_cfg=agent_cfg_dict,
            log_dir=None,
            device=device,
        )

    else:
        raise ValueError(
            f"Unsupported runner class: {agent_cfg.class_name}"
        )

    print(f"[INFO] Loading model: {checkpoint_path}")

    runner.load(
        str(checkpoint_path),
        load_optimizer=False,
        map_location=device,
    )

    policy = runner.get_inference_policy(
        device=env.unwrapped.device
    )

    # --------------------------------------------------------
    # 재생 loop
    # --------------------------------------------------------

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)

                # Phase 3 Student는 recurrent policy이므로 물리 환경이 reset된
                # 에피소드의 LSTM hidden/cell state도 같은 시점에 초기화해야 한다.
                # 그렇지 않으면 새 부상 조건에서도 이전 에피소드의 기억이 남는다.
                if getattr(runner.alg.policy, "is_recurrent", False):
                    runner.alg.policy.reset(dones)


            timestep += 1

            if args.video and timestep >= args.video_length:
                break

            if args.real_time:
                sleep_time = dt - (time.time() - start_time)

                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
