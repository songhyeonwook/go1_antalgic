#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""체크포인트 재생 (옵션: 명령 고정, 랜덤화 제거, 5조건 비교, 영상 저장). phase 1/2/3 공통.

    # 전진 0.5 m/s 고정, 랜덤화 없이
    python test.py --phase_config_path configs/phase/1/phase1.yaml \
        --checkpoint <model.pt> --fixed_x 0.5 --clean

    # phase 2/3: Normal/FL/FR/RL/RR 를 env 0~4 에 고정 배치해 한 화면에서 비교
    python test.py --phase_config_path configs/phase/2/phase2_at.yaml \
        --checkpoint <model.pt> --compare_all --real_time

    # 영상 저장 (headless 가능, --enable_cameras 는 자동)
    python test.py --phase_config_path configs/phase/3/phase3.yaml \
        --checkpoint <model.pt> --peg_leg fl --video --video_length 1000 --headless

지표가 필요하면 play_result.py 를 쓴다.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

from utils.eval_common import (
    add_common_args, build_agent_cfg, build_env_cfg, check_splint_length_arg, load_config, log,
    make_gym_env, make_policy, print_conditions, print_header, resolve_checkpoint,
    resolve_eval_mode, set_viewer_follow, step_env, wrap_rsl,
)

parser = argparse.ArgumentParser(description="체크포인트 재생 + 평가 옵션 (RSL-RL)")
add_common_args(parser)
parser.add_argument("--steps", type=int, default=0, help="재생할 스텝 수. 0 이면 창을 닫을 때까지")
parser.add_argument("--real_time", action="store_true", help="실시간 속도로 늦춘다")
parser.add_argument("--follow_env", type=int, default=0, help="카메라가 따라갈 env 번호")
parser.add_argument("--fixed_x", type=float, default=None, help="전진 명령 고정 [m/s] (좌우 명령은 0)")
parser.add_argument("--fixed_yaw", type=float, default=None, help="yaw 명령 고정 [rad/s]")
parser.add_argument(
    "--compare_all", action="store_true",
    help="5 env 를 Normal/FL/FR/RL/RR 로 고정 배치 (peg_leg.enabled=true 인 phase 2/3 전용)",
)
parser.add_argument("--video", action="store_true", help="재생 영상을 mp4 로 저장 (체크포인트 폴더/videos)")
parser.add_argument("--video_length", type=int, default=1000, help="영상 길이 [step]. 50 Hz 라 1000 = 20 s")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
check_splint_length_arg(args.splint_length)

if args.compare_all:
    if args.peg_leg not in (None, "balanced"):
        parser.error("--compare_all 은 --peg_leg balanced 와만 함께 쓸 수 있습니다.")
    args.peg_leg = "balanced"
    args.num_envs = 5
if args.video:
    args.enable_cameras = True  # AppLauncher 가 읽는다 — 부팅 전에 켜야 한다

config = load_config(args)
checkpoint = resolve_checkpoint(args)
eval_mode = resolve_eval_mode(args, config)
num_envs = args.num_envs if args.num_envs is not None else 1
seed = args.seed if args.seed is not None else config.train.seed
device = args.device

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    env_cfg = build_env_cfg(
        config, num_envs=num_envs, seed=seed, device=device, eval_mode=eval_mode,
        splint_length=args.splint_length, clean=args.clean,
    )
    set_viewer_follow(env_cfg, env_index=args.follow_env)
    if args.compare_all:
        env_cfg.scene.env_spacing = 2.5  # 다섯 마리가 한 화면에 들어오도록

    # 명령 고정: 샘플 범위를 한 점으로 좁힌다 (리셋마다 같은 값이 뽑힌다)
    ranges = env_cfg.commands.base_velocity.ranges
    if args.fixed_x is not None:
        ranges.lin_vel_x = (args.fixed_x, args.fixed_x)
        ranges.lin_vel_y = (0.0, 0.0)
    if args.fixed_yaw is not None:
        ranges.ang_vel_z = (args.fixed_yaw, args.fixed_yaw)

    agent_cfg = build_agent_cfg(config, seed=seed, device=device)
    print_header(config, checkpoint, device, seed, num_envs, eval_mode, env_cfg)
    if args.clean:
        log("[eval] Clean          : 마찰·질량 랜덤화, push, 관측 노이즈 끔")
    if args.compare_all:
        log("[eval] Fixed mapping  : env0=Normal, env1=FL, env2=FR, env3=RL, env4=RR")

    env = make_gym_env(config, env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        video_folder = checkpoint.parent / "videos" / f"{config.phase}_{eval_mode}"
        log(f"[eval] Video          : {video_folder} ({args.video_length} step)")
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(video_folder), step_trigger=lambda step: step == 0,
            video_length=args.video_length, disable_logger=True,
        )
    env = wrap_rsl(env, agent_cfg)
    base = env.unwrapped
    print_conditions(base)
    _, policy_fn, policy_module = make_policy(env, agent_cfg, config, checkpoint, device)

    dt = float(base.step_dt)
    obs = env.get_observations()
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            if args.steps and step >= args.steps:
                break
            if args.video and step >= args.video_length:
                break
            start = time.time()
            obs, _, _, _, _ = step_env(env, policy_fn, policy_module, obs)
            step += 1
            if args.real_time:
                remaining = dt - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
    log(f"[test] {step} step 재생 완료")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()  # Isaac Sim 종료는 파이썬 stdout 버퍼를 비우지 않는다
        simulation_app.close()
