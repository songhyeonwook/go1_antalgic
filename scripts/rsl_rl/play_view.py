#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""학습된 체크포인트를 뷰어로 재생한다 (phase 1/2/3 공통). 지표는 만들지 않는다.

    # phase 1 (GUI — --headless 를 주지 않으면 뷰어가 뜬다)
    python play_view.py --phase_config_path configs/phase/1/phase1.yaml \
        --checkpoint logs/unitree_go1_antalgic/<run>/model_4999.pt --num_envs 16

    # phase 2/3: 조건 지정 (normal / fl / fr / rl / rr / balanced)
    python play_view.py --phase_config_path configs/phase/2/phase2_at.yaml \
        --checkpoint <model.pt> --peg_leg fl --num_envs 4 --real_time

지표·플롯이 필요하면 play_result.py, 영상 저장·명령 고정이 필요하면 test.py 를 쓴다.
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

parser = argparse.ArgumentParser(description="체크포인트 뷰어 재생 (RSL-RL)")
add_common_args(parser)
parser.add_argument("--steps", type=int, default=0, help="재생할 스텝 수. 0 이면 창을 닫을 때까지")
parser.add_argument("--real_time", action="store_true", help="실시간 속도로 늦춘다 (눈으로 보기 편함)")
parser.add_argument("--follow_env", type=int, default=0, help="카메라가 따라갈 env 번호")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
check_splint_length_arg(args.splint_length)

config = load_config(args)
checkpoint = resolve_checkpoint(args)
eval_mode = resolve_eval_mode(args, config)
num_envs = args.num_envs if args.num_envs is not None else 16
seed = args.seed if args.seed is not None else config.train.seed
device = args.device

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import time  # noqa: E402

import torch  # noqa: E402


def main() -> None:
    env_cfg = build_env_cfg(
        config, num_envs=num_envs, seed=seed, device=device, eval_mode=eval_mode,
        splint_length=args.splint_length, clean=args.clean,
    )
    set_viewer_follow(env_cfg, env_index=args.follow_env)
    agent_cfg = build_agent_cfg(config, seed=seed, device=device)
    print_header(config, checkpoint, device, seed, num_envs, eval_mode, env_cfg)

    env = wrap_rsl(make_gym_env(config, env_cfg), agent_cfg)
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
            start = time.time()
            obs, _, _, _, _ = step_env(env, policy_fn, policy_module, obs)
            step += 1
            if args.real_time:
                remaining = dt - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
    log(f"[play_view] {step} step 재생 완료")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()  # Isaac Sim 종료는 파이썬 stdout 버퍼를 비우지 않는다
        simulation_app.close()
