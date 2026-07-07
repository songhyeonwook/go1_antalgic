#!/usr/bin/env python3
"""Export a selected RSL-RL Go1 policy checkpoint to JIT and ONNX, then exit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from peg_leg_action_wrapper import PegLegActionMaskWrapper  # isort: skip


parser = argparse.ArgumentParser(description="Export an RSL-RL Go1 checkpoint to JIT/ONNX.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. If omitted, use selected Phase 3.")
parser.add_argument("--output_dir", type=str, default=None, help="Export directory. Default: <checkpoint_dir>/exported.")
parser.add_argument("--task", type=str, default="Template-Go1-Lab-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distill_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--phase", type=str, default="student", choices=("healthy", "teacher", "student"))
parser.add_argument("--jit_name", type=str, default="policy.pt")
parser.add_argument("--onnx_name", type=str, default="policy.onnx")
parser.add_argument("--descriptor_name", type=str, default="policy_io.json")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
os.environ["GO1_PHASE"] = args_cli.phase
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_lab.tasks  # noqa: F401


def _selected_phase3_checkpoint() -> str:
    path = Path(__file__).resolve().parent / "logs" / "rsl_rl" / "unitree_go1_rough_student"
    selected = path / "PAPER_GRADE_PHASE3_CHECKPOINT.txt"
    if selected.is_file():
        value = selected.read_text(encoding="utf-8").strip()
        if value and value != "NO_PAPER_GRADE_CANDIDATE":
            return value
    raise FileNotFoundError(
        "No selected Phase 3 checkpoint. Pass --checkpoint or run select_phase3_student_candidate.py first."
    )


def _normalizer(policy_nn):
    if hasattr(policy_nn, "actor_obs_normalizer"):
        return policy_nn.actor_obs_normalizer
    if hasattr(policy_nn, "student_obs_normalizer"):
        return policy_nn.student_obs_normalizer
    return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    checkpoint = os.path.abspath(args_cli.checkpoint or _selected_phase3_checkpoint())
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = os.path.dirname(checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = PegLegActionMaskWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.load(checkpoint)
    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    normalizer = _normalizer(policy_nn)

    export_dir = Path(args_cli.output_dir or (Path(checkpoint).parent / "exported"))
    export_dir.mkdir(parents=True, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(export_dir), filename=args_cli.jit_name)
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(export_dir), filename=args_cli.onnx_name)

    obs = env.get_observations()
    obs_policy = obs["policy"] if isinstance(obs, dict) else obs
    policy_is_recurrent = bool(getattr(policy_nn, "is_recurrent", False))
    descriptor = {
        "checkpoint": checkpoint,
        "phase": args_cli.phase,
        "agent": args_cli.agent,
        "task": args_cli.task,
        "jit": str(export_dir / args_cli.jit_name),
        "onnx": str(export_dir / args_cli.onnx_name),
        "is_recurrent": policy_is_recurrent,
        "observation_shape": list(obs_policy.shape),
        "action_shape": list(env.action_space.shape),
    }
    if policy_is_recurrent:
        rnn = policy_nn.memory_s.rnn if hasattr(policy_nn, "memory_s") else policy_nn.memory_a.rnn
        descriptor.update(
            {
                "rnn_type": type(rnn).__name__.lower(),
                "rnn_num_layers": int(rnn.num_layers),
                "rnn_hidden_size": int(rnn.hidden_size),
                "onnx_inputs": ["obs", "h_in", "c_in"] if type(rnn).__name__.lower() == "lstm" else ["obs", "h_in"],
                "onnx_outputs": ["actions", "h_out", "c_out"]
                if type(rnn).__name__.lower() == "lstm"
                else ["actions", "h_out"],
            }
        )
    else:
        descriptor.update({"onnx_inputs": ["obs"], "onnx_outputs": ["actions"]})

    descriptor_path = export_dir / args_cli.descriptor_name
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    print(f"[SAVED] {export_dir / args_cli.jit_name}")
    print(f"[SAVED] {export_dir / args_cli.onnx_name}")
    print(f"[SAVED] {descriptor_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
