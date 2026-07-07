#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint and compute/plot duty factor and contact forces per leg.

Duty factor = (time in contact) / (total time), computed from ContactSensor net forces.
Contact Force = Average magnitude of force when in contact (or overall average).

Outputs are saved next to the loaded checkpoint (same directory as play.py uses).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from peg_leg_action_wrapper import PegLegActionMaskWrapper  # isort: skip

# add argparse arguments (keep style close to play.py)
parser = argparse.ArgumentParser(description="Play an RL agent and plot duty factor per leg (RSL-RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O ops.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate (default: 10).")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Go1-Lab-v0",
    help="Task for loading agent/checkpoint (default: Template-Go1-Lab-v0).",
)
parser.add_argument(
    "--play_env",
    type=str,
    default=None,
    help="Environment to play in (e.g. Isaac-Velocity-Flat-Unitree-Go1-v0). If set, only the simulation env changes; result output (duty factor, contact force, groups, plots) is unchanged.",
)
parser.add_argument(
    "--flat",
    action="store_true",
    default=False,
    help="Use flat terrain for play (equivalent to --play_env Isaac-Velocity-Flat-Unitree-Go1-v0). Result format unchanged.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.") # Removed duplicate

# duty-factor specific args
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run for duty factor estimation.")
parser.add_argument(
    "--contact_sensor",
    type=str,
    default="contact_forces",
    help="Scene name for ContactSensor (default: contact_forces).",
)
parser.add_argument(
    "--contact_threshold",
    type=float,
    default=1.0,
    help="Contact threshold in N for detecting contact (default: 1.0).",
)
parser.add_argument(
    "--use_z_only",
    action="store_true",
    default=False,
    help="Use |Fz| only as GRF proxy instead of ||F||.",
)
parser.add_argument(
    "--no_show",
    action="store_true",
    default=False,
    help="Do not open plot window (always saves png).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear sys.argv for hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment (incl. Isaac Lab built-in tasks for --play_env)
import isaaclab_tasks  # noqa: F401
import go1_lab.tasks  # noqa: F401

# Import peg leg helper
from go1_lab.tasks.manager_based.go1_lab.mdp.events import _get_peg_leg_per_env

def calculate_duty_factor(contact_history, threshold=1.0):
    """
    contact_history: (steps, num_legs) tensor or array
    Returns: (num_legs,) array of duty factors (0.0 to 1.0)
    """
    if isinstance(contact_history, torch.Tensor):
        contact_history = contact_history.cpu().numpy()
    
    steps, num_legs = contact_history.shape
    if steps == 0:
        return np.zeros(num_legs)
        
    # Count contacts
    contact_counts = np.sum(contact_history > threshold, axis=0)
    return contact_counts / steps

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and compute duty factor."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # Optional: use flat/other terrain for play.
    # Keep task cfg (obs/action spaces, reward terms, result format) unchanged for checkpoint compatibility.
    if args_cli.flat and not args_cli.play_env:
        args_cli.play_env = "Isaac-Velocity-Flat-Unitree-Go1-v0"
    play_env_id = args_cli.task
    if args_cli.play_env:
        device = args_cli.device or getattr(env_cfg.sim, "device", "cuda:0")
        num_envs_override = args_cli.num_envs if args_cli.num_envs is not None else getattr(env_cfg.scene, "num_envs", 10)
        play_env_cfg = parse_env_cfg(
            args_cli.play_env,
            device=device,
            num_envs=num_envs_override,
        )
        # Replace only terrain-related config from the target play env.
        if hasattr(env_cfg, "scene") and hasattr(play_env_cfg, "scene") and hasattr(play_env_cfg.scene, "terrain"):
            env_cfg.scene.terrain = play_env_cfg.scene.terrain
        if (
            hasattr(env_cfg, "curriculum")
            and hasattr(play_env_cfg, "curriculum")
            and hasattr(play_env_cfg.curriculum, "terrain_levels")
        ):
            env_cfg.curriculum.terrain_levels = play_env_cfg.curriculum.terrain_levels
        print(f"[INFO] Applied terrain from env: {args_cli.play_env} (policy/task cfg kept from: {args_cli.task})")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # find checkpoint
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # create environment (task id stays the same; terrain may be overridden above)
    env = gym.make(play_env_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 고장 다리 calf action masking 적용
    env = PegLegActionMaskWrapper(env)

    # wrap for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # load policy
    from rsl_rl.runners import OnPolicyRunner, DistillationRunner
    
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner.load(resume_path)

    # play.py와 동일하게 policy를 JIT/ONNX로 export
    try:
        # rsl-rl >= 2.3
        policy_nn = runner.alg.policy
    except AttributeError:
        # rsl-rl <= 2.2
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # reset environment
    obs, _ = env.reset()
    
    # Get peg leg info
    base_env = env.unwrapped
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    peg_leg_per_env = _get_peg_leg_per_env(base_env, env_ids)
    
    # Print environment configuration
    print("\n" + "="*80)
    print("환경 구성 및 의족 상태")
    print("="*80)
    
    groups = {0: [], 1: [], 2: [], 3: [], 4: []}
    leg_names = ["FL", "FR", "RL", "RR"]
    
    for env_id, leg_idx in peg_leg_per_env.items():
        group = env_id % 5
        groups[group].append(env_id)
        
        status = "온전한 상태" if leg_idx is None else f"{leg_names[leg_idx]} 의족"
        print(f"Env {env_id:3d}: {status}")
        
    print("-" * 80)
    print(f"그룹 0 (온전한 상태): {len(groups[0])}개")
    for i in range(1, 5):
        print(f"그룹 {i} ({leg_names[i-1]} 의족): {len(groups[i])}개")
    print("="*80 + "\n")
    
    # Collect data
    print(f"데이터 수집 중... ({args_cli.steps} 스텝)")
    
    # Store contact forces: (steps, num_envs, num_legs)
    contact_forces_hist = []
    
    for i in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            # RslRlVecEnvWrapper returns (obs, rew, done, info) - 4 values
            ret = env.step(actions)
            if len(ret) == 5:
                obs, _, _, _, _ = ret
            else:
                obs, _, _, _ = ret
            
            # Get contact forces
            # Assuming sensor name is 'contact_forces'
            try:
                sensor = base_env.scene.sensors[args_cli.contact_sensor]
                
                # Debug: Print sensor body names once
                if i == 0:
                    print(f"\n[Sensor Debug] Sensor '{args_cli.contact_sensor}' body names:")
                    for idx, name in enumerate(sensor.body_names):
                        print(f"  {idx}: {name}")
                
                # Find foot indices dynamically if not done yet
                if 'foot_indices' not in locals():
                    foot_indices = []
                    target_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
                    for foot in target_feet:
                        found = False
                        for idx, body_name in enumerate(sensor.body_names):
                            if foot in body_name:  # Match substring (e.g. "base/FL_foot")
                                foot_indices.append(idx)
                                found = True
                                break
                        if not found:
                            print(f"Warning: Could not find index for {foot}")
                            foot_indices.append(0) # Fallback to 0 to avoid crash
                    
                    if i == 0:
                        print(f"[Sensor Debug] Mapped foot indices: {foot_indices}\n")

                forces = sensor.data.net_forces_w
                
                # Calculate magnitude
                if args_cli.use_z_only:
                    force_mag = torch.abs(forces[..., 2])
                else:
                    force_mag = torch.norm(forces, dim=-1)
                
                # Use mapped indices
                feet_forces = force_mag[:, foot_indices]
                
                # Debug: Print max force every 100 steps
                if i % 100 == 0:
                    max_force = torch.max(feet_forces).item()
                    mean_force = torch.mean(feet_forces).item()
                    print(f"Step {i}: Max Force={max_force:.2f}, Mean Force={mean_force:.2f}")
                
                contact_forces_hist.append(feet_forces.cpu().numpy())
                
            except Exception as e:
                if i == 0:
                    print(f"Warning: Could not read contact forces: {e}")
                
        if i % 100 == 0:
            print(f"Step {i}/{args_cli.steps}")
            
    # Process data
    contact_forces_hist = np.array(contact_forces_hist) # (steps, num_envs, 4)
    
    # Calculate duty factor per env, per leg
    # Thresholding
    in_contact = contact_forces_hist > args_cli.contact_threshold
    duty_factors = np.mean(in_contact, axis=0) # (num_envs, 4)
    
    # Calculate average force per env, per leg
    avg_forces = np.mean(contact_forces_hist, axis=0) # (num_envs, 4)

    # -------------------------------------------------------------------------
    # 1. Duty Factor Analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Duty Factor 분석 결과")
    print("="*80)
    print(f"{'Group':<20} | {'FL':<8} | {'FR':<8} | {'RL':<8} | {'RR':<8} | {'Avg':<8}")
    print("-" * 80)
    
    df_stats = {}
    
    # Normal Group (Group 0)
    if groups[0]:
        df_normal = duty_factors[groups[0]]
        avg_normal = np.mean(df_normal, axis=0)
        total_avg = np.mean(avg_normal)
        print(f"{'Normal (No Peg)':<20} | {avg_normal[0]:.4f}   | {avg_normal[1]:.4f}   | {avg_normal[2]:.4f}   | {avg_normal[3]:.4f}   | {total_avg:.4f}")
        df_stats['normal'] = avg_normal
    
    # Peg Leg Groups
    for i in range(1, 5):
        leg_name = leg_names[i-1]
        if groups[i]:
            df_peg = duty_factors[groups[i]]
            avg_peg = np.mean(df_peg, axis=0)
            total_avg = np.mean(avg_peg)
            
            row_str = f"{leg_name + ' Peg Leg':<20} | "
            for j in range(4):
                val = avg_peg[j]
                if j == i-1:
                    row_str += f"*{val:.4f}*  | "
                else:
                    row_str += f"{val:.4f}   | "
            row_str += f"{total_avg:.4f}"
            print(row_str)
            df_stats[f'peg_{leg_name}'] = avg_peg
            
    print("-" * 80)
    print("* 표시: 의족 다리")
    print("="*80 + "\n")

    # -------------------------------------------------------------------------
    # 2. Contact Force Analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("Contact Force (N) 분석 결과")
    print("="*80)
    print(f"{'Group':<20} | {'FL':<8} | {'FR':<8} | {'RL':<8} | {'RR':<8} | {'Avg':<8}")
    print("-" * 80)
    
    force_stats = {}
    
    # Normal Group (Group 0)
    if groups[0]:
        force_normal = avg_forces[groups[0]]
        avg_normal = np.mean(force_normal, axis=0)
        total_avg = np.mean(avg_normal)
        print(f"{'Normal (No Peg)':<20} | {avg_normal[0]:.2f}     | {avg_normal[1]:.2f}     | {avg_normal[2]:.2f}     | {avg_normal[3]:.2f}     | {total_avg:.2f}")
        force_stats['normal'] = avg_normal
    
    # Peg Leg Groups
    for i in range(1, 5):
        leg_name = leg_names[i-1]
        if groups[i]:
            force_peg = avg_forces[groups[i]]
            avg_peg = np.mean(force_peg, axis=0)
            total_avg = np.mean(avg_peg)
            
            row_str = f"{leg_name + ' Peg Leg':<20} | "
            for j in range(4):
                val = avg_peg[j]
                if j == i-1:
                    row_str += f"*{val:.2f}*    | "
                else:
                    row_str += f"{val:.2f}     | "
            row_str += f"{total_avg:.2f}"
            print(row_str)
            force_stats[f'peg_{leg_name}'] = avg_peg
            
    print("-" * 80)
    print("* 표시: 의족 다리 (낮을수록 좋음/절뚝거림 성공)")
    print("="*80 + "\n")
    
    # Visualization
    if not args_cli.no_show and len(groups[0]) > 0:
        try:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            x = np.arange(4)
            width = 0.15
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Default matplotlib colors
            
            # --- Plot 1: Duty Factor ---
            # Plot Normal
            ax1.bar(x - 2*width, df_stats['normal'], width, label='Normal', alpha=0.9, color=colors[0])
            
            # Plot Peg Legs
            for i in range(1, 5):
                key = f'peg_{leg_names[i-1]}'
                if key in df_stats:
                    ax1.bar(x + (i-2)*width, df_stats[key], width, label=f'{leg_names[i-1]} Peg', alpha=0.8, color=colors[i])
            
            ax1.set_ylabel('Duty Factor')
            ax1.set_title('Duty Factor (Contact Time Ratio)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(leg_names)
            ax1.legend()
            ax1.set_ylim(0, 1.0)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # --- Plot 2: Contact Force ---
            # Plot Normal
            ax2.bar(x - 2*width, force_stats['normal'], width, label='Normal', alpha=0.9, color=colors[0])
            
            # Plot Peg Legs
            for i in range(1, 5):
                key = f'peg_{leg_names[i-1]}'
                if key in force_stats:
                    ax2.bar(x + (i-2)*width, force_stats[key], width, label=f'{leg_names[i-1]} Peg', alpha=0.8, color=colors[i])
            
            ax2.set_ylabel('Average Force (N)')
            ax2.set_title('Contact Force Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(leg_names)
            ax2.legend()
            ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(os.path.dirname(resume_path), "gait_analysis.png")
            plt.savefig(save_path)
            print(f"Analysis plot saved to: {save_path}")
            
            if not args_cli.no_show:
                plt.show()
                
        except Exception as e:
            print(f"Plotting failed: {e}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
