#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from utils.config_builder import load_experiment_config

SCRIPT_DIR = Path(__file__).resolve().parent
LEG_NAMES = ["FL", "FR", "RL", "RR"]
GROUP_LABELS = ["Normal", "FL Peg", "FR Peg", "RL Peg", "RR Peg"]
# rls_estimate / aux head 정규화 규약 (mdp/rls.py, DistillationAuxCfg 와 동일)
L_PRIOR, L_SCALE = 0.39, 0.06
MU_PRIOR, MU_SCALE = 1.0, 0.5

parser = argparse.ArgumentParser(description="보행 지표 + 부목 추정 결과 추출 (RSL-RL)")
parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=3)
parser.add_argument("--checkpoint", type=str, required=True, help="재생할 model_*.pt 경로")
parser.add_argument("--num_envs", type=int, default=40, help="5의 배수 권장 (조건 균등 배정)")
parser.add_argument("--steps", type=int, default=1500, help="수집 스텝 (50 Hz — 1500 = 30 s)")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--contact_threshold", type=float, default=5.0, help="접지 판정 힘 [N]")
parser.add_argument("--use_z_only", action="store_true", help="||F|| 대신 |Fz| 사용")
parser.add_argument("--no_show", action="store_true", help="plot 창을 띄우지 않음 (png 는 항상 저장)")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()

config = load_experiment_config(
    phase_path=SCRIPT_DIR / "configs" / "phase" / f"phase{args.phase}.yaml",
    common_path=str(SCRIPT_DIR / "configs" / "common.yaml"),
)

sys.argv = [sys.argv[0], *hydra_args,
            "hydra/job_logging=disabled", "hydra.output_subdir=null", "hydra.run.dir=."]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac Sim 시작 이후 import ──────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # 저장 우선 — --no_show 아니면 마지막에 show
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import go1_lab.tasks  # noqa: F401, E402


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
    out_dir = checkpoint_path.parent

    seed = args.seed if args.seed is not None else config.train.seed
    device = config.common.get("device", "cuda:0")
    agent_cfg.seed = seed
    agent_cfg.device = device
    if config.phase in {"phase1", "phase2", "phase3"}:
        agent_cfg.policy.noise_std_type = config.exploration.noise_std_type
        agent_cfg.policy.init_noise_std = config.exploration.init_noise_std

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = seed
    env_cfg.apply_environment_settings(
        config.environment.values, int(agent_cfg.num_steps_per_env)
    )

    # balanced 고정 배정: env_id % 5 → Normal/FL/FR/RL/RR
    peg_event = env_cfg.events.randomize_peg_leg_actuation
    if peg_event is None:
        raise RuntimeError("peg_leg.enabled=true 환경에서만 사용할 수 있습니다.")
    peg_event.params["target_leg"] = "balanced_env"
    peg_event.params["healthy_slots"] = 1
    if getattr(env_cfg.curriculum, "peg_leg_difficulty", None) is not None:
        env_cfg.curriculum.peg_leg_difficulty = None

    env = gym.make(config.train.task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_cfg_dict = patch_rsl_rl_agent_cfg(agent_cfg.to_dict())
    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env=env, train_cfg=agent_cfg_dict, log_dir=None, device=device)
    runner.load(str(checkpoint_path), load_optimizer=False, map_location=device)
    policy = runner.get_inference_policy(device=base.device)
    policy_nn = runner.alg.policy
    has_aux = hasattr(policy_nn, "aux_predict")

    contacts = base.scene["contact_forces"]
    body_names = list(contacts.body_names)
    foot_b = [body_names.index(f"{leg}_foot") for leg in LEG_NAMES]
    splint_b = [body_names.index(f"{leg}_splint") for leg in LEG_NAMES]

    N, T = args.num_envs, args.steps
    print(f"[INFO] Checkpoint : {checkpoint_path}")
    print(f"[INFO] 조건 배정   : balanced (env%5), N={N}, steps={T}")
    print(f"[INFO] aux head   : {'있음 — L̂/μ̂ 추정 로깅' if has_aux else '없음 — 추정 생략'}")

    # ── 수집 ──
    force_hist = np.zeros((T, N, 4), np.float32)       # 다리별 유효 접촉력
    gt_leg_h = np.zeros((T, N), np.int8)
    gt_L_h = np.zeros((T, N), np.float32)
    gt_mu_h = np.zeros((T, N), np.float32)
    t_reset = np.zeros((T, N), np.float32)             # 리셋 후 경과 [s]
    aux_h = np.zeros((T, N, 2), np.float32)
    rls_h = np.zeros((T, N), np.float32)
    dt = base.step_dt
    since_reset = torch.zeros(N, device=base.device)

    obs = env.get_observations()
    with torch.inference_mode():
        for t in range(T):
            actions = policy(obs)
            if has_aux:
                aux_h[t] = policy_nn.aux_predict().cpu().numpy()
            rls_h[t] = obs["policy"][:, 49].cpu().numpy()
            obs, _, dones, _ = env.step(actions)
            if getattr(policy_nn, "is_recurrent", False):
                policy_nn.reset(dones)

            f = contacts.data.net_forces_w
            mag = f[..., 2].abs() if args.use_z_only else f.norm(dim=-1)
            feet = mag[:, foot_b].clone()
            leg_idx = base._peg_leg_index                     # (N,) -1=정상
            inj = leg_idx >= 0
            if bool(inj.any()):
                rows = torch.nonzero(inj).squeeze(-1)
                feet[rows, leg_idx[rows]] = mag[:, splint_b][rows, leg_idx[rows]]
            force_hist[t] = feet.cpu().numpy()
            gt_leg_h[t] = leg_idx.cpu().numpy()
            gt_L_h[t] = base._peg_leg_splint_length.cpu().numpy()
            gt_mu_h[t] = base._peg_leg_foot_friction.cpu().numpy()
            t_reset[t] = since_reset.cpu().numpy()
            since_reset += dt
            since_reset[dones.view(-1) > 0] = 0.0
            if t % 300 == 0:
                print(f"  step {t}/{T}")

    # ── 그룹 배정 (balanced: env_id % 5) ──
    groups = {g: [n for n in range(N) if n % 5 == g] for g in range(5)}
    inj_mask = gt_leg_h >= 0

    in_contact = force_hist > args.contact_threshold
    duty = in_contact.mean(axis=0)                     # (N, 4)
    avg_force = np.array([
        [force_hist[in_contact[:, n, k], n, k].mean() if in_contact[:, n, k].any() else 0.0
         for k in range(4)] for n in range(N)
    ])

    def print_table(title: str, data: np.ndarray, fmt: str):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"{'Group':<16} | {'FL':<9} | {'FR':<9} | {'RL':<9} | {'RR':<9} | {'Avg':<8}")
        print("-" * 80)
        stats = {}
        for g in range(5):
            if not groups[g]:
                continue
            avg = data[groups[g]].mean(axis=0)
            cells = []
            for k in range(4):
                v = format(avg[k], fmt)
                cells.append(f"*{v}*".ljust(9) if g - 1 == k else f"{v}".ljust(9))
            print(f"{GROUP_LABELS[g]:<16} | " + " | ".join(cells) + f" | {avg.mean():{fmt}}")
            stats[g] = avg
        print("-" * 80)
        print("* 표시: 부상 다리 (부목 접촉 기준)")
        return stats

    df_stats = print_table("Duty Factor 분석 결과", duty, ".4f")
    force_stats = print_table("Contact Force (N) 분석 결과 (접지 중 평균)", avg_force, ".2f")

    # ── plot 1: gait_analysis.png (기존 형식 유지) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(4)
    width = 0.15
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for ax, stats, ylabel, title in (
        (ax1, df_stats, "Duty Factor", "Duty Factor (Contact Time Ratio)"),
        (ax2, force_stats, "Average Force (N)", "Contact Force (injured leg = splint)"),
    ):
        for g in range(5):
            if g in stats:
                ax.bar(x + (g - 2) * width, stats[g], width,
                       label=GROUP_LABELS[g], alpha=0.85, color=colors[g])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(LEG_NAMES)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax1.set_ylim(0, 1.0)
    plt.tight_layout()
    gait_png = out_dir / "gait_analysis.png"
    plt.savefig(gait_png, dpi=150)
    print(f"\n[INFO] 저장: {gait_png}")

    # ── 부목 추정 결과 ──
    if has_aux:
        ti, ni = np.where(inj_mask)
        L_hat = aux_h[..., 0] * L_SCALE + L_PRIOR
        mu_hat = aux_h[..., 1] * MU_SCALE + MU_PRIOR
        rls_L = rls_h * L_SCALE + L_PRIOR

        # μ̂ 누적평균: LSTM 순간 출력은 ~3 s 에서 포화하지만 (지수창 추정기),
        # 요동이 참값 주위에서 대체로 무편향이라 에피소드 내 누적 평균이
        # 더 정확하다 (실측: 10-20 s 구간 0.093 → 0.070). 초기 1 s 는
        # 수렴 전 값이라 평균에서 제외한다.
        WARM = max(1, int(round(1.0 / dt)))
        mu_avg = np.full((T, N), np.nan, np.float32)
        for n in range(N):
            starts = [t for t in range(T) if t_reset[t, n] == 0.0]
            for i, s in enumerate(starts):
                e = starts[i + 1] if i + 1 < len(starts) else T
                if e - s <= WARM:
                    continue
                seg = mu_hat[s:e, n]
                csum = np.cumsum(seg[WARM:])
                mu_avg[s + WARM:e, n] = csum / np.arange(1, e - s - WARM + 1)

        errL = np.abs(L_hat - gt_L_h)[ti, ni]
        errR = np.abs(rls_L - gt_L_h)[ti, ni]
        errM = np.abs(mu_hat - gt_mu_h)[ti, ni]
        errMA = np.abs(mu_avg - gt_mu_h)[ti, ni]        # NaN = 워밍업 구간
        tr = t_reset[ti, ni]
        conv = tr > 5.0
        ma_ok = ~np.isnan(errMA)

        print("\n" + "=" * 80)
        print("부목 길이 · 마찰 추정 결과 (부상 env)")
        print("=" * 80)
        print(f"  aux L̂     : MAE median {np.median(errL)*1000:.1f} mm "
              f"(90%: {np.quantile(errL, 0.9)*1000:.1f}), 수렴 후(>5s) "
              f"{np.median(errL[conv])*1000:.1f} mm")
        print(f"  RLS 채널 L̂: MAE median {np.median(errR)*1000:.1f} mm "
              f"(90%: {np.quantile(errR, 0.9)*1000:.1f})")
        print(f"  aux μ̂     : MAE median {np.median(errM):.3f} "
              f"(90%: {np.quantile(errM, 0.9):.3f}), 수렴 후(>5s) "
              f"{np.median(errM[conv]):.3f}   [상수 예측 "
              f"{np.abs(gt_mu_h[ti, ni] - np.median(gt_mu_h[ti, ni])).mean():.3f}]")
        print(f"  aux μ̂ 누적평균: MAE median {np.median(errMA[ma_ok]):.3f} "
              f"(90%: {np.quantile(errMA[ma_ok], 0.9):.3f}), 수렴 후(>5s) "
              f"{np.median(errMA[ma_ok & conv]):.3f}  ← 배포 권장 판독값")

        # plot 2: estimation_analysis.png
        fig2, axes = plt.subplots(2, 2, figsize=(14, 9))
        bins = np.arange(0.0, min(20.0, T * dt), 0.5)

        def conv_curve(ax, err, label, color, ls="-"):
            bx, med = [], []
            for b in bins:
                m = (tr >= b) & (tr < b + 0.5) & ~np.isnan(err)
                if m.sum() >= 20:
                    bx.append(b + 0.25)
                    med.append(np.median(err[m]))
            ax.plot(bx, med, label=label, color=color, lw=2, ls=ls)

        ax = axes[0, 0]
        conv_curve(ax, errL * 1000, "aux head L̂", "#1f77b4")
        conv_curve(ax, errR * 1000, "RLS channel L̂", "#ff7f0e")
        ax.set_xlabel("time since reset [s]")
        ax.set_ylabel("|L̂ − L| median [mm]")
        ax.set_title("Splint length estimation convergence")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        sc = conv & (np.arange(len(ti)) % 5 == 0)      # 산점도 서브샘플
        ax.scatter(gt_L_h[ti, ni][sc] * 1000, L_hat[ti, ni][sc] * 1000,
                   s=4, alpha=0.25, color="#1f77b4")
        ax.plot([320, 460], [320, 460], "k--", lw=1)
        ax.set_xlabel("GT L [mm]")
        ax.set_ylabel("aux L̂ [mm]")
        ax.set_title("L̂ vs GT (converged, >5 s)")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        conv_curve(ax, errM, "aux head μ̂ (instant)", "#2ca02c")
        conv_curve(ax, errMA, "aux μ̂ running mean", "#d62728")
        base_mae = np.abs(gt_mu_h[ti, ni] - np.median(gt_mu_h[ti, ni])).mean()
        ax.axhline(base_mae, color="gray", ls="--", lw=1, label="constant predictor")
        ax.set_xlabel("time since reset [s]")
        ax.set_ylabel("|μ̂ − μ| median")
        ax.set_title("Splint friction estimation convergence")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        sm = sc & ~np.isnan(mu_avg[ti, ni])
        ax.scatter(gt_mu_h[ti, ni][sm], mu_avg[ti, ni][sm],
                   s=4, alpha=0.25, color="#d62728")
        ax.plot([0.5, 1.5], [0.5, 1.5], "k--", lw=1)
        ax.set_xlabel("GT μ")
        ax.set_ylabel("aux μ̂ (running mean)")
        ax.set_title("averaged μ̂ vs GT (converged, >5 s)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        est_png = out_dir / "estimation_analysis.png"
        plt.savefig(est_png, dpi=150)
        print(f"[INFO] 저장: {est_png}")

    env.close()


if __name__ == "__main__":
    # simulation_app.close() 는 종료 시 세그폴트가 날 수 있어 (rollout_dump 와 동일)
    # traceback 출력 후 os._exit 로 끝낸다.
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
