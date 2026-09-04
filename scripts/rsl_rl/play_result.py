#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보행 지표 + (phase 3) 부목 길이·속도 추정 결과 추출. phase 1/2/3 공통.

    python play_result.py --phase_config_path configs/phase/2/phase2_at.yaml \
        --checkpoint <model.pt> --num_envs 40 --steps 1500 --headless

조건 배정은 env_fixed 블록이다 (balanced + 40 env: env 0-7 Normal, 8-15 FL, 16-23 FR,
24-31 RL, 32-39 RR). 지표는 각 스텝의 실제 _peg_leg_index 로 그룹을 나누므로 배정
방식과 무관하게 맞는다. 부상 다리의 '접지'는 발이 아니라 부목 끝단({leg}_splint) 접촉력이다.

출력 (기본: 체크포인트 폴더):
  gait_analysis.png         조건별 다리별 duty factor / 접촉력
  estimation_analysis.png   phase 3 만: L̂ 수렴 곡선, L̂ vs GT, v̂ 오차
  metrics_summary.csv       조건별 수치 요약
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from utils.eval_common import (
    CONDITION_LABELS, LEGS, add_common_args, build_agent_cfg, build_env_cfg,
    check_splint_length_arg, load_config, log, make_gym_env, make_policy, peg_leg_enabled,
    print_conditions, print_header, resolve_checkpoint, resolve_eval_mode, wrap_rsl,
)

parser = argparse.ArgumentParser(description="보행 지표 + 부목 추정 결과 추출 (RSL-RL)")
add_common_args(parser)
parser.add_argument("--steps", type=int, default=1500, help="수집 스텝 (50 Hz — 1500 = 30 s)")
parser.add_argument("--contact_threshold", type=float, default=5.0, help="접지 판정 힘 [N]")
parser.add_argument("--use_z_only", action="store_true", help="||F|| 대신 |Fz| 사용")
parser.add_argument("--out_dir", type=str, default=None, help="결과 저장 폴더 (기본: 체크포인트 폴더)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
check_splint_length_arg(args.splint_length)

config = load_config(args)
checkpoint = resolve_checkpoint(args)
# 부상 env 가 있는 phase 는 다섯 조건을 고르게, healthy 는 normal 만
eval_mode = resolve_eval_mode(args, config, default="balanced" if peg_leg_enabled(config) else "normal")
num_envs = args.num_envs if args.num_envs is not None else 40
seed = args.seed if args.seed is not None else config.train.seed
device = args.device
out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else checkpoint.parent

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def collect(env, base, policy_fn, policy_module, T: int):
    """T 스텝 동안 지표 원자료를 모은다. 모든 배열의 첫 축은 스텝, 둘째 축은 env."""
    N = base.num_envs
    contacts = base.scene["contact_forces"]
    names = list(contacts.body_names)
    foot_b = torch.tensor([names.index(f"{leg}_foot") for leg in LEGS], device=base.device)
    splint_b = torch.tensor([names.index(f"{leg}_splint") for leg in LEGS], device=base.device)
    robot = base.scene["robot"]
    has_aux = hasattr(policy_module, "aux_inference")   # phase 3 student 만
    recurrent = getattr(policy_module, "is_recurrent", False)
    dt = float(base.step_dt)

    rec = {
        "force": np.zeros((T, N, 4), np.float32),   # 다리별 유효 접촉력 (부상 다리 = 부목 끝단)
        "leg": np.zeros((T, N), np.int8),            # -1 정상, 0..3 부상 다리
        "L": np.zeros((T, N), np.float32),           # GT 부목 길이 (정상 = 0)
        "t_reset": np.zeros((T, N), np.float32),     # 정책 호출 시점의 리셋 후 경과 [s]
        "verr": np.zeros((T, N), np.float32),        # |v_cmd,xy - v_xy|
        "werr": np.zeros((T, N), np.float32),        # |w_cmd,z - w_z|
        "fall": np.zeros((T, N), bool),              # time-out 이 아닌 종료
        "L_hat": np.zeros((T, N), np.float32),       # phase 3: 부목 길이 추정 [m]
        "v_err": np.zeros((T, N), np.float32),       # phase 3: |v̂ - v_gt|
    }
    since_reset = torch.zeros(N, device=base.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for t in range(T):
            leg_pre = base._peg_leg_index.clone()
            rec["leg"][t] = leg_pre.cpu().numpy()
            rec["L"][t] = base._peg_leg_splint_length.cpu().numpy()
            rec["t_reset"][t] = since_reset.cpu().numpy()

            actions = policy_fn(obs)
            if has_aux:
                # act_inference 가 남긴 LSTM latent 위에서 보조 헤드를 평가한다.
                L_hat, v_hat = policy_module.aux_inference()
                priv = obs["privileged_obs"]           # [L, vx, vy, vz]
                rec["L_hat"][t] = L_hat.cpu().numpy()
                rec["v_err"][t] = (v_hat - priv[:, 1:4]).norm(dim=-1).cpu().numpy()

            obs, _, dones, extras = env.step(actions)
            if recurrent:
                policy_module.reset(dones)
            done_mask = dones.view(-1).bool()
            time_outs = extras.get("time_outs", torch.zeros_like(done_mask)).view(-1).bool()

            f = contacts.data.net_forces_w
            mag = f[..., 2].abs() if args.use_z_only else f.norm(dim=-1)
            feet = mag[:, foot_b].clone()
            inj = leg_pre >= 0
            if bool(inj.any()):
                rows = torch.nonzero(inj).squeeze(-1)
                feet[rows, leg_pre[rows]] = mag[:, splint_b][rows, leg_pre[rows]]
            rec["force"][t] = feet.cpu().numpy()

            cmd = base.command_manager.get_command("base_velocity")
            rec["verr"][t] = (cmd[:, :2] - robot.data.root_lin_vel_b[:, :2]).norm(dim=-1).cpu().numpy()
            rec["werr"][t] = (cmd[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs().cpu().numpy()
            rec["fall"][t] = (done_mask & ~time_outs).cpu().numpy()

            since_reset += dt
            since_reset[done_mask] = 0.0
            if t % 300 == 0:
                log(f"  step {t}/{T}", flush=True)
    rec["dt"] = dt
    rec["has_aux"] = has_aux
    return rec


def summarize(rec, threshold: float):
    """조건(그룹)별 지표. 그룹 = 그 스텝의 _peg_leg_index + 1 (0 Normal … 4 RR)."""
    cond = rec["leg"].astype(int) + 1
    contact = rec["force"] > threshold
    dt = rec["dt"]
    rows = []
    for g in range(5):
        m = cond == g                           # (T, N)
        n = int(m.sum())
        if n == 0:
            continue
        duty = [float(contact[..., k][m].mean()) for k in range(4)]
        force = []
        for k in range(4):
            c = contact[..., k] & m
            force.append(float(rec["force"][..., k][c].mean()) if c.any() else 0.0)
        row = {
            "group": CONDITION_LABELS[g], "samples": n,
            **{f"duty_{leg}": duty[k] for k, leg in enumerate(LEGS)},
            **{f"force_{leg}": force[k] for k, leg in enumerate(LEGS)},
            "vel_err_xy": float(rec["verr"][m].mean()),
            "yaw_err": float(rec["werr"][m].mean()),
            "falls_per_min": float(rec["fall"][m].sum() / (n * dt / 60.0)),
        }
        if rec["has_aux"]:
            if g > 0:
                # 전 구간 / 수렴 후(리셋 5 s 이후) 두 가지. 짧은 에피소드만 있으면 수렴 후 값은 비어 있다.
                err_all = np.abs(rec["L_hat"] - rec["L"])[m]
                row["L_mae_all_mm"] = float(np.median(err_all) * 1000)
                conv = m & (rec["t_reset"] > 5.0)
                if conv.any():
                    err = np.abs(rec["L_hat"] - rec["L"])[conv]
                    row["L_mae_mm"] = float(np.median(err) * 1000)
                    row["L_mae90_mm"] = float(np.quantile(err, 0.9) * 1000)
            row["v_mae"] = float(rec["v_err"][m].mean())
        rows.append(row)
    return rows


def print_tables(rows) -> None:
    def table(title, key, fmt):
        log("\n" + "=" * 84)
        log(title)
        log("=" * 84)
        log(f"{'Group':<8} | {'FL':<9} | {'FR':<9} | {'RL':<9} | {'RR':<9} | {'Avg':<8}")
        log("-" * 84)
        for r in rows:
            vals = [r[f"{key}_{leg}"] for leg in LEGS]
            g = CONDITION_LABELS.index(r["group"])
            cells = [(f"*{v:{fmt}}*" if g - 1 == k else f"{v:{fmt}}").ljust(9) for k, v in enumerate(vals)]
            log(f"{r['group']:<8} | " + " | ".join(cells) + f" | {np.mean(vals):{fmt}}")
        log("-" * 84)
        log("* 표시: 부상 다리 (부목 끝단 접촉 기준)")

    table("Duty Factor (접지 시간 비율)", "duty", ".3f")
    table("Contact Force (N, 접지 중 평균)", "force", ".2f")

    log("\n" + "=" * 84)
    log("Tracking / Stability")
    log("=" * 84)
    log(f"{'Group':<8} | {'samples':>8} | {'|v_xy err| m/s':>14} | {'|yaw err| rad/s':>15} | {'falls/min':>9}")
    log("-" * 84)
    for r in rows:
        log(f"{r['group']:<8} | {r['samples']:>8,} | {r['vel_err_xy']:>14.3f} | {r['yaw_err']:>15.3f} | {r['falls_per_min']:>9.2f}")

    if any("v_mae" in r for r in rows):
        log("\n" + "=" * 84)
        log("Phase 3 보조 헤드 (L̂ 는 부상 env 만; '수렴 후' = 리셋 5 s 이후 샘플)")
        log("=" * 84)
        log(f"{'Group':<8} | {'L̂ MAE 전체 mm':>14} | {'수렴 후 median':>14} | {'수렴 후 90%':>12} | {'v̂ MAE m/s':>10}")
        log("-" * 84)
        for r in rows:
            l0 = f"{r['L_mae_all_mm']:.1f}" if "L_mae_all_mm" in r else "-"
            l1 = f"{r['L_mae_mm']:.1f}" if "L_mae_mm" in r else "-"
            l2 = f"{r['L_mae90_mm']:.1f}" if "L_mae90_mm" in r else "-"
            log(f"{r['group']:<8} | {l0:>14} | {l1:>14} | {l2:>12} | {r['v_mae']:>10.3f}")


def plot_gait(rows, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(4)
    width = 0.15
    for ax, key, ylabel, title in (
        (ax1, "duty", "Duty Factor", "Duty Factor (Contact Time Ratio)"),
        (ax2, "force", "Average Force (N)", "Contact Force (injured leg = splint tip)"),
    ):
        for r in rows:
            g = CONDITION_LABELS.index(r["group"])
            ax.bar(x + (g - 2) * width, [r[f"{key}_{leg}"] for leg in LEGS], width,
                   label=r["group"], alpha=0.85, color=COLORS[g])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(LEGS)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax1.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_estimation(rec, path: Path) -> None:
    inj = rec["leg"] >= 0
    if not inj.any():
        return
    tr = rec["t_reset"][inj]
    errL = np.abs(rec["L_hat"] - rec["L"])[inj] * 1000.0
    L_gt = rec["L"][inj] * 1000.0
    L_hat = rec["L_hat"][inj] * 1000.0
    v_err = rec["v_err"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    bins = np.arange(0.0, min(20.0, tr.max() + 0.5), 0.5)

    ax = axes[0]
    bx, med, q90 = [], [], []
    for b in bins:
        m = (tr >= b) & (tr < b + 0.5)
        if m.sum() >= 20:
            bx.append(b + 0.25)
            med.append(np.median(errL[m]))
            q90.append(np.quantile(errL[m], 0.9))
    ax.plot(bx, med, lw=2, label="median", color=COLORS[0])
    ax.plot(bx, q90, lw=1.5, ls="--", label="90%", color=COLORS[0])
    ax.set_xlabel("time since reset [s]")
    ax.set_ylabel("|L̂ − L| [mm]")
    ax.set_title("Splint length estimation convergence (injured envs)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    conv = tr > 5.0
    sub = np.flatnonzero(conv)[::5]
    ax.scatter(L_gt[sub], L_hat[sub], s=4, alpha=0.25, color=COLORS[0])
    lo, hi = min(L_gt.min(), 300), max(L_gt.max(), 480)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("GT L [mm]")
    ax.set_ylabel("L̂ [mm]")
    ax.set_title("L̂ vs GT (converged, >5 s)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    cond = rec["leg"].astype(int) + 1
    for g in range(5):
        m = cond == g
        if m.any():
            ax.hist(v_err[m], bins=40, range=(0, 1.0), histtype="step", lw=1.5,
                    label=CONDITION_LABELS[g], color=COLORS[g], density=True)
    ax.set_xlabel("|v̂ − v| [m/s]")
    ax.set_ylabel("density")
    ax.set_title("Base velocity head error")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    env_cfg = build_env_cfg(
        config, num_envs=num_envs, seed=seed, device=device, eval_mode=eval_mode,
        splint_length=args.splint_length, clean=args.clean,
    )
    agent_cfg = build_agent_cfg(config, seed=seed, device=device)
    print_header(config, checkpoint, device, seed, num_envs, eval_mode, env_cfg)
    log(f"[eval] Steps / thresh : {args.steps} / {args.contact_threshold} N ({'|Fz|' if args.use_z_only else '||F||'})")

    env = wrap_rsl(make_gym_env(config, env_cfg), agent_cfg)
    base = env.unwrapped
    print_conditions(base)
    _, policy_fn, policy_module = make_policy(env, agent_cfg, config, checkpoint, device)
    log(f"[eval] Aux head       : {'있음 — L̂ / v̂ 로깅' if hasattr(policy_module, 'aux_inference') else '없음'}")

    rec = collect(env, base, policy_fn, policy_module, args.steps)
    rows = summarize(rec, args.contact_threshold)
    print_tables(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    gait_png = out_dir / "gait_analysis.png"
    plot_gait(rows, gait_png)
    log(f"\n[INFO] 저장: {gait_png}")
    if rec["has_aux"]:
        est_png = out_dir / "estimation_analysis.png"
        plot_estimation(rec, est_png)
        log(f"[INFO] 저장: {est_png}")
    csv_path = out_dir / "metrics_summary.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "group", k != "samples", k))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    log(f"[INFO] 저장: {csv_path}")
    env.close()


if __name__ == "__main__":
    # simulation_app.close() 가 종료 시 세그폴트를 낼 수 있어 os._exit 로 끝낸다.
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
