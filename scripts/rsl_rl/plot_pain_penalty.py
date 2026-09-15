#!/usr/bin/env python3
"""논문 Fig. pain: 정규화 통각 비용 C_pain (v4) 을 부목 끝단 수직력에 대해 그린다.

    python plot_pain_penalty.py                       # antalgic.yaml 의 penalty_pain 값 사용
    python plot_pain_penalty.py --mg 130 --mg_range 122 138 --op_band 33 60 --healthy_band 64 86

식 (eq:cpain):  C_pain = min( ([η·F_splint − θ]_+ / ρ)^n , 1 ),  θ = θ_bw·mg,  ρ = ρ_bw·mg
출력: figure/fig_pain_penalty.png / .pdf  (Isaac 불필요)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

p = argparse.ArgumentParser()
p.add_argument("--env_yaml", type=Path, default=HERE / "configs" / "env" / "antalgic.yaml")
p.add_argument("--mg", type=float, default=130.0, help="공칭 체중 [N] (덤프 평균)")
p.add_argument("--mg_range", type=float, nargs=2, default=(122.0, 138.0), help="질량 랜덤화 체중 범위 [N]")
p.add_argument("--op_band", type=float, nargs=2, default=(33.0, 60.0), help="학습된 정책의 부목 끝단 per-stance peak [N]")
p.add_argument("--healthy_band", type=float, nargs=2, default=(64.0, 86.0), help="정상 정책의 다리별 peak [N]")
p.add_argument("--w_task", type=float, default=2.5, help="선속도 추적 가중치 (스텝당 최대 보상 = w_task·dt)")
p.add_argument("--dt", type=float, default=0.02)
p.add_argument("--fmax", type=float, default=130.0, help="가로축 상한 [N]")
p.add_argument("--out", type=Path, default=ROOT / "figure" / "fig_pain_penalty")
args = p.parse_args()

cfg = yaml.safe_load(open(args.env_yaml))["reward"]["injury"]["penalty_pain"]
th_bw, rho_bw, n = float(cfg["threshold_bw"]), float(cfg["scale_bw"]), float(cfg["exponent"])
eta, w_pain = float(cfg["splint_transmission"]), abs(float(cfg["weight"]))


def c_pain(F_splint: np.ndarray, mg: float) -> np.ndarray:
    theta, rho = th_bw * mg, rho_bw * mg
    over = np.clip(eta * F_splint - theta, 0.0, None) / rho
    return np.minimum(over ** n, 1.0)


F = np.linspace(0.0, args.fmax, 1000)
C = c_pain(F, args.mg)
C_lo, C_hi = c_pain(F, args.mg_range[1]), c_pain(F, args.mg_range[0])   # 무거울수록 비용 작음
F_th = th_bw * args.mg / eta
F_sat = (th_bw + rho_bw) * args.mg / eta
r_task_max = args.w_task * args.dt

# ── 색 (단일 계열 + 주석 밴드) ───────────────────────────────────────────────
INK, INK2 = "#1f2430", "#5d6675"
CURVE = "#2f5fb3"
OP, HEALTHY = "#e08a1e", "#8b939f"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=150)


def decorate(ax, ylabel: str, title: str, band_labels: bool = True):
    ax.axvspan(*args.op_band, color=OP, alpha=0.18, lw=0, label="antalgic teacher, per-stance peaks" if band_labels else None)
    ax.axvspan(*args.healthy_band, color=HEALTHY, alpha=0.25, lw=0, label="healthy-limb peaks (intact policy)" if band_labels else None)
    ax.axvline(F_th, color=INK2, ls=":", lw=1.2)
    ax.set_xlim(0, args.fmax)
    ax.set_xlabel(r"splint-tip vertical force $F_z^{\mathrm{splint}}$ [N]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="#e3e6ea", lw=0.6)
    ax.set_axisbelow(True)
    top = ax.secondary_xaxis("top", functions=(lambda f: f / args.mg * 100.0, lambda pct: pct * args.mg / 100.0))
    top.set_xlabel("% body weight", color=INK2)
    top.tick_params(colors=INK2)
    top.spines["top"].set_visible(True)


# (a) C_pain
decorate(ax_a, r"$C_{\mathrm{pain}}$", "a")
ax_a.fill_between(F, C_lo, C_hi, color=CURVE, alpha=0.18, lw=0, label=f"±15 % mass randomisation ({args.mg_range[0]:.0f}–{args.mg_range[1]:.0f} N)")
ax_a.plot(F, C, color=CURVE, lw=2, label=f"nominal $mg$ = {args.mg:.0f} N")
ax_a.set_ylim(0, max(c_pain(np.array([args.fmax]), args.mg_range[0])[0], 0.3) * 1.08)
ax_a.annotate(rf"$\theta/\eta$ = {F_th:.1f} N", (F_th, ax_a.get_ylim()[1] * 0.97), xytext=(6, 0),
              textcoords="offset points", va="top", fontsize=8, color=INK2)
ax_a.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.90))
for f in (args.op_band[0], args.op_band[1], args.healthy_band[1]):
    c = c_pain(np.array([f]), args.mg)[0]
    ax_a.plot(f, c, "o", ms=5, color=CURVE, mec="white", mew=1.2)
    ax_a.annotate(f"{c:.3f}", (f, c), xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=8, color=INK)
ax_a.text(args.fmax * 0.985, 0.006, rf"saturates at {F_sat:.0f} N" + "\n" + r"($\approx 2\,mg$, off-scale $\rightarrow$)",
          ha="right", va="bottom", fontsize=7.5, color=INK2, linespacing=1.3)

# (b) per-step penalty
decorate(ax_b, r"per-step penalty $W_{\mathrm{pain}}\,\Delta t\,C_{\mathrm{pain}}$", "b", band_labels=False)
ax_b.fill_between(F, w_pain * args.dt * C_lo, w_pain * args.dt * C_hi, color=CURVE, alpha=0.18, lw=0)
ax_b.plot(F, w_pain * args.dt * C, color=CURVE, lw=2, label=rf"$W_{{\mathrm{{pain}}}}$ = {w_pain:g}, $\Delta t$ = {args.dt:g} s")
ax_b.axhline(r_task_max, color=INK, ls="--", lw=1.2, label=rf"max per-step tracking reward $W_{{\mathrm{{task}}}}\Delta t$ = {r_task_max:.2f}")
ax_b.set_ylim(0, max(r_task_max * 1.6, w_pain * args.dt * C_hi.max() * 1.08))
for f in (args.op_band[0], args.op_band[1], args.healthy_band[1]):
    v = w_pain * args.dt * c_pain(np.array([f]), args.mg)[0]
    ax_b.plot(f, v, "o", ms=5, color=CURVE, mec="white", mew=1.2)
    ax_b.annotate(f"{v:.3f}", (f, v), xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=8, color=INK)
ax_b.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.0))

fig.tight_layout(w_pad=2.0)
args.out.parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
print(f"params: theta_bw={th_bw} rho_bw={rho_bw} n={n} eta={eta} W_pain={w_pain}  mg={args.mg} N")
print(f"F_th/eta = {F_th:.2f} N, saturation = {F_sat:.1f} N")
for f in (*args.op_band, *args.healthy_band):
    c = c_pain(np.array([f]), args.mg)[0]
    print(f"  F_splint={f:5.1f} N  C_pain={c:.4f}  per-step={w_pain*args.dt*c:.4f}")
print("saved:", f"{args.out}.png", f"{args.out}.pdf")
