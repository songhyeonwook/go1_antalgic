"""부목 끝단 마찰 μ 추정 연구 — EMA 시정수 / 윈도우 길이 비교.

물리적 배경: 접촉력 비율 ρ = F_t / F_n 은 stick 중에는 ρ ≤ μ (부등식 정보),
슬립 순간에만 ρ ≈ μ (등식). 따라서:
  - ρ 의 EMA        → "마찰 사용률 평균"으로 수렴 (μ 아님 — 하향 편향)
  - ρ 의 윈도우 최대/고분위 → 윈도우 안에 근접-슬립 이벤트가 있어야 μ 에 접근

산출물:
  dumps/analysis/mu_study.png  — ① EMA 시정수별 / ② 윈도우 길이별 수렴 곡선,
                                 ③ 속도별 슬립 근접도 (식별 가능성)
  콘솔 표 — 시점별 추정 오차, 권고 요약

    /home/shw/miniconda3/envs/isaac/bin/python mu_estimation_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK KR", "Noto Sans CJK SC", "Noto Sans CJK HK", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MAIN_DUMP = HERE / "dumps" / "p3_004_balanced.npz"
SWEEPS = [("0.3", "sweep_v030"), ("0.5", "sweep_v050"),
          ("0.7", "sweep_v070"), ("0.9", "sweep_v090")]

STANCE_N = 10.0          # ρ 계산에 쓸 최소 수직력 (저하중 ρ 는 노이즈 지배)
DT = 0.02
EMA_TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)          # [s]
WINDOWS = (0.5, 1.0, 2.0, 4.0, 8.0, None)      # [s], None = 에피소드 전체 누적
QUANT = 0.95             # 윈도우 추정기의 분위 (max 는 outlier 취약)
EVAL_STRIDE = 5          # 곡선 평가 격자 (0.1 s)


def load(path):
    npz = np.load(path)
    d = {k: npz[k] for k in npz.files if k != "meta"}
    return d, json.loads(str(npz["meta"]))


def episodes(d, min_len=150):
    """(env, s, e, leg, mu) — done 스텝은 새 에피소드 시작 (경계 검증됨)."""
    T, N = d["gt_leg"].shape
    out = []
    for n in range(N):
        done_ts = list(np.where(d["dones"][:, n])[0])
        for s, e in zip([0] + done_ts, [t - 1 for t in done_ts] + [T - 1]):
            k = int(d["gt_leg"][min(s + 10, e), n])
            if e - s < min_len or k < 0:
                continue
            out.append((n, s, e, k, float(d["gt_mu"][min(s + 10, e), n])))
    return out


def ratio_series(d, n, s, e, k):
    """에피소드 내 ρ_t (스탠스 아니면 NaN)."""
    f = d["contact_splint"][s:e + 1, n, k]        # (L, 3) world
    fn = np.abs(f[:, 2])
    ft = np.hypot(f[:, 0], f[:, 1])
    rho = np.where(fn > STANCE_N, ft / np.maximum(fn, 1e-6), np.nan)
    return rho


def ema_curve(rho, tau):
    """스탠스 샘플에만 갱신되는 EMA (시정수 tau 초)."""
    alpha = np.exp(-DT / tau)
    est, out = np.nan, np.empty_like(rho)
    for i, r in enumerate(rho):
        if not np.isnan(r):
            est = r if np.isnan(est) else alpha * est + (1 - alpha) * r
        out[i] = est
    return out


def window_curve(rho, window):
    """트레일링 윈도우의 q95 (window=None 이면 처음부터 누적)."""
    L = len(rho)
    out = np.full(L, np.nan)
    w = L if window is None else int(round(window / DT))
    for i in range(0, L, EVAL_STRIDE):
        seg = rho[max(0, i - w + 1):i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 5:
            out[i] = np.quantile(seg, QUANT)
    # 평가 격자 사이는 직전 값 유지
    last = np.nan
    for i in range(L):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def collect_curves(d, eps, fn, args_list):
    """각 파라미터에 대해 시간격자별 '중앙값 부호오차' 곡선을 만든다."""
    t_grid = np.arange(0, 900, EVAL_STRIDE)      # 최대 18 s
    curves = {}
    for arg in args_list:
        errs = [[] for _ in t_grid]
        for n, s, e, k, mu in eps:
            rho = ratio_series(d, n, s, e, k)
            est = fn(rho, arg)
            for gi, t in enumerate(t_grid):
                if t < len(est) and not np.isnan(est[t]):
                    errs[gi].append(est[t] - mu)
        curves[arg] = (
            t_grid * DT,
            np.array([np.median(v) if len(v) >= 10 else np.nan for v in errs]),
        )
    return curves


def main():
    d, meta = load(MAIN_DUMP)
    eps = episodes(d)
    print(f"메인 덤프: {MAIN_DUMP.name} (phase {meta['phase']}), "
          f"부상 에피소드 {len(eps)}개, μ 범위 "
          f"[{min(m for *_, m in eps):.2f}, {max(m for *_, m in eps):.2f}]")

    # ── 슬립 근접도 (식별 가능성) ──
    def proximity(d, eps):
        near = {0.8: 0, 0.9: 0, 0.95: 0}
        tot = 0
        for n, s, e, k, mu in eps:
            rho = ratio_series(d, n, s, e, k)
            v = rho[~np.isnan(rho)]
            tot += len(v)
            for th in near:
                near[th] += int((v > th * mu).sum())
        return {th: c / max(tot, 1) for th, c in near.items()}, tot

    prox_main, n_main = proximity(d, eps)
    print(f"\n[식별 가능성] 스탠스 샘플 {n_main}개 중 ρ>0.9μ 비율: "
          f"{prox_main[0.9]*100:.1f}%  (근접-슬립 이벤트 공급량)")

    prox_by_speed = {}
    for label, name in SWEEPS:
        try:
            ds, _ = load(HERE / "dumps" / f"{name}.npz")
            ps, cnt = proximity(ds, episodes(ds))
            prox_by_speed[label] = ps
            print(f"  v={label} m/s: ρ>0.8μ {ps[0.8]*100:5.1f}% | "
                  f">0.9μ {ps[0.9]*100:5.1f}% | >0.95μ {ps[0.95]*100:5.1f}%")
        except FileNotFoundError:
            pass

    # ── 수렴 곡선 ──
    ema_curves = collect_curves(d, eps, ema_curve, EMA_TAUS)
    win_curves = collect_curves(d, eps, window_curve, WINDOWS)

    # ── 시점별 요약 표 ──
    def at_time(curves, t_s):
        i_target = t_s / DT
        rows = {}
        for arg, (ts, med) in curves.items():
            idx = np.argmin(np.abs(ts - t_s))
            rows[arg] = med[idx]
        return rows

    print("\n[EMA] 부호오차 median (μ̂ − μ), 시점별:")
    for tau in EMA_TAUS:
        r5, r15 = at_time(ema_curves, 5.0)[tau], at_time(ema_curves, 15.0)[tau]
        print(f"  τ={tau:4.2f}s : 5s {r5:+.3f} | 15s {r15:+.3f}")
    print("[윈도우 q95] 부호오차 median:")
    for w in WINDOWS:
        r5, r15 = at_time(win_curves, 5.0)[w], at_time(win_curves, 15.0)[w]
        lab = "누적(∞)" if w is None else f"{w:.1f}s"
        print(f"  W={lab:7s}: 5s {r5:+.3f} | 15s {r15:+.3f}")

    # ── 그림 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for tau, (ts, med) in ema_curves.items():
        ax.plot(ts, med, label=f"τ={tau}s")
    ax.set_title("(1) EMA of ratio — by time constant")
    ax = axes[1]
    for w, (ts, med) in win_curves.items():
        ax.plot(ts, med, label="cumulative" if w is None else f"W={w}s")
    ax.set_title(f"(2) windowed q{int(QUANT*100)} of ratio — by window")
    for ax in axes[:2]:
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("episode time [s]")
        ax.set_ylabel("median(μ̂ − μ)")
        ax.set_ylim(-1.05, 0.15)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[2]
    if prox_by_speed:
        xs = [float(v) for v in prox_by_speed]
        for th, mk in ((0.8, "o-"), (0.9, "s-"), (0.95, "^-")):
            ax.plot(xs, [prox_by_speed[f"{x:.1f}"][th] * 100 for x in xs], mk,
                    label=f"ρ > {th}μ")
        ax.set_title("(3) near-slip fraction vs speed (identifiability)")
        ax.set_xlabel("command speed [m/s]")
        ax.set_ylabel("stance samples [%]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    out = HERE / "dumps" / "analysis"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "mu_study.png", dpi=140)
    print(f"\n그림 저장: {out / 'mu_study.png'}")


if __name__ == "__main__":
    main()
