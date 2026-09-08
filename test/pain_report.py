"""수렴 정책의 통증 비용 C_pain 분포 리포트 (Isaac 불필요 — numpy/matplotlib).

rollout_dump.py 덤프에서 부상 env 의 (step, env) 표본을 모아 다음을 본다:
  - C_pain = min(([F−θ]₊/ρ)^n, 1) ∈ [0,1] 분포: P(C=0) / P(C>0) / 포화(C=1), 분위수
  - 유효 통증 하중 F_pain = F_foot + F_calf + η·F_splint 분포 (θ = 0.01·mg, ρ = mg/4)
  - 통증원 분해: 직접 접촉(발/calf) vs 부목 경유 — 통증 스텝에서 어느 쪽이 원인인가
  - 부상 다리별 / 부목 길이 L 구간별 통계
  - 에피소드당 누적 C_pain (리워드 기여 = weight · Σ C)

덤프에 `pain` 키(리워드 매니저가 그 스텝에 본 원시 C_pain)가 있으면 그것을 기준으로
쓰고, 접촉력에서 같은 식으로 재계산한 값과의 오차를 함께 찍는다 (식/파라미터
불일치 검출). 없는 옛 덤프는 재계산값만 쓴다 — 이때 파라미터는 meta.pain_params,
없으면 --env_yaml 의 injury.penalty_pain 을 읽는다. 체중 mg 는 덤프의 body_weight_n
(env 별) 을 쓰고, 없으면 --body_weight_n 으로 준다.

    /home/shw/miniconda3/envs/isaac/bin/python pain_report.py dumps/p2_0907_final_balanced.npz
    (여러 덤프를 주면 마지막에 비교표를 찍는다)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
ENV_YAML = HERE.parent / "scripts" / "rsl_rl" / "configs" / "env" / "antalgic.yaml"
LEGS = ("FL", "FR", "RL", "RR")
QS = (0.5, 0.9, 0.95, 0.99, 0.999)

# 차트 색 (dataviz 기본 팔레트, 라이트 서피스)
C_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
C_GRID, C_AXIS, C_MUTED, C_TEXT = "#e1e0d9", "#c3c2b7", "#898781", "#52514e"


# ── 파라미터 / 계산 ──────────────────────────────────────────────────────────

def load_pain_params(meta: dict, env_yaml: Path) -> tuple[dict, str]:
    """penalty_pain 파라미터. meta.pain_params(덤프 시점 실제값) > env yaml."""
    p = meta.get("pain_params") or {}
    src = "meta.pain_params"
    if not p:
        raw = yaml.safe_load(env_yaml.read_text())
        p = raw["reward"]["injury"]["penalty_pain"]
        src = str(env_yaml)
    return {
        "threshold_bw": float(p.get("threshold_bw", 0.01)),
        "scale_bw": float(p.get("scale_bw", 0.25)),
        "exponent": float(p.get("exponent", 1.0)),
        "include_calf": bool(p.get("include_calf", True)),
        "include_splint": bool(p.get("include_splint", True)),
        "eta": float(p.get("splint_transmission", 0.5)),
        "body_weight_n": p.get("body_weight_n"),
        "weight": float(meta.get("pain_weight", 0.0)),
    }, src


def body_weight_per_env(d: dict, prm: dict, override: float | None) -> np.ndarray:
    """env 별 mg [N]: CLI 값 > 파라미터 고정값 > 덤프의 body_weight_n."""
    n = d["gt_leg"].shape[1]
    if override is not None:
        return np.full(n, float(override))
    if prm["body_weight_n"] is not None:
        return np.full(n, float(prm["body_weight_n"]))
    if "body_weight_n" in d:
        return np.asarray(d["body_weight_n"], dtype=float)
    raise SystemExit("덤프에 body_weight_n 없음 — --body_weight_n <N> 을 지정하세요")


def injured_samples(d: dict, prm: dict, mg_env: np.ndarray) -> dict:
    """부상 env 의 (step, env) 표본과 rewards.penalty_pain 과 동일한 식의 C_pain.

    F_pain = F_foot + F_calf + η·F_splint
    C_pain = min( ([F_pain − θ]_+ / ρ)^n , 1 ),  θ = threshold_bw·mg,  ρ = scale_bw·mg
    """
    gt = d["gt_leg"]
    ti, ni = np.where(gt >= 0)
    k = gt[ti, ni].astype(int)
    foot = np.abs(d["contact_foot"][ti, ni, k, 2])
    calf = np.abs(d["contact_calf"][ti, ni, k, 2]) if prm["include_calf"] else np.zeros_like(foot)
    splint = np.abs(d["contact_splint"][ti, ni, k, 2]) if prm["include_splint"] else np.zeros_like(foot)
    direct = foot + calf
    F = direct + prm["eta"] * splint
    mg = mg_env[ni]
    theta = prm["threshold_bw"] * mg
    rho = prm["scale_bw"] * mg
    overload = np.clip(F - theta, 0.0, None) / rho
    C_calc = np.minimum(overload ** prm["exponent"], 1.0)
    out = dict(ti=ti, ni=ni, leg=k, foot=foot, calf=calf, splint=splint, direct=direct,
               F=F, mg=mg, theta=theta, rho=rho, overload=overload, C_calc=C_calc,
               L=d["gt_L"][ti, ni], dones=d["dones"][ti, ni])
    if "pain" in d:
        out["C_rec"] = d["pain"][ti, ni]
    out["C"] = out.get("C_rec", C_calc)
    return out


def dist(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "std": float(x.std()),
            **{f"p{q * 100:g}": float(np.quantile(x, q)) for q in QS},
            "max": float(x.max())}


def summarize(s: dict, prm: dict, mask: np.ndarray | None = None) -> dict:
    m = np.ones(len(s["C"]), bool) if mask is None else mask
    C, F = s["C"][m], s["F"][m]
    over = s["overload"][m] > 0          # F > θ
    sat = s["overload"][m] >= 1.0        # C 포화 (F ≥ θ + ρ)
    direct_c = s["direct"][m] > s["theta"][m]
    splint_c = s["splint"][m] > s["theta"][m]
    n = int(m.sum())
    r = {
        "n": n,
        "C": dist(C),
        "P_zero": float((C <= 0).mean()),
        "P_over": float(over.mean()),
        "P_sat": float(sat.mean()),
        "F": dist(F),
        "F_over_mg": dist(F / s["mg"][m]),
        "P_direct_contact": float(direct_c.mean()),
        "P_splint_contact": float(splint_c.mean()),
        "mean_direct_N": float(s["direct"][m].mean()),
        "mean_eta_splint_N": float(prm["eta"] * s["splint"][m].mean()),
        "weighted_mean_per_step": float(prm["weight"] * C.mean()),
    }
    if over.any():
        # 통증 스텝의 원인 분해 — 직접 접촉이 있었는가 / 부목만으로 넘었는가
        r["overload_cause"] = {
            "direct_involved": float(direct_c[over].mean()),
            "splint_only": float((~direct_c[over] & splint_c[over]).mean()),
            "mean_C": float(C[over].mean()),
            "mean_F": float(F[over].mean()),
        }
    return r


def episode_sums(s: dict, num_envs: int) -> np.ndarray:
    """env 별로 dones 경계로 끊어 에피소드당 Σ C_pain (끝나지 않은 꼬리 구간 포함)."""
    sums = []
    for e in range(num_envs):
        sel = s["ni"] == e
        if not sel.any():
            continue
        C, dn = s["C"][sel], s["dones"][sel]
        cut = np.flatnonzero(dn) + 1
        for seg in np.split(C, cut):
            if len(seg) >= 50:  # 1 s 미만 조각(리셋 직후 잔여)은 제외
                sums.append(float(seg.sum()))
    return np.asarray(sums)


# ── 출력 ───────────────────────────────────────────────────────────────────

def fmt_row(name: str, r: dict) -> str:
    c = r["C"]
    return (f"{name:>10} {r['n']:>7d} {r['P_zero'] * 100:>6.1f} {r['P_over'] * 100:>6.1f} "
            f"{r['P_sat'] * 100:>6.2f} {c['mean']:>8.4f} {c['p90']:>8.3f} {c['p99']:>8.3f} "
            f"{c['max']:>7.3f} {r['F']['mean']:>7.2f} {r['F']['p99']:>7.2f}")


HEADER = (f"{'group':>10} {'n':>7} {'C=0%':>6} {'C>0%':>6} {'sat%':>6} {'mean':>8} "
          f"{'p90':>8} {'p99':>8} {'max':>7} {'F_mean':>7} {'F_p99':>7}")


def print_report(name: str, meta: dict, prm: dict, src: str, s: dict, tot: dict,
                 by_leg: dict, by_L: dict, ep: np.ndarray) -> None:
    print("=" * 96)
    print(f"[{name}]  checkpoint: {meta.get('checkpoint', '?')}")
    print(f"  condition={meta.get('condition')}  envs={meta.get('num_envs')}  steps={meta.get('steps')}  "
          f"dt={meta.get('step_dt')}")
    mg_mean = float(s["mg"].mean())
    print(f"  pain params ({src}): θ={prm['threshold_bw']:g}·mg  ρ={prm['scale_bw']:g}·mg  n={prm['exponent']:g}  "
          f"η={prm['eta']:g}  calf={prm['include_calf']}  splint={prm['include_splint']}  weight={prm['weight']}")
    print(f"  mg: mean={mg_mean:.2f} N  [{s['mg'].min():.2f}, {s['mg'].max():.2f}]  "
          f"→ θ≈{prm['threshold_bw'] * mg_mean:.2f} N  ρ≈{prm['scale_bw'] * mg_mean:.2f} N  "
          f"(포화 F ≥ {(prm['threshold_bw'] + prm['scale_bw']) * mg_mean:.2f} N)")
    if "C_rec" in s:
        err = np.abs(s["C_rec"] - s["C_calc"])
        print(f"  기록값(reward manager) vs 재계산: max|Δ|={err.max():.3e}, "
              f"mean|Δ|={err.mean():.3e}  → 기준 = 기록값")
    else:
        print("  덤프에 pain 키 없음 → 접촉력 재계산값 사용")

    c = tot["C"]
    print(f"\n  C_pain ∈ [0,1] (부상 env·step, n={tot['n']}):")
    print(f"    mean={c['mean']:.4f}  std={c['std']:.4f}  "
          + "  ".join(f"{q}={c[q]:.3f}" for q in ("p50", "p90", "p95", "p99", "p99.9"))
          + f"  max={c['max']:.3f}")
    print(f"    P(C=0)={tot['P_zero'] * 100:.1f}%   P(C>0, F>θ)={tot['P_over'] * 100:.2f}%   "
          f"P(포화 C=1)={tot['P_sat'] * 100:.3f}%")
    print(f"    스텝당 기대 리워드 기여 weight·E[C] = {tot['weighted_mean_per_step']:+.4f}")
    f = tot["F"]
    fb = tot["F_over_mg"]
    print(f"\n  F_pain [N] (= F_foot + F_calf + {prm['eta']:g}·F_splint):")
    print(f"    mean={f['mean']:.2f}  "
          + "  ".join(f"{q}={f[q]:.2f}" for q in ("p50", "p90", "p95", "p99"))
          + f"  max={f['max']:.1f}   (F/mg: mean={fb['mean']:.3f}  p99={fb['p99']:.3f})")
    print(f"    직접접촉률(발/calf > θ)={tot['P_direct_contact'] * 100:.1f}%   "
          f"부목접지율(> θ)={tot['P_splint_contact'] * 100:.1f}%   "
          f"E[F_direct]={tot['mean_direct_N']:.2f} N   E[η·F_splint]={tot['mean_eta_splint_N']:.2f} N")
    oc = tot.get("overload_cause")
    if oc:
        print(f"    통증 스텝 원인: 직접접촉 동반 {oc['direct_involved'] * 100:.1f}%  /  "
              f"부목만 {oc['splint_only'] * 100:.1f}%   (통증 시 E[C]={oc['mean_C']:.3f}, E[F]={oc['mean_F']:.1f} N)")

    print(f"\n  부상 다리별:\n  {HEADER}")
    for leg, r in by_leg.items():
        print("  " + fmt_row(leg, r))
    print(f"\n  부목 길이 L 구간별:\n  {HEADER}")
    for lab, r in by_L.items():
        print("  " + fmt_row(lab, r))
    if len(ep):
        print(f"\n  에피소드당 Σ C_pain (n={len(ep)}): mean={ep.mean():.3f}  p50={np.median(ep):.3f}  "
              f"p90={np.quantile(ep, 0.9):.3f}  max={ep.max():.2f}   "
              f"→ weight·Σ = {prm['weight'] * ep.mean():+.3f} /에피소드")


def plot(name: str, s: dict, prm: dict, tot: dict, by_leg: dict, out_png: Path) -> None:
    C, F = s["C"], s["F"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C_AXIS)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.title.set_color(C_TEXT)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)

    # (a) C_pain > 0 의 로그 히스토그램 — C=0 질량은 텍스트로
    ax = axes[0]
    pos = C[C > 0]
    if len(pos):
        lo = max(pos.min(), 1e-4)
        bins = np.logspace(np.log10(lo), np.log10(pos.max() * 1.05), 40)
        w = np.full(len(pos), 1.0 / len(C))
        ax.hist(pos, bins=bins, weights=w, color=C_SERIES[0], edgecolor="#fcfcfb", lw=0.5)
        ax.set_xscale("log")
    ax.text(0.98, 0.95, f"P(C=0) = {tot['P_zero'] * 100:.1f}%\nP(C>0) = {(1 - tot['P_zero']) * 100:.1f}%\n"
            f"P(C=1) = {tot['P_sat'] * 100:.2f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=C_TEXT)
    ax.set_xlabel("C_pain (log)")
    ax.set_ylabel("fraction of injured steps")
    ax.set_title("C_pain distribution (C > 0 shown)")

    # (b) F_pain 히스토그램 + F_th
    ax = axes[1]
    theta, rho = float(s["theta"].mean()), float(s["rho"].mean())
    hi = max(float(np.quantile(F, 0.999)) * 1.1, (theta + rho) * 1.2, 1.0)
    ax.hist(F, bins=np.linspace(0, hi, 50), weights=np.full(len(F), 1.0 / len(F)),
            color=C_SERIES[0], edgecolor="#fcfcfb", lw=0.5)
    for x, lab in ((theta, f" θ={theta:.1f} N"), (theta + rho, f" θ+ρ={theta + rho:.1f} N")):
        ax.axvline(x, color=C_MUTED, lw=1, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.95, lab, color=C_MUTED, fontsize=8, va="top")
    ax.text(0.98, 0.95, f"P(F>θ) = {tot['P_over'] * 100:.2f}%\n"
            f"E[F] = {tot['F']['mean']:.2f} N", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=C_TEXT)
    ax.set_xlabel("F_pain [N]  (foot + calf + η·splint, |Fz|)")
    ax.set_title("Effective pain load (99.9% range)")

    # (c) 다리별 ECDF (로그 x, C>0 구간)
    ax = axes[2]
    for i, leg in enumerate(LEGS):
        m = s["leg"] == i
        if not m.any():
            continue
        x = np.sort(C[m])
        y = np.arange(1, len(x) + 1) / len(x)
        keep = x > 0
        ax.step(x[keep], y[keep], where="post", color=C_SERIES[i], lw=2,
                label=f"{leg}  (mean {by_leg[leg]['C']['mean']:.3f})")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("C_pain (log)")
    ax.set_ylabel("ECDF")
    ax.set_title("Per-leg ECDF of C_pain")
    ax.legend(frameon=False, fontsize=8, labelcolor=C_TEXT, loc="lower right")

    fig.suptitle(f"C_pain — {name}", color=C_TEXT, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────

def analyze(path: Path, env_yaml: Path, out_dir: Path, no_plot: bool,
            body_weight_n: float | None = None) -> dict:
    npz = np.load(path)
    d = {k: npz[k] for k in npz.files if k != "meta"}
    meta = json.loads(str(npz["meta"]))
    prm, src = load_pain_params(meta, env_yaml)
    mg_env = body_weight_per_env(d, prm, body_weight_n)
    s = injured_samples(d, prm, mg_env)
    if len(s["C"]) == 0:
        raise ValueError(f"{path.name}: 부상 env 표본이 없다 (condition={meta.get('condition')})")

    tot = summarize(s, prm)
    by_leg = {leg: summarize(s, prm, s["leg"] == i) for i, leg in enumerate(LEGS)
              if (s["leg"] == i).any()}
    L = s["L"]
    edges = np.linspace(L.min(), L.max() + 1e-6, 4)
    by_L = {}
    for j in range(3):
        m = (L >= edges[j]) & (L < edges[j + 1])
        if m.any():
            by_L[f"L{edges[j]:.2f}-{edges[j + 1]:.2f}"] = summarize(s, prm, m)
    ep = episode_sums(s, int(meta.get("num_envs", d["gt_leg"].shape[1])))

    name = path.stem
    print_report(name, meta, prm, src, s, tot, by_leg, by_L, ep)
    result = {"dump": str(path), "checkpoint": meta.get("checkpoint"), "pain_params": prm,
              "params_source": src, "uses_recorded_pain": "C_rec" in s, "total": tot,
              "by_leg": by_leg, "by_L": by_L,
              "episode_sum": dist(ep) if len(ep) else None}
    if "C_rec" in s:
        result["rec_vs_calc_max_abs_err"] = float(np.abs(s["C_rec"] - s["C_calc"]).max())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"pain_dist_{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  저장: {out_dir / f'pain_dist_{name}.json'}")
    if not no_plot:
        png = out_dir / f"pain_dist_{name}.png"
        plot(name, s, prm, tot, by_leg, png)
        print(f"  저장: {png}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="C_pain 분포 리포트")
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--env_yaml", type=Path, default=ENV_YAML,
                    help="meta 에 pain_params 가 없을 때 읽을 env yaml")
    ap.add_argument("--out_dir", type=Path, default=HERE / "dumps" / "analysis")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--body_weight_n", type=float, default=None,
                    help="체중 mg [N] 고정값 (덤프에 body_weight_n 이 없을 때)")
    args = ap.parse_args()

    results = [analyze(p, args.env_yaml, args.out_dir, args.no_plot, args.body_weight_n)
               for p in args.dumps]
    if len(results) > 1:
        print("\n" + "=" * 96 + f"\n비교\n{HEADER}")
        for p, r in zip(args.dumps, results):
            print(fmt_row(p.stem[:10], r["total"]))


if __name__ == "__main__":
    main()
