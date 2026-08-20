"""μ 강건성 리포트 — "부목 끝단 마찰에 강건하다" 주장의 정량 증거.

mu_sweep_*.npz (rollout_dump --fixed_mu 산출물)를 모아 μ별 성능 지표를 비교:
  - 속도 추종 오차 (body-frame vx vs 명령)
  - 에피소드 조기종료(낙상) 비율 / 평균 에피소드 길이
  - 부목 duty / 접지력 (antalgic 보행 유지 여부)
  - 마찰 사용률 ρ (μ 여유 확인)
  - 부상 발/calf 접촉 (통증 회피 유지)

전 지표가 μ 에 대해 평탄하면 = 강건성 입증. DR 학습 범위 [0.5, 1.5] 밖
외삽(μ=0.3, 2.0)까지 평탄하면 더 강한 주장.

    /home/shw/miniconda3/envs/isaac/bin/python mu_robustness_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MUS = (0.3, 0.6, 1.0, 1.5, 2.0)
TRAIN_RANGE = (0.5, 1.5)
STANCE_N = 10.0
DT = 0.02


def quat_rot_inv_x(quat, v):
    """world 벡터 v 를 body x 축으로 투영 (R^T v 의 x 성분)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # body x축의 world 표현 = R @ [1,0,0]
    bx = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
    return (bx * v).sum(-1)


def metrics(path):
    npz = np.load(path)
    d = {k: npz[k] for k in npz.files if k != "meta"}
    meta = json.loads(str(npz["meta"]))
    gt = d["gt_leg"]
    inj_env = (gt >= 0).any(0)

    # 속도 추종 (부상 env, 워밍업 제외 전 스텝)
    vx_body = quat_rot_inv_x(d["root_state"][..., 3:7], d["root_state"][..., 7:10])
    track_err = np.abs(vx_body - d["commands"][..., 0])[:, inj_env].mean()

    # 낙상/에피소드 길이 (time_out 1000 step 이면 dones 는 timeout 포함 —
    # 조기종료 비율은 평균 에피소드 길이로 간접 측정)
    T = gt.shape[0]
    ep_len = T / np.maximum(d["dones"][:, inj_env].sum(0), 1)

    # 부목 접지 (부상 다리)
    ti, ni = np.where(gt >= 0)
    k = gt[ti, ni]
    fsp = d["contact_splint"][ti, ni, k]
    fn = np.abs(fsp[:, 2])
    ft = np.hypot(fsp[:, 0], fsp[:, 1])
    st = fn > STANCE_N
    duty = (fn > 5.0).mean()
    force = fn[st].mean() if st.any() else 0.0
    rho95 = np.quantile(ft[st] / fn[st], 0.95) if st.any() else 0.0

    # 통증 접촉 (부상 발+calf)
    pain = (np.linalg.norm(d["contact_foot"][ti, ni, k], axis=-1)
            + np.linalg.norm(d["contact_calf"][ti, ni, k], axis=-1))
    pain_rate = (pain > 1.0).mean()

    # 기구학적 슬립: 스탠스 중 팁 수평속도 (리셋 순간이동 diff 는 dones 로 제외).
    # ⚠️ 접촉센서 net_forces_w 는 접선 성분을 보고하지 않으므로(실측 ρ≡0)
    # 슬립은 힘이 아니라 기구학으로 측정해야 한다.
    slips = []
    N = gt.shape[1]
    for n in range(N):
        legs = gt[:, n]
        if (legs < 0).all():
            continue
        kk = int(legs[legs >= 0][0])
        pos = d["pos_splint_w"][:, n, kk]
        fn_n = np.abs(d["contact_splint"][:, n, kk, 2])
        v_xy = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / DT
        ok = (fn_n[1:] > STANCE_N) & (fn_n[:-1] > STANCE_N)
        ok &= ~d["dones"][:-1, n] & ~d["dones"][1:, n]
        slips.append(v_xy[ok])
    slip = np.concatenate(slips) if slips else np.zeros(1)

    # μ 라벨은 실제 적용값(gt_mu)에서 유도 — meta 는 요청값이라 어긋날 수 있음
    gm = d["gt_mu"][gt >= 0]
    mu_applied = float(np.median(gm[gm > 0])) if (gm > 0).any() else None
    meta_mu = meta.get("fixed_mu")
    if mu_applied is not None and meta_mu is not None and abs(mu_applied - meta_mu) > 1e-3:
        print(f"  ⚠️ {Path(path).name}: meta μ={meta_mu} vs 실측 μ={mu_applied:.3f} 불일치")
    return {
        "mu": mu_applied if mu_applied is not None else meta_mu,
        "track_err": float(track_err),
        "ep_len_mean": float(ep_len.mean()),
        "duty": float(duty),
        "force": float(force),
        "rho95": float(rho95),
        "pain_rate": float(pain_rate),
        "slip_p95": float(np.quantile(slip, 0.95)),
    }


def main():
    rows = []
    for mu in MUS:
        p = HERE / "dumps" / f"mu_sweep_{mu}.npz"
        if p.exists():
            rows.append(metrics(p))
        else:
            print(f"누락: {p.name}")
    if not rows:
        raise SystemExit("스윕 덤프 없음")

    print(f"{'μ':>5} {'추종오차[m/s]':>12} {'평균ep길이':>10} {'duty':>6} "
          f"{'접지력[N]':>9} {'슬립p95[m/s]':>12} {'통증접촉':>8}")
    for r in rows:
        print(f"{r['mu']:>5.1f} {r['track_err']:>12.3f} {r['ep_len_mean']:>10.0f} "
              f"{r['duty']:>6.2f} {r['force']:>9.1f} {r['slip_p95']:>12.3f} "
              f"{r['pain_rate']*100:>7.2f}%")

    mus = [r["mu"] for r in rows]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    panels = [
        ("track_err", "velocity tracking error [m/s]", (0, None)),
        ("ep_len_mean", "mean episode length [steps]", (0, None)),
        ("duty", "splint duty factor", (0, 1)),
        ("force", "splint stance force [N]", (0, None)),
        ("slip_p95", "stance tip slip p95 [m/s]", (0, None)),
    ]
    for ax, (key, label, ylim) in zip(axes, panels):
        ax.plot(mus, [r[key] for r in rows], "o-", color="#1a6faf")
        ax.axvspan(*TRAIN_RANGE, alpha=0.12, color="green",
                   label="training DR range")
        ax.set_xlabel("splint-tip friction μ")
        ax.set_title(label, fontsize=10)
        if ylim[1] is not None:
            ax.set_ylim(*ylim)
        else:
            lo = 0
            hi = max(r[key] for r in rows) * 1.3 + 1e-6
            ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    out = HERE / "dumps" / "analysis" / "mu_robustness.png"
    out.parent.mkdir(exist_ok=True)
    fig.suptitle("Deployed student policy vs splint-tip friction "
                 "(flat = robust; shaded = training range)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n그림 저장: {out}")


if __name__ == "__main__":
    main()
