"""GT L 민감도 — 폐루프 리포트 (teacher 입력 L 채널만 조작한 롤아웃 비교).

`rollout_dump.py --l_obs_fixed X` (부상 env 의 privileged L 채널만 X 로 고정,
물리 부목 길이는 무손상) 산출물을 모아 μ 리포트와 같은 지표로 비교한다:
  - 속도 추종 오차 / 평균 에피소드 길이
  - 부상 다리 하중: 부목 duty, 스탠스 접지력, 전체 평균 수직력
  - 통증 접촉률 / 스탠스 팁 슬립

전 지표가 주입 L 에 대해 평탄하면 = teacher 가 GT L 을 사실상 쓰지 않는다.

    /home/shw/miniconda3/envs/isaac/bin/python l_sensitivity_report.py [덤프디렉터리]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from dump_utils import quat_rot_inv_x

STANCE_N = 10.0
DT = 0.02
CONDS = [("base", "원본 (GT L)"), ("L033", "L→0.33 (범위 최소)"),
         ("L039", "L→0.39 (prior)"), ("L045", "L→0.45 (범위 최대)"),
         ("L000", "L→0.00 (정상 sentinel)")]


def metrics(path: Path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"]))
    gt = d["gt_leg"]
    inj_env = (gt >= 0).any(0)

    vx = quat_rot_inv_x(d["root_state"][..., 3:7], d["root_state"][..., 7:10])
    track_err = np.abs(vx - d["commands"][..., 0])[:, inj_env].mean()

    # 낙상 = 시간초과(모든 env 공통 step)가 아닌 종료. ep 길이 평균은 이 설정에서
    # env 당 done 이 1 회(시간초과)뿐이라 T 로 포화되므로 지표로 쓰지 않는다.
    dn = d["dones"]
    idx = np.concatenate([np.flatnonzero(dn[:, n]) for n in range(dn.shape[1])]) \
        if dn.any() else np.zeros(0, int)
    timeout_step = np.bincount(idx).argmax() if idx.size else -1
    falls = int((idx != timeout_step).sum())

    ti, ni = np.where(gt >= 0)
    k = gt[ti, ni]
    fsp = d["contact_splint"][ti, ni, k]
    fn = np.abs(fsp[:, 2])
    st = fn > STANCE_N
    duty = (fn > 5.0).mean()
    force = fn[st].mean() if st.any() else 0.0
    load_mean = fn.mean()                      # 시간평균 하중 (duty × 강도 합성)

    pain = (np.linalg.norm(d["contact_foot"][ti, ni, k], axis=-1)
            + np.linalg.norm(d["contact_calf"][ti, ni, k], axis=-1))
    pain_rate = (pain > 1.0).mean()

    slips = []
    for n in range(gt.shape[1]):
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

    gl = d["gt_L"][gt >= 0]
    return {
        "l_obs": meta.get("l_obs_fixed"),
        "gt_L_med": float(np.median(gl[gl > 0])) if (gl > 0).any() else 0.0,
        "track_err": float(track_err),
        "falls": falls,
        "duty": float(duty),
        "force": float(force),
        "load": float(load_mean),
        "pain": float(pain_rate),
        "slip": float(np.quantile(slip, 0.95)),
    }


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "dumps" / "lsens"
    rows = []
    for tag, label in CONDS:
        p = root / f"{tag}.npz"
        if not p.exists():
            print(f"누락: {p}")
            continue
        r = metrics(p)
        r["label"] = label
        rows.append(r)
    if not rows:
        raise SystemExit("덤프 없음")

    print(f"{'조건':<24} {'실제 L':>7} {'추종오차':>9} {'낙상':>5} "
          f"{'duty':>6} {'접지력[N]':>10} {'평균하중[N]':>11} {'슬립p95':>8} {'통증':>7}")
    for r in rows:
        print(f"{r['label']:<24} {r['gt_L_med']:>7.3f} {r['track_err']:>9.3f} "
              f"{r['falls']:>5d} {r['duty']:>6.2f} {r['force']:>10.1f} "
              f"{r['load']:>11.1f} {r['slip']:>8.3f} {r['pain']*100:>6.2f}%")

    base = rows[0]
    print(f"\n원본 대비 변화율 (%):")
    print(f"{'조건':<24} {'추종오차':>9} {'슬립p95':>9} {'duty':>7} "
          f"{'접지력':>8} {'평균하중':>9}")
    for r in rows[1:]:
        def pct(k):
            return 100 * (r[k] - base[k]) / base[k] if base[k] else float("nan")
        print(f"{r['label']:<24} {pct('track_err'):>+8.1f}% {pct('slip'):>+8.1f}% "
              f"{pct('duty'):>+6.1f}% {pct('force'):>+7.1f}% {pct('load'):>+8.1f}%")


if __name__ == "__main__":
    main()
