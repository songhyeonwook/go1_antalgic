"""보행 충격량 · 대칭지수(SI) 분석 (rollout_dump .npz — Isaac 불필요).

analyze_dump_perf.py 의 [E](duty/접촉력) 를 넘어 임상 보행분석 계열 지표를 뽑는다:

  [I] 충격량 지표 (stance 이벤트 단위, 다리별)
      - peak force        : stance 중 최대 수직력 [N]
      - touchdown peak    : 착지 후 0.1 s 내 최대 힘 [N] (충격 스파이크)
      - loading rate      : 착지 초기 60 ms 의 최대 dF/dt [N/s]
      - impulse           : stance 전체 ∫F dt [N·s]
      - stance time       : 접지 지속 시간 [s]
  [S] 대칭지수 SI (Robinson): SI(a,b) = (Xa − Xb) / (0.5·(Xa+Xb)) × 100 [%]
      - 앞다리 좌/우 (FL vs FR), 뒷다리 좌/우 (RL vs RR)
      - 부상 다리 vs 반대쪽(같은 girdle) — 절뚝임 정량화
  [R] 체간 안정성: 높이 std, 수직속도 RMS, roll/pitch RMS

부상 다리는 발이 접혀 있으므로 부목 접촉력을, 나머지는 발 접촉력을 쓴다.

    PYTHONPATH= python3 analyze_gait_metrics.py dumps/p3_aux_balanced.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
GROUP_NAMES = {-1: "Normal", 0: "FL Peg", 1: "FR Peg", 2: "RL Peg", 3: "RR Peg"}
CONTACT_N = 5.0        # 접지 판정 [N]
MIN_STANCE_STEPS = 3   # 60 ms 미만 접촉은 채터링으로 간주
TOUCHDOWN_STEPS = 5    # 착지 스파이크 관찰 창 0.1 s
LOADRATE_STEPS = 3     # loading rate 창 60 ms
# 같은 girdle 의 반대쪽 다리: FL<->FR, RL<->RR
CONTRA = {0: 1, 1: 0, 2: 3, 3: 2}


def stance_events(force: np.ndarray, dones: np.ndarray, dt: float) -> dict:
    """단일 (env, leg) 힘 시계열에서 stance 이벤트 지표를 수집한다."""
    contact = force > CONTACT_N
    # done 스텝은 세그먼트 경계 — 이벤트가 에피소드를 넘지 않게 자름
    out = {k: [] for k in ("peak", "td_peak", "load_rate", "impulse", "stance_t")}
    t = 0
    T = len(force)
    while t < T:
        if contact[t] and (t == 0 or not contact[t - 1]):
            s = t
            while t < T and contact[t] and not dones[t]:
                t += 1
            e = t  # [s, e)
            if e == s:
                # done 이 착지 스텝과 겹침 — 전진 없이는 무한루프
                t += 1
                continue
            if e - s >= MIN_STANCE_STEPS:
                seg = force[s:e]
                out["peak"].append(seg.max())
                out["td_peak"].append(seg[:TOUCHDOWN_STEPS].max())
                head = seg[: LOADRATE_STEPS + 1]
                if len(head) > 1:
                    out["load_rate"].append(np.diff(head).max() / dt)
                out["impulse"].append(seg.sum() * dt)
                out["stance_t"].append((e - s) * dt)
        else:
            t += 1
    return out


def si(a: float, b: float) -> float:
    """Robinson symmetry index [%] — 0 이면 완전 대칭."""
    denom = 0.5 * (a + b)
    return float((a - b) / denom * 100.0) if denom > 1e-9 else float("nan")


def main(path: str):
    npz = np.load(path)
    meta = json.loads(str(npz["meta"]))
    d = {k: npz[k] for k in npz.files if k != "meta"}
    dt = meta["step_dt"]
    T, N = d["gt_leg"].shape

    gt_leg = d["gt_leg"]
    inj = gt_leg >= 0
    ti, ni = np.where(inj)
    ki = gt_leg[ti, ni]
    rep = np.empty(N, dtype=int)
    for n in range(N):
        vals, counts = np.unique(gt_leg[:, n], return_counts=True)
        rep[n] = int(vals[counts.argmax()])

    # 다리별 유효 접촉력: 정상 다리 = 발, 부상 다리 = 부목
    f_ft = np.linalg.norm(d["contact_foot"], axis=-1)     # (T, N, 4)
    f_sp = np.linalg.norm(d["contact_splint"], axis=-1)
    force = f_ft.copy()
    inj_arr = np.zeros((T, N, 4), dtype=bool)
    inj_arr[ti, ni, ki] = True
    force[inj_arr] = f_sp[inj_arr]

    # ── env × leg 별 stance 이벤트 수집 ──
    keys = ("peak", "td_peak", "load_rate", "impulse", "stance_t")
    per = np.full((N, 4, len(keys)), np.nan)   # 이벤트 평균
    n_events = np.zeros((N, 4), dtype=int)
    for n in range(N):
        for k in range(4):
            ev = stance_events(force[:, n, k], d["dones"][:, n], dt)
            n_events[n, k] = len(ev["peak"])
            if ev["peak"]:
                for j, key in enumerate(keys):
                    if ev[key]:
                        per[n, k, j] = float(np.mean(ev[key]))

    print("=" * 84)
    print(f"덤프: {path}  (T={T}, N={N}, dt={dt:.3f}s)")

    # ── [I] 충격량 지표 (그룹별, 부상 다리 vs 나머지 평균) ──
    print("=" * 84)
    print("[I] 충격량 지표 (stance 이벤트 평균)")
    hdr = (f"  {'Group':<8} | {'다리':<6} | {'peak N':>7} | {'착지피크 N':>9} | "
           f"{'하중률 N/s':>9} | {'임펄스 N·s':>9} | {'stance s':>8} | {'착지/s':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr)))
    dur = T * dt
    for g in (-1, 0, 1, 2, 3):
        envs = np.where(rep == g)[0]
        if len(envs) == 0:
            continue
        if g < 0:
            rows = [("전체평균", per[envs].reshape(-1, len(keys)),
                     n_events[envs].mean())]
        else:
            others = [k for k in range(4) if k != g]
            rows = [
                (f"부목({LEGS[g]})", per[envs][:, g], n_events[envs][:, g].mean()),
                ("정상평균", per[envs][:, others].reshape(-1, len(keys)),
                 n_events[envs][:, others].mean()),
            ]
        for name, block, ev_rate in rows:
            m = np.nanmean(block, axis=0)
            print(f"  {GROUP_NAMES[g]:<8} | {name:<6} | {m[0]:>7.1f} | {m[1]:>9.1f} | "
                  f"{m[2]:>9.0f} | {m[3]:>9.2f} | {m[4]:>8.3f} | {ev_rate/dur:>6.2f}")

    # 부목 길이별 충격 (부상 env)
    print("  --- 부목 길이별 (부상 다리 착지 피크 / 하중률) ---")
    gt_L = d["gt_L"]
    L_env = np.array([np.median(gt_L[inj[:, n], n]) if inj[:, n].any() else 0.0
                      for n in range(N)])
    for lo, hi in ((0.33, 0.37), (0.37, 0.41), (0.41, 0.45)):
        sel = [(n, rep[n]) for n in range(N) if rep[n] >= 0 and lo <= L_env[n] < hi]
        if not sel:
            continue
        vals = np.array([per[n, g] for n, g in sel])
        m = np.nanmean(vals, axis=0)
        print(f"  L∈[{lo:.2f},{hi:.2f}): 착지피크 {m[1]:.1f} N, 하중률 {m[2]:.0f} N/s, "
              f"임펄스 {m[3]:.2f} N·s (env {len(sel)}개)")

    # ── [S] 대칭지수 ──
    print("=" * 84)
    print("[S] 대칭지수 SI [%] (0=완전 대칭, |SI|>10% 는 임상적 비대칭 관례)")
    print(f"  {'Group':<8} | {'지표':<8} | {'FL-FR':>7} | {'RL-RR':>7} | {'부상-반대':>8}")
    print("  " + "-" * 52)
    for g in (-1, 0, 1, 2, 3):
        envs = np.where(rep == g)[0]
        if len(envs) == 0:
            continue
        for j, (key, label) in enumerate(
            (("peak", "peak"), ("impulse", "impulse"), ("stance_t", "stance"))
        ):
            fl_fr = np.nanmean([si(per[n, 0, j if key != "impulse" else 3],
                                   per[n, 1, j if key != "impulse" else 3])
                                for n in envs])
            rl_rr = np.nanmean([si(per[n, 2, j if key != "impulse" else 3],
                                   per[n, 3, j if key != "impulse" else 3])
                                for n in envs])
            if g >= 0:
                inj_con = np.nanmean([si(per[n, g, j if key != "impulse" else 3],
                                         per[n, CONTRA[g], j if key != "impulse" else 3])
                                      for n in envs])
                inj_str = f"{inj_con:>8.1f}"
            else:
                inj_str = f"{'—':>8}"
            print(f"  {GROUP_NAMES[g]:<8} | {label:<8} | {fl_fr:>7.1f} | {rl_rr:>7.1f} | {inj_str}")

    # ── [R] 체간 안정성 ──
    print("=" * 84)
    print("[R] 체간 안정성 (그룹별)")
    root = d["root_state"]
    grav = d["projected_gravity"]
    roll = np.arctan2(grav[..., 1], -grav[..., 2])
    pitch = np.arctan2(-grav[..., 0], -grav[..., 2])
    print(f"  {'Group':<8} | {'높이 m':>7} | {'높이 std':>8} | {'수직속도 RMS':>11} | "
          f"{'roll RMS°':>9} | {'pitch RMS°':>10}")
    print("  " + "-" * 66)
    for g in (-1, 0, 1, 2, 3):
        envs = np.where(rep == g)[0]
        if len(envs) == 0:
            continue
        h = root[:, envs, 2]
        vz = root[:, envs, 9]
        print(f"  {GROUP_NAMES[g]:<8} | {h.mean():>7.3f} | {h.std():>8.3f} | "
              f"{np.sqrt((vz**2).mean()):>11.3f} | "
              f"{np.degrees(np.sqrt((roll[:, envs]**2).mean())):>9.2f} | "
              f"{np.degrees(np.sqrt((pitch[:, envs]**2).mean())):>10.2f}")
    print("=" * 84)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(Path(__file__).parent / "dumps" / "p3_aux_balanced.npz"))
