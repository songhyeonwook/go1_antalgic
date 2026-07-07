#!/usr/bin/env python3
"""Extract paper-ready biomechanics metrics from a biomech npz (OFFLINE, no GPU).

Reuses biomech_analyze.analyze() for the per-condition aggregation, then adds the
metrics the manuscript needs but biomech_analyze does not print:
  - Symmetry Index SI (eq.7) on affected-limb peak GRF  (the "+138%" headline)
  - GRF reduction %  and  stance-fraction (duty) reduction %
  - per-limb vertical-impulse % change: contralateral / ipsilateral / diagonal
    (paper targets: contra +14..20%, ipsi +10..17%)
  - CoM lateral shift toward the intact side (cm; paper target 1..3 cm)
  - direction-of-change agreement (delegated to biomech_analyze.report)

Usage:
  python3 extract_paper_metrics.py biomech/ucstu.npz --label student
  python3 extract_paper_metrics.py biomech/ucf.npz   --label teacher
"""
import argparse
import numpy as np
from biomech_analyze import analyze, report, REL, LEGS, LEFT_LEGS


def si_eq7(x_healthy, x_affected):
    """Symmetry Index, paper eq.(7): (X_h - X_a)/(0.5(X_h+X_a)) * 100%."""
    return (x_healthy - x_affected) / (0.5 * (x_healthy + x_affected) + 1e-9) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--label", default="policy")
    args = ap.parse_args()

    agg = analyze(args.npz)
    if 0 not in agg:
        print(f"[{args.label}] no Normal baseline group in {args.npz}")
        return
    N = agg[0]

    print("=" * 92)
    print(f"PAPER METRICS — {args.label}   ({args.npz})")
    print("=" * 92)
    print(f"  Normal peak GRF (N)  FL/FR/RL/RR = {np.round(N['peak'], 1)}")
    print(f"  Normal duty          FL/FR/RL/RR = {np.round(N['duty'], 2)}")
    print("-" * 92)
    hdr = (f"  {'Cond':4} {'affGRF':>7} {'GRFred':>7} {'SI(eq7)':>8} "
           f"{'duty':>5} {'dutyRed':>8} {'contraI':>8} {'ipsiI':>7} {'diagI':>7} {'CoM(cm)':>8}")
    print(hdr)

    si_l, red_l, dred_l, ic_l, ii_l, idg_l, com_l = ([] for _ in range(7))
    for cond in [1, 2, 3, 4]:
        if cond not in agg:
            continue
        a = agg[cond]
        leg = cond - 1
        r = REL[leg]
        red = (1 - a["peak"][leg] / max(N["peak"][leg], 1e-6)) * 100
        si_v = si_eq7(N["peak"][leg], a["peak"][leg])
        dred = (1 - a["duty"][leg] / max(N["duty"][leg], 1e-6)) * 100

        def imp_pct(k):
            return (a["impulse"][k] / max(N["impulse"][k], 1e-9) - 1) * 100

        ic, ii_, idg = imp_pct(r["contra"]), imp_pct(r["ipsi"]), imp_pct(r["diag"])
        # CoM lateral shift toward intact side (m -> cm). left injury (leg in LEFT) -> intact is right (-y)
        side = +1.0 if leg in LEFT_LEGS else -1.0
        com_cm = (a["base_y"] - N["base_y"]) * (-side) * 100.0

        for lst, v in ((si_l, si_v), (red_l, red), (dred_l, dred),
                       (ic_l, ic), (ii_l, ii_), (idg_l, idg), (com_l, com_cm)):
            lst.append(v)
        print(f"  {LEGS[leg]:4} {a['peak'][leg]:7.1f} {red:6.0f}% {si_v:7.0f}% "
              f"{a['duty'][leg]:5.2f} {dred:7.0f}% {ic:+7.0f}% {ii_:+6.0f}% {idg:+6.0f}% {com_cm:+7.1f}")

    print("-" * 92)
    print(f"  MEAN(4 conds):  GRF reduction {np.mean(red_l):.0f}%   "
          f"SI(eq.7) {np.mean(si_l):.0f}%   stance/duty reduction {np.mean(dred_l):.0f}%")
    print(f"    vertical impulse  contralateral {np.mean(ic_l):+.0f}%  (target +14..20%)")
    print(f"                      ipsilateral   {np.mean(ii_l):+.0f}%  (target +10..17%)")
    print(f"                      diagonal      {np.mean(idg_l):+.0f}%")
    print(f"    CoM lateral shift to intact side {np.mean(com_l):+.1f} cm  (target +1..3 cm)")
    print("-" * 92)
    # direction-of-change agreement (reuse biomech_analyze)
    report(args.npz, args.label)


if __name__ == "__main__":
    main()
