#!/usr/bin/env python3
"""여러 정책의 table_paradigm.csv 를 한 표로 합치고 논문 표 2 (tab:3paradigm) LaTeX 행을 출력한다.

    python merge_paradigm_tables.py \
        logs/unitree_go1_antalgic/P3-final/eval_x0.5_n200_s3000/table_paradigm.csv \
        logs/unitree_go1_antalgic/P2-ft/eval_x0.5_n200_s3000/table_paradigm.csv \
        logs/unitree_go1_antalgic/P2-sym/eval_x0.5_n200_s3000/table_paradigm.csv \
        --out table_3paradigm.csv

Healthy limb 행은 첫 번째 파일의 것을 쓴다 (모든 정책의 Normal 그룹이 같은 환경이라 동일해야 한다).
Isaac 불필요.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("csvs", nargs="+", type=Path, help="play_result.py 가 만든 table_paradigm.csv 들")
p.add_argument("--out", type=Path, default=None, help="합친 CSV 경로 (기본: 첫 파일 옆 table_3paradigm.csv)")
p.add_argument("--bold", type=str, default="Antalgic", help="굵게 표시할 paradigm 행 이름")
args = p.parse_args()

rows, healthy = [], None
for f in args.csvs:
    for r in csv.DictReader(open(f)):
        if r["paradigm"] == "Healthy limb":
            healthy = healthy or r
        else:
            rows.append(r)

out = args.out or args.csvs[0].parent / "table_3paradigm.csv"
keys = sorted({k for r in rows + ([healthy] if healthy else []) for k in r}, key=lambda k: (k != "paradigm", k))
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows + ([healthy] if healthy else []):
        w.writerow(r)
print(f"[INFO] 저장: {out}\n")

def g(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")

def pm(r, k, d=1, bold=False):
    m, s = g(r, k), g(r, k + "_sd")
    txt = f"{m:.{d}f} \\pm {s:.{d}f}"
    return f"$\\mathbf{{{txt}}}$" if bold else f"${txt}$"

print("% ---- tab:3paradigm rows (Tracking error | Peak GRF %BW | GRF reduction | SI_vi | SI_st | dz Aff / Contra) ----")
for r in rows:
    b = r["paradigm"] == args.bold
    name = f"\\textbf{{{r['paradigm']}}}" if b else r["paradigm"]
    te = f"\\textbf{{{g(r,'tracking_err'):.3f}}}" if b else f"{g(r,'tracking_err'):.3f}"
    red = f"$\\mathbf{{{g(r,'grf_red_pct'):+.0f}\\%}}$" if b else f"${g(r,'grf_red_pct'):+.0f}\\%$"
    dz = f"${g(r,'dz_aff_mm'):+.1f} \\pm {g(r,'dz_aff_mm_sd'):.1f}$ / ${g(r,'dz_contra_mm'):+.1f} \\pm {g(r,'dz_contra_mm_sd'):.1f}$"
    print(f"{name} & {te} & {pm(r,'peak_grf_bw',1,b)} & {red}\n"
          f"    & {pm(r,'si_vi_pct',1,b)} & {pm(r,'si_stance_pct',1)} & {dz} \\\\")
if healthy:
    h = healthy
    print("\\midrule")
    print(f"Healthy limb & {g(h,'tracking_err'):.3f} & {g(h,'peak_grf_bw_min'):.0f}--{g(h,'peak_grf_bw_max'):.0f} & --- "
          f"& $\\pm{g(h,'si_vi_pct_absmean'):.1f}$ & $\\pm{g(h,'si_stance_pct_absmean'):.1f}$ "
          f"& ${g(h,'dz_mm_min'):+.1f}$ to ${g(h,'dz_mm_max'):+.1f}$ / --- \\\\")
