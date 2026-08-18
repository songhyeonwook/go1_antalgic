"""rollout_dump.py 가 만든 .npz 에서 phase 3 입력 관측을 부상상태별 CSV 로 정리한다.

policy(=student 입력 51ch) + privileged(=teacher/critic 전용 10ch) 채널을
부상상태(Normal/FL/FR/RL/RR)별로 나눠 내보낸다.

    # 채널별 mean/std/min/max 요약표 1개
    PYTHONPATH= python3 dump_obs_csv.py dumps/p3_splint003_final_balanced.npz

    # 원시 값 전체 — 상태별 CSV 5개 (행 = env_id×step, 열 = 61ch + gt_L/gt_mu/done)
    # env_id, step 으로 시퀀스 복원 가능 (관절/L 예측 모델 학습용)
    PYTHONPATH= python3 dump_obs_csv.py dumps/p3_splint003_final_balanced.npz --raw
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

STATES = ((-1, "normal"), (0, "FL"), (1, "FR"), (2, "RL"), (3, "RR"))
LEGS = ("FL", "FR", "RL", "RR")


def channel_names(meta: dict) -> list[tuple[str, str]]:
    """(group, name) 목록 — meta 의 obs_*_layout 문자열과 동일한 순서."""
    joints = meta["leg_joint_names"]  # per-TYPE 순서 (hip×4, thigh×4, calf×4)
    names = []
    names += [("policy", f"ang_vel_{a}") for a in "xyz"]
    names += [("policy", f"gravity_{a}") for a in "xyz"]
    names += [("policy", c) for c in ("cmd_vx", "cmd_vy", "cmd_wz")]
    names += [("policy", f"jpos_{j.removesuffix('_joint')}") for j in joints]
    names += [("policy", f"jvel_{j.removesuffix('_joint')}") for j in joints]
    names += [("policy", f"act_{j.removesuffix('_joint')}") for j in joints]
    names += [("policy", f"calf_nom_{leg}") for leg in LEGS]
    names += [("policy", "rls_L_norm"), ("policy", "rls_sqrtP_norm")]
    names += [("privileged", f"onehot_{leg}") for leg in LEGS]
    names += [("privileged", "injured_flag")]
    names += [("privileged", "splint_L"), ("privileged", "splint_mu")]
    names += [("privileged", f"lin_vel_{a}") for a in "xyz"]
    return names


def write_summary(out_path, names, flat_obs, flat_leg) -> dict:
    rows = []
    counts = {}
    for ch, (group, name) in enumerate(names):
        row = {"group": group, "idx": ch, "channel": name}
        for code, state in STATES:
            v = flat_obs[flat_leg == code, ch]
            counts[state] = v.size
            row[f"{state}_mean"] = f"{v.mean():.6g}"
            row[f"{state}_std"] = f"{v.std():.6g}"
            row[f"{state}_min"] = f"{v.min():.6g}"
            row[f"{state}_max"] = f"{v.max():.6g}"
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return counts


def write_raw_per_state(npz_path, names, obs, gt_leg, d) -> list[Path]:
    """상태별 원시 CSV — 행 = (env_id, step), 열 = 61ch + gt/done.

    balanced 덤프는 env 별 부상상태가 고정이므로 env_id, step 정렬로
    시퀀스가 그대로 복원된다. done=1 행은 에피소드 경계(리셋 직전).
    """
    T, N, C = obs.shape
    gt_L, gt_mu, dones = d["gt_L"], d["gt_mu"], d["dones"]
    header = (["state", "env_id", "step"]
              + [name for _, name in names]
              + ["gt_L", "gt_mu", "done"])
    out_paths = []
    for code, state in STATES:
        # 상태가 고정인 env 만 선택 (리셋 중 일시 불일치 방지: 최빈값 기준)
        env_state = np.array([
            np.bincount(gt_leg[:, e] + 1, minlength=5).argmax() - 1
            for e in range(N)
        ])
        env_ids = np.where(env_state == code)[0]

        out = npz_path.with_name(f"{npz_path.stem}_raw_{state}.csv")
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for e in env_ids:
                block = np.column_stack([
                    np.full(T, e), np.arange(T), obs[:, e, :],
                    gt_L[:, e], gt_mu[:, e], dones[:, e].astype(int),
                ])
                for r in block:
                    writer.writerow(
                        [state, int(r[0]), int(r[1])]
                        + [f"{v:.6g}" for v in r[2:-1]]
                        + [int(r[-1])]
                    )
        out_paths.append(out)
        print(f"[OK] {out} — env {len(env_ids)}개 × {T} step "
              f"= {len(env_ids) * T} 행")
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="?",
                    default="dumps/p3_splint003_final_balanced.npz")
    ap.add_argument("out", nargs="?", default=None,
                    help="요약 CSV 경로 (기본: <npz>_obs_by_state.csv)")
    ap.add_argument("--raw", action="store_true",
                    help="상태별 원시 값 전체를 CSV 5개로 저장")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    # (T, N, 61)
    obs = np.concatenate([d["obs_policy"], d["obs_privileged"]], axis=-1)
    gt_leg = d["gt_leg"]  # (T, N) — -1=정상, 0..3=FL/FR/RL/RR

    names = channel_names(meta)
    assert len(names) == obs.shape[-1], \
        f"채널 수 불일치: {len(names)} != {obs.shape[-1]}"

    if args.raw:
        write_raw_per_state(npz_path, names, obs, gt_leg, d)
        return

    out_path = (Path(args.out) if args.out
                else npz_path.with_name(npz_path.stem + "_obs_by_state.csv"))
    counts = write_summary(
        out_path, names, obs.reshape(-1, obs.shape[-1]), gt_leg.reshape(-1)
    )
    print(f"[OK] {out_path}")
    print(f"     소스: {npz_path.name} — obs {obs.shape}, "
          "상태별 샘플 수(step×env): "
          + ", ".join(f"{s}={n}" for s, n in counts.items()))


if __name__ == "__main__":
    main()
