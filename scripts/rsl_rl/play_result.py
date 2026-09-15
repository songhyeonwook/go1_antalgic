#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보행 지표 + (phase 3) 부목 길이·속도 추정 결과 추출. phase 1/2/3 공통.
# 무작위 명령 (yaml 범위)
play_result.py --phase_config_path configs/phase/3/phase3.yaml \
  --checkpoint logs/unitree_go1_antalgic/P3-final/model_3999.pt --num_envs 40 --steps 1500 --headless

play_result.py --phase_config_path configs/phase/3/phase3.yaml \
  --checkpoint logs/unitree_go1_antalgic/P3-final/model_3999.pt --num_envs 40 --steps 1500 --headless \
  --fixed_x 0.5 --out_dir logs/unitree_go1_antalgic/P3-final/eval_fixed_x0.5



접지 판정은 Isaac Lab ContactSensor 의 판정을 그대로 쓴다 (current_contact_time > 0, 즉
||F|| > ContactSensorCfg.force_threshold — 학습 보상(feet_air_time)과 같은 기준, 이 환경은 1 N).
GRF/충격량은 수직력 |Fz| 로, 리셋 후 settle_s 이후의 완결된 stance(리셋 경계에 잘리지 않고
min_stance_s 이상 지속) 단위로 계산한다. --fixed_x 0.5 를 주면 전진 명령을 고정한다.

출력 (기본: 체크포인트 폴더):
  gait_analysis.png         조건별 다리별 duty factor / 접촉력
  grf_impulse_analysis.png  다리별 peak GRF / 부상 다리 GRF 감소율·SI / 역할별 충격량 변화율
  estimation_analysis.png   phase 3 만: L̂ 수렴 곡선, L̂ vs GT, v̂ 오차
  metrics_summary.csv       조건별 수치 요약
  table_conditions.csv      논문 표 1: 조건별 survival / 전진 속도 / 부상 다리 GRF·감소율·duty / SI_GRF + Mean (injured)
  table_paradigm.csv        논문 표 2: Antalgic 한 줄(부상 4조건 평균±s.d.) + Healthy limb 범위 한 줄
  rollout_raw.npz           --dump_npz 시 원자료
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
parser.add_argument("--out_dir", type=str, default=None, help="결과 저장 폴더 (기본: 체크포인트 폴더)")
parser.add_argument("--settle_s", type=float, default=2.0, help="GRF/Δz 통계에서 제외할 리셋 직후 과도 구간 [s]")
parser.add_argument("--min_stance_s", type=float, default=0.06, help="이보다 짧은 접지는 채터링으로 버린다 [s]")
parser.add_argument("--fixed_x", type=float, default=None, help="전진 명령 고정 [m/s] (좌우 명령은 0)")
parser.add_argument("--fixed_yaw", type=float, default=None, help="yaw 명령 고정 [rad/s]")
parser.add_argument("--dump_npz", action="store_true", help="원자료(force, fz, contact, leg, t_reset, base_z, mg …)를 out_dir/rollout_raw.npz 로 저장 (오프라인 분석용)")
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
    if not getattr(contacts.cfg, "track_air_time", False):
        raise RuntimeError("ContactSensor.track_air_time 가 꺼져 있어 센서 접지 판정을 쓸 수 없습니다.")
    log(f"[eval] Contact rule   : ContactSensor (||F|| > force_threshold {float(contacts.cfg.force_threshold):g} N)")
    foot_b = torch.tensor([names.index(f"{leg}_foot") for leg in LEGS], device=base.device)
    splint_b = torch.tensor([names.index(f"{leg}_splint") for leg in LEGS], device=base.device)
    robot = base.scene["robot"]
    try:  # env 별 체중 mg [N] (startup 질량 랜덤화 반영) — %BW 정규화용
        mg = robot.root_physx_view.get_masses().sum(dim=-1).to(base.device).float() * 9.81
    except Exception as exc:  # pragma: no cover
        log(f"[WARN] physx 질량을 읽지 못해 default_mass 를 씁니다: {exc}")
        mg = robot.data.default_mass.sum(dim=-1).to(base.device).float() * 9.81
    log(f"[eval] Body weight    : {mg.mean().item():.1f} N (env 평균, range {mg.min().item():.1f}–{mg.max().item():.1f})")
    has_aux = hasattr(policy_module, "aux_inference")   # phase 3 student 만
    recurrent = getattr(policy_module, "is_recurrent", False)
    dt = float(base.step_dt)

    rec = {
        "force": np.zeros((T, N, 4), np.float32),   # 다리별 유효 접촉력 ||F|| (부상 다리 = 부목 끝단)
        "contact": np.zeros((T, N, 4), bool),        # 다리별 접지 (ContactSensor 판정)
        "fz": np.zeros((T, N, 4), np.float32),      # 다리별 수직력 |Fz| (GRF/충격량용, 부상 다리 = 부목 끝단)
        "leg": np.zeros((T, N), np.int8),            # -1 정상, 0..3 부상 다리
        "L": np.zeros((T, N), np.float32),           # GT 부목 길이 (정상 = 0)
        "t_reset": np.zeros((T, N), np.float32),     # 정책 호출 시점의 리셋 후 경과 [s]
        "verr": np.zeros((T, N), np.float32),        # |v_cmd,xy - v_xy|
        "werr": np.zeros((T, N), np.float32),        # |w_cmd,z - w_z|
        "fall": np.zeros((T, N), bool),              # time-out 이 아닌 종료
        "L_hat": np.zeros((T, N), np.float32),       # phase 3: 부목 길이 추정 [m]
        "v_err": np.zeros((T, N), np.float32),       # phase 3: |v̂ - v_gt|
        "base_z": np.zeros((T, N), np.float32),            # 몸통(base) 높이 [m] (평지 z=0 기준)
        "vx": np.zeros((T, N), np.float32),                # 전진 속도 v_x (base 프레임) [m/s]
        "cmd_vx": np.zeros((T, N), np.float32),            # 전진 명령 v_x* [m/s]
        "timeout": np.zeros((T, N), bool),                 # time-out 종료 (생존)
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
            mag = f.norm(dim=-1)
            feet = mag[:, foot_b].clone()
            inj = leg_pre >= 0
            if bool(inj.any()):
                rows = torch.nonzero(inj).squeeze(-1)
                feet[rows, leg_pre[rows]] = mag[:, splint_b][rows, leg_pre[rows]]
            rec["force"][t] = feet.cpu().numpy()
            # 접지: 센서 판정 (||F|| > cfg.force_threshold 의 누적 상태)
            c_all = contacts.data.current_contact_time > 0.0
            c_feet = c_all[:, foot_b].clone()
            if bool(inj.any()):
                c_feet[rows, leg_pre[rows]] = c_all[:, splint_b][rows, leg_pre[rows]]
            rec["contact"][t] = c_feet.cpu().numpy()
            fz_all = f[..., 2].abs()
            feet_z = fz_all[:, foot_b].clone()
            if bool(inj.any()):
                feet_z[rows, leg_pre[rows]] = fz_all[:, splint_b][rows, leg_pre[rows]]
            rec["fz"][t] = feet_z.cpu().numpy()

            rec["base_z"][t] = robot.data.root_pos_w[:, 2].cpu().numpy()

            cmd = base.command_manager.get_command("base_velocity")
            rec["verr"][t] = (cmd[:, :2] - robot.data.root_lin_vel_b[:, :2]).norm(dim=-1).cpu().numpy()
            rec["werr"][t] = (cmd[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs().cpu().numpy()
            rec["fall"][t] = (done_mask & ~time_outs).cpu().numpy()
            rec["timeout"][t] = (done_mask & time_outs).cpu().numpy()
            rec["vx"][t] = robot.data.root_lin_vel_b[:, 0].cpu().numpy()
            rec["cmd_vx"][t] = cmd[:, 0].cpu().numpy()

            since_reset += dt
            since_reset[done_mask] = 0.0
            if t % 300 == 0:
                log(f"  step {t}/{T}", flush=True)
    rec["dt"] = dt
    rec["has_aux"] = has_aux
    rec["mg"] = mg.cpu().numpy()                          # (N,) [N]
    rec["contact_force_threshold"] = float(contacts.cfg.force_threshold)
    return rec


# 부상 다리 k 에 대한 나머지 다리 역할 (LEGS 순서 FL FR RL RR)
#   contra = 같은 girdle 반대쪽, ipsi = 같은 쪽 다른 girdle, diag = 대각
ROLES = {0: dict(contra=1, ipsi=2, diag=3), 1: dict(contra=0, ipsi=3, diag=2),
         2: dict(contra=3, ipsi=0, diag=1), 3: dict(contra=2, ipsi=1, diag=0)}
ROLE_NAMES = ("affected", "contra", "ipsi", "diag")


def _contact_runs(mask: np.ndarray):
    """True 가 연속된 구간 [start, end) 목록."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def stance_stats(rec, settle_s: float, min_stance_s: float):
    """완결된 stance 단위 GRF 지표. 반환: {group: {"peak","impulse","dur": (4,) 평균, "n": (4,) stance 수}}.

    에피소드 경계(t_reset == 0)로 나누고, 리셋 후 settle_s 는 버리며, 경계에 잘린 stance 와
    min_stance_s 미만 접지는 제외한다. 조건은 에피소드 시작 스텝의 _peg_leg_index 로 정한다.
    """
    fz, t_reset, leg = rec["fz"], rec["t_reset"], rec["leg"]
    base_z, mg = rec["base_z"], rec["mg"]
    T, N, _ = fz.shape
    dt = rec["dt"]
    settle = int(round(settle_s / dt))
    min_len = max(1, int(round(min_stance_s / dt)))
    KEYS = ("peak", "peak_bw", "impulse", "dur", "dz")
    acc = {g: {k: {key: [] for key in KEYS} for k in range(4)} for g in range(5)}
    for n in range(N):
        starts = np.flatnonzero(t_reset[:, n] == 0.0)
        if starts.size == 0 or starts[0] != 0:
            starts = np.concatenate(([0], starts))
        ends = np.append(starts[1:], T)
        for s0, s1 in zip(starts, ends):
            g = int(leg[s0, n]) + 1
            lo = s0 + settle
            if lo >= s1:
                continue
            z_ref = float(base_z[lo:s1, n].mean())          # 에피소드(정착 후) 평균 몸통 높이
            for k in range(4):
                m = rec["contact"][lo:s1, n, k]
                for a, b in _contact_runs(m):
                    if a == 0 or b == len(m) or b - a < min_len:
                        continue
                    seg = fz[lo + a:lo + b, n, k]
                    acc[g][k]["peak"].append(float(seg.max()))
                    acc[g][k]["peak_bw"].append(float(seg.max() / mg[n] * 100.0))
                    acc[g][k]["impulse"].append(float(seg.sum() * dt))
                    acc[g][k]["dur"].append((b - a) * dt)
                    # 체간 상하 이동: 이 다리 stance 중 몸통 높이 − 에피소드 평균 (음수 = 내려앉음)
                    acc[g][k]["dz"].append(float(base_z[lo + a:lo + b, n].mean() - z_ref))
    out = {}
    for g in range(5):
        if not any(acc[g][k]["peak"] for k in range(4)):
            continue
        out[g] = {key: np.array([_nanmean(acc[g][k][key]) for k in range(4)]) for key in KEYS}
        out[g]["n"] = np.array([len(acc[g][k]["peak"]) for k in range(4)])
    return out


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _nanstd(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std()) if x.size > 1 else float("nan")


def summarize(rec, settle_s: float = 2.0, min_stance_s: float = 0.06):
    """조건(그룹)별 지표. 그룹 = 그 스텝의 _peg_leg_index + 1 (0 Normal … 4 RR)."""
    cond = rec["leg"].astype(int) + 1
    contact = rec["contact"]
    dt = rec["dt"]
    settled = rec["t_reset"] >= settle_s
    st = stance_stats(rec, settle_s, min_stance_s)
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
        ms = m & settled
        n_to, n_fall = int(rec["timeout"][m].sum()), int(rec["fall"][m].sum())
        row["episodes"] = n_to + n_fall
        row["survival_pct"] = float(100.0 * n_to / (n_to + n_fall)) if (n_to + n_fall) else float("nan")
        fwd = ms & (rec["cmd_vx"] > 0.1)
        row["fwd_speed"] = _nanmean(rec["vx"][fwd])
        row["fwd_cmd"] = _nanmean(rec["cmd_vx"][fwd])
        row["fwd_samples"] = int(fwd.sum())
        # GRF / 충격량 (|Fz|, 완결 stance 단위). load = 시간평균 수직 하중 (swing 포함, 길이 무관)
        sg = st.get(g)
        for k, leg in enumerate(LEGS):
            row[f"peak_grf_{leg}"] = float(sg["peak"][k]) if sg else float("nan")
            row[f"peak_grf_bw_{leg}"] = float(sg["peak_bw"][k]) if sg else float("nan")
            row[f"dz_{leg}_mm"] = float(sg["dz"][k]) * 1000.0 if sg else float("nan")
            row[f"impulse_{leg}"] = float(sg["impulse"][k]) if sg else float("nan")
            row[f"stance_dur_{leg}"] = float(sg["dur"][k]) if sg else float("nan")
            row[f"n_stance_{leg}"] = int(sg["n"][k]) if sg else 0
            row[f"load_{leg}"] = _nanmean(rec["fz"][..., k][ms])
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

    # Normal 대비 변화율 (부상 그룹만)
    ref = next((r for r in rows if r["group"] == CONDITION_LABELS[0]), None)
    if ref is not None:
        for r in rows:
            g = CONDITION_LABELS.index(r["group"])
            if g == 0:
                continue
            k = g - 1
            def pct(key, j):
                base = ref[f"{key}_{LEGS[j]}"]
                return float((r[f"{key}_{LEGS[j]}"] / base - 1.0) * 100.0) if base and np.isfinite(base) else float("nan")

            aff, con = LEGS[k], LEGS[ROLES[k]["contra"]]
            r["grf_red_pct"] = -pct("peak_grf", k)                       # 부상 다리 peak GRF 감소율 (+ = 감소)
            pa, pc = r[f"peak_grf_{aff}"], r[f"peak_grf_{con}"]
            r["si_peak_grf_pct"] = float(100.0 * (pc - pa) / (0.5 * (pc + pa))) if (pc + pa) > 0 else float("nan")
            r["duty_red_pct"] = -pct("duty", k)
            da, dc = r[f"stance_dur_{aff}"], r[f"stance_dur_{con}"]
            r["si_stance_pct"] = float(100.0 * (dc - da) / (0.5 * (dc + da))) if (dc + da) > 0 else float("nan")
            r["peak_grf_aff_n"] = pa
            r["peak_grf_aff_bw"] = r[f"peak_grf_bw_{aff}"]
            r["dz_aff_mm"] = r[f"dz_{aff}_mm"]
            r["dz_contra_mm"] = r[f"dz_{con}_mm"]
            roles = {"affected": k, **ROLES[k]}
            for name, j in roles.items():
                r[f"impulse_chg_{name}_pct"] = pct("impulse", j)      # stance 당 ∫Fz dt 변화율
                r[f"load_chg_{name}_pct"] = pct("load", j)            # 시간평균 수직 하중 변화율 (논문 스크립트 방식)
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
    table("Peak GRF (N, |Fz|, 완결 stance 최대값 평균)", "peak_grf", ".2f")
    table("Vertical Impulse (N·s, stance 당 ∫|Fz| dt)", "impulse", ".3f")
    table("Stance Duration (s)", "stance_dur", ".3f")

    inj_rows = [r for r in rows if "grf_red_pct" in r]
    if inj_rows:
        log("\n" + "=" * 84)
        log("부상 다리 GRF (Normal 대비)")
        log("=" * 84)
        log(f"{'Group':<8} | {'peak GRF 감소':>12} | {'SI(eq.7)':>9} | {'duty 감소':>9}")
        log("-" * 84)
        for r in inj_rows:
            log(f"{r['group']:<8} | {r['grf_red_pct']:>+11.1f}% | {r['si_peak_grf_pct']:>+8.0f}% | {r['duty_red_pct']:>+8.1f}%")
        log("-" * 84)
        log("SI = 100·(X_contra − X_aff)/(0.5·(X_contra + X_aff)), 양수 = 부상 다리가 작음")

        for key, title in (("impulse", "Stance 당 수직 충격량 변화율 (Normal 대비, %)"),
                           ("load", "시간평균 수직 하중 변화율 (Normal 대비, %; 논문 스크립트 방식)")):
            log("\n" + "=" * 84)
            log(title)
            log("=" * 84)
            log(f"{'Group':<8} | " + " | ".join(f"{n:>9}" for n in ROLE_NAMES))
            log("-" * 84)
            for r in inj_rows:
                log(f"{r['group']:<8} | " + " | ".join(f"{r[f'{key}_chg_{n}_pct']:>+8.1f}%" for n in ROLE_NAMES))
            log("-" * 84)
        log("논문 목표: contra +14..20%, ipsi +10..17%, affected 감소")

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


def table_conditions(rows) -> list[dict]:
    """논문 표 1 (조건별). 열: survival_pct, fwd_speed, injured_grf_n, grf_red_pct, injured_duty, si_grf_pct.

    Healthy 행: duty 는 네 다리 평균, SI_GRF 는 좌우 동명 다리 쌍(FL-FR, RL-RR) |SI| 평균.
    Mean (injured) 행: 부상 4조건의 단순 평균 (survival, fwd_speed 포함).
    """
    out = []
    inj = []
    for r in rows:
        g = CONDITION_LABELS.index(r["group"])
        if g == 0:
            def si(a, b):
                xa, xb = r[f"peak_grf_{a}"], r[f"peak_grf_{b}"]
                return abs(100.0 * (xb - xa) / (0.5 * (xa + xb))) if (xa + xb) > 0 else float("nan")
            out.append({"condition": "Healthy", "survival_pct": r["survival_pct"], "fwd_speed": r["fwd_speed"],
                        "fwd_cmd": r["fwd_cmd"], "injured_grf_n": float("nan"), "grf_red_pct": float("nan"),
                        "injured_duty": float(np.mean([r[f"duty_{l}"] for l in LEGS])),
                        "si_grf_pct": float(np.nanmean([si("FL", "FR"), si("RL", "RR")])),
                        "episodes": r["episodes"]})
        else:
            aff = LEGS[g - 1]
            d = {"condition": f"{r['group']} injury", "survival_pct": r["survival_pct"], "fwd_speed": r["fwd_speed"],
                 "fwd_cmd": r["fwd_cmd"], "injured_grf_n": r[f"peak_grf_{aff}"], "grf_red_pct": r["grf_red_pct"],
                 "injured_duty": r[f"duty_{aff}"], "si_grf_pct": r["si_peak_grf_pct"], "episodes": r["episodes"]}
            out.append(d)
            inj.append(d)
    if inj:
        mean = {"condition": "Mean (injured)", "episodes": int(sum(d["episodes"] for d in inj))}
        for k in ("survival_pct", "fwd_speed", "fwd_cmd", "injured_grf_n", "grf_red_pct", "injured_duty", "si_grf_pct"):
            mean[k] = _nanmean([d[k] for d in inj])
        out.append(mean)
    return out


def table_paradigm(rows) -> list[dict]:
    """논문 표 2 (paradigm). Antalgic 행: 부상 4조건의 부상 다리 지표 평균 ± 조건 간 s.d.
    Healthy limb 행: Normal 그룹 다리별 범위; SI 는 좌우 동명 다리 쌍 기준 최대 |SI|.
      tracking_err            |v_cmd − v|_xy [m/s]
      peak_grf_n / peak_grf_bw 부상 다리 per-stance peak |Fz| [N] / [%BW]
      grf_red_pct             Normal 대비 감소율 [%]
      si_grf_pct / si_stance_pct  peak GRF / stance 시간 대칭지수 (대측 − 부상)/평균 [%]
      dz_aff_mm / dz_contra_mm    부상측/대측 stance 중 몸통 높이 − 에피소드 평균 [mm]
    """
    inj = [r for r in rows if "grf_red_pct" in r]
    ref = next((r for r in rows if r["group"] == CONDITION_LABELS[0]), None)
    out = []
    if inj:
        keys = {"tracking_err": "vel_err_xy", "peak_grf_n": "peak_grf_aff_n", "peak_grf_bw": "peak_grf_aff_bw",
                "grf_red_pct": "grf_red_pct", "si_grf_pct": "si_peak_grf_pct", "si_stance_pct": "si_stance_pct",
                "dz_aff_mm": "dz_aff_mm", "dz_contra_mm": "dz_contra_mm"}
        row = {"paradigm": "Antalgic", "n_conditions": len(inj)}
        for out_k, in_k in keys.items():
            v = np.array([r[in_k] for r in inj], float)
            row[out_k] = _nanmean(v)
            row[out_k + "_sd"] = _nanstd(v)
        out.append(row)
    if ref is not None:
        def rng(vals):
            v = np.array(vals, float); v = v[np.isfinite(v)]
            return (float(v.min()), float(v.max())) if v.size else (float("nan"), float("nan"))
        def si(key, a, b):
            xa, xb = ref[f"{key}_{a}"], ref[f"{key}_{b}"]
            return float(100.0 * (xb - xa) / (0.5 * (xa + xb))) if (xa + xb) > 0 else float("nan")
        pk = rng([ref[f"peak_grf_{l}"] for l in LEGS]); pb = rng([ref[f"peak_grf_bw_{l}"] for l in LEGS])
        dz = rng([ref[f"dz_{l}_mm"] for l in LEGS])
        si_g = [si("peak_grf", "FL", "FR"), si("peak_grf", "RL", "RR")]
        si_s = [si("stance_dur", "FL", "FR"), si("stance_dur", "RL", "RR")]
        out.append({"paradigm": "Healthy limb", "n_conditions": 1, "tracking_err": ref["vel_err_xy"],
                    "peak_grf_n_min": pk[0], "peak_grf_n_max": pk[1], "peak_grf_bw_min": pb[0], "peak_grf_bw_max": pb[1],
                    "si_grf_pct_absmax": float(np.nanmax(np.abs(si_g))), "si_grf_pct_FLFR": si_g[0], "si_grf_pct_RLRR": si_g[1],
                    "si_stance_pct_absmax": float(np.nanmax(np.abs(si_s))), "si_stance_pct_FLFR": si_s[0], "si_stance_pct_RLRR": si_s[1],
                    "dz_mm_min": dz[0], "dz_mm_max": dz[1]})
    return out


def print_tables_paper(t1: list[dict], t2: list[dict], fixed_x: float | None) -> None:
    log("\n" + "=" * 96)
    cmd_note = f"전진 명령 고정 {fixed_x} m/s" if fixed_x is not None else "전진 명령 샘플(v_x* > 0.1) 평균"
    log(f"Table 1 — per condition (Survival = time-out 종료 / 전체 종료; Fwd. speed = {cmd_note})")
    log("=" * 96)
    log(f"{'Condition':<15} | {'Survival':>8} | {'Fwd speed':>9} | {'(cmd)':>6} | {'Inj GRF N':>9} | {'GRF red':>8} | {'Inj duty':>8} | {'SI_GRF %':>8} | {'episodes':>8}")
    log("-" * 96)
    for r in t1:
        grf = f"{r['injured_grf_n']:.1f}" if np.isfinite(r["injured_grf_n"]) else "—"
        red = f"{r['grf_red_pct']:.0f}%" if np.isfinite(r["grf_red_pct"]) else "—"
        log(f"{r['condition']:<15} | {r['survival_pct']:>7.1f}% | {r['fwd_speed']:>9.3f} | {r['fwd_cmd']:>6.2f} | {grf:>9} | {red:>8} | "
            f"{r['injured_duty']:>8.2f} | {r['si_grf_pct']:>8.1f} | {r['episodes']:>8}")
    log("-" * 96)

    log("\n" + "=" * 96)
    log("Table 2 — paradigm (Antalgic: 부상 4조건 평균 ± 조건 간 s.d. / Healthy limb: Normal 그룹 다리별 범위)")
    log("=" * 96)
    log(f"{'Paradigm':<13} | {'track err':>9} | {'peak N':>11} | {'peak %BW':>11} | {'GRF red':>8} | {'SI_GRF':>10} | {'SI_st':>10} | {'dz aff':>10} | {'dz contra':>10}")
    log("-" * 96)
    for r in t2:
        if r["paradigm"] == "Antalgic":
            f = lambda k, d=1: f"{r[k]:.{d}f}±{r[k + '_sd']:.{d}f}"
            log(f"{r['paradigm']:<13} | {r['tracking_err']:>9.3f} | {f('peak_grf_n'):>11} | {f('peak_grf_bw'):>11} | {r['grf_red_pct']:>+7.0f}% | "
                f"{f('si_grf_pct'):>10} | {f('si_stance_pct'):>10} | {f('dz_aff_mm'):>10} | {f('dz_contra_mm'):>10}")
        else:
            log(f"{r['paradigm']:<13} | {r['tracking_err']:>9.3f} | {r['peak_grf_n_min']:>4.0f}–{r['peak_grf_n_max']:<6.0f} | "
                f"{r['peak_grf_bw_min']:>4.0f}–{r['peak_grf_bw_max']:<6.0f} | {'—':>8} | ±{r['si_grf_pct_absmax']:<9.1f} | "
                f"±{r['si_stance_pct_absmax']:<9.1f} | {r['dz_mm_min']:>+4.1f}…{r['dz_mm_max']:<+5.1f} | {'—':>10}")
    log("-" * 96)


def write_csv(path: Path, rows: list[dict], first: str) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != first, k))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})


def plot_grf(rows, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    x = np.arange(4)
    width = 0.15

    ax = axes[0]   # 다리별 peak GRF
    for r in rows:
        g = CONDITION_LABELS.index(r["group"])
        vals = [r[f"peak_grf_{leg}"] for leg in LEGS]
        bars = ax.bar(x + (g - 2) * width, vals, width, label=r["group"], alpha=0.85, color=COLORS[g])
        if g > 0:
            bars[g - 1].set_edgecolor("k")
            bars[g - 1].set_linewidth(1.8)
            bars[g - 1].set_hatch("xx")
    ax.set_xticks(x)
    ax.set_xticklabels(LEGS)
    ax.set_ylabel("Peak vertical GRF [N]")
    ax.set_title("Peak GRF per leg (hatched = injured leg)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    inj = [r for r in rows if "grf_red_pct" in r]
    ax = axes[1]   # 부상 다리 GRF 감소율 + SI
    if inj:
        xi = np.arange(len(inj))
        w = 0.38
        ax.bar(xi - w / 2, [r["grf_red_pct"] for r in inj], w, label="peak GRF reduction",
               color=[COLORS[CONDITION_LABELS.index(r["group"])] for r in inj], alpha=0.9)
        ax.bar(xi + w / 2, [r["si_peak_grf_pct"] for r in inj], w, label="SI eq.7 (vs contra)",
               color=[COLORS[CONDITION_LABELS.index(r["group"])] for r in inj], alpha=0.45, hatch="//")
        ax.set_xticks(xi)
        ax.set_xticklabels([r["group"] for r in inj])
        ax.axhline(0, color="k", lw=0.8)
        ax.legend(fontsize=8)
    ax.set_ylabel("[%]")
    ax.set_title("Injured-leg peak GRF vs Normal")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[2]   # 역할별 충격량 변화율
    if inj:
        xr = np.arange(len(ROLE_NAMES))
        w = 0.8 / len(inj)
        for i, r in enumerate(inj):
            g = CONDITION_LABELS.index(r["group"])
            ax.bar(xr + (i - (len(inj) - 1) / 2) * w, [r[f"impulse_chg_{n}_pct"] for n in ROLE_NAMES],
                   w, label=r["group"], color=COLORS[g], alpha=0.85)
        ax.axhspan(14, 20, xmin=1 / 4 + 0.02, xmax=2 / 4 - 0.02, color="green", alpha=0.12)
        ax.axhspan(10, 17, xmin=2 / 4 + 0.02, xmax=3 / 4 - 0.02, color="green", alpha=0.12,
                   label="paper target (contra +14..20, ipsi +10..17)")
        ax.set_xticks(xr)
        ax.set_xticklabels(ROLE_NAMES)
        ax.axhline(0, color="k", lw=0.8)
        ax.legend(fontsize=8)
    ax.set_ylabel("Δ impulse per stance vs Normal [%]")
    ax.set_title("Vertical impulse redistribution by leg role")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

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
    # 명령 고정: 샘플 범위를 한 점으로 좁힌다 (리셋마다 같은 값이 뽑힌다)
    ranges = env_cfg.commands.base_velocity.ranges
    if args.fixed_x is not None:
        ranges.lin_vel_x = (args.fixed_x, args.fixed_x)
        ranges.lin_vel_y = (0.0, 0.0)
    if args.fixed_yaw is not None:
        ranges.ang_vel_z = (args.fixed_yaw, args.fixed_yaw)
    agent_cfg = build_agent_cfg(config, seed=seed, device=device)
    print_header(config, checkpoint, device, seed, num_envs, eval_mode, env_cfg)
    log(f"[eval] Steps          : {args.steps}")
    log(f"[eval] Settle / stance: {args.settle_s} s / min {args.min_stance_s} s (GRF·충격량·Δz 통계)")

    env = wrap_rsl(make_gym_env(config, env_cfg), agent_cfg)
    base = env.unwrapped
    print_conditions(base)
    _, policy_fn, policy_module = make_policy(env, agent_cfg, config, checkpoint, device)
    log(f"[eval] Aux head       : {'있음 — L̂ / v̂ 로깅' if hasattr(policy_module, 'aux_inference') else '없음'}")

    rec = collect(env, base, policy_fn, policy_module, args.steps)
    if args.dump_npz:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "rollout_raw.npz", **{k: v for k, v in rec.items() if isinstance(v, np.ndarray)},
                            dt=rec["dt"], contact_force_threshold=rec["contact_force_threshold"])
        log(f"[INFO] 저장: {out_dir / 'rollout_raw.npz'}")
    rows = summarize(rec, settle_s=args.settle_s, min_stance_s=args.min_stance_s)
    print_tables(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    gait_png = out_dir / "gait_analysis.png"
    plot_gait(rows, gait_png)
    log(f"\n[INFO] 저장: {gait_png}")
    grf_png = out_dir / "grf_impulse_analysis.png"
    plot_grf(rows, grf_png)
    log(f"[INFO] 저장: {grf_png}")
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
    t1, t2 = table_conditions(rows), table_paradigm(rows)
    print_tables_paper(t1, t2, args.fixed_x)
    write_csv(out_dir / "table_conditions.csv", t1, "condition")
    write_csv(out_dir / "table_paradigm.csv", t2, "paradigm")
    log(f"[INFO] 저장: {out_dir / 'table_conditions.csv'}, {out_dir / 'table_paradigm.csv'}")
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