#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보행 지표 + (phase 3) 부목 길이·속도 추정 결과 추출. phase 1/2/3 공통.

    python eval.py --phase_config_path configs/phase/2/phase2_at.yaml \
        --checkpoint <model.pt> --num_envs 40 --steps 1500 --headless

조건 배정은 env_fixed 블록이다 (balanced + 40 env: env 0-7 Normal, 8-15 FL, 16-23 FR,
24-31 RL, 32-39 RR). 지표는 각 스텝의 실제 _peg_leg_index 로 그룹을 나누므로 배정
방식과 무관하게 맞는다. 부상 다리의 '접지'는 발이 아니라 부목 끝단({leg}_splint) 접촉력이다.

학습 코드와의 대응:
  - 접촉력은 기본 |Fz| (학습 보상 penalty_pain / duty 관련 항이 쓰는 정의, 평지 기준).
    ||F|| 를 보려면 --force norm.
  - 부상 다리 통증 하중 F_pain = Fz_foot + Fz_calf + η·Fz_splint 와 C_pain 은 학습의
    mdp.penalty_pain 함수와 reward yaml 파라미터를 그대로 호출해 계산한다.
  - duty factor 는 임계값(기본 5 N)에 민감하다 (특히 부목 끝단: 가벼운 접촉이 많음). 임계값과
    무관한 하중 분담률(시간평균 |Fz| 비율)을 함께 낸다.
  - 낙상은 env 가 돌려주는 terminated (grace period 반영, time-out 과 겹쳐도 낙상) 이다.
  - 상태 기반 지표(duty/힘/하중/추적/통증)는 리셋이 일어난 스텝과 리셋 직후 --warmup_s 구간을
    제외한다. 낙상 수·에피소드 수·L̂ 전체 지표는 전 구간을 쓴다.
  - 추적 오차의 명령은 정책이 그 스텝에 본 명령(step 이전 값)이다. step 안에서 명령이
    재샘플되므로 step 이후에 읽으면 리셋/재샘플 스텝에서 짝이 어긋난다.
  - 부목 길이·속도 헤드는 정책의 norm_* 버퍼로 역정규화된 물리 단위(m, m/s)로 비교한다.
    z-공간 MSE 도 함께 내어 학습 TB 의 Loss/splint_length, Loss/base_lin_vel 과 직접 비교한다.

출력 (기본: 체크포인트 폴더):
  gait_analysis.png         조건별 다리별 duty factor / 접촉력
  estimation_analysis.png   phase 3 만: L̂ 수렴 곡선, L̂ vs GT, v̂ 오차
  metrics_summary.csv       조건별 수치 요약
  raw_records.npz           --dump 시 스텝×env 원자료 (사후 분석용)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path
import logging
from datetime import datetime
from utils.config_builder import read_yaml
from utils.logger import create_logger, redirect_python_streams

SCRIPT_DIR = Path(__file__).resolve().parent   # scripts/rsl_rl

from isaaclab.app import AppLauncher

from utils.eval_common import (
    CONDITION_LABELS, LEGS, add_common_args, build_agent_cfg, build_env_cfg,
    check_splint_length_arg, load_config, log, make_gym_env, make_policy, peg_leg_enabled,
    print_conditions, print_header, resolve_checkpoint, resolve_eval_mode, wrap_rsl,
)

parser = argparse.ArgumentParser(description="보행 지표 + 부목 추정 결과 추출 (RSL-RL)")
add_common_args(parser)
parser.add_argument("--steps", type=int, default=1500, help="수집 스텝 (50 Hz — 1500 = 30 s)")
parser.add_argument("--contact_threshold", type=float, default=5.0, help="접지 판정 힘 [N]")
parser.add_argument("--force", type=str, choices=("z", "norm"), default="z",help="접촉력 정의. z = |Fz| (학습 보상과 동일, 기본), norm = ||F||",)
parser.add_argument(
    "--warmup_s", type=float, default=0.5,
    help="리셋 후 이 시간[s] 미만 샘플은 duty/접촉력/하중/추적/통증 지표에서 제외 (낙상·L̂ 전체 지표는 전 구간). "
         "기본 0.5 s: 드롭인 과도 구간 (측정: 0-0.5 s 에서 추적오차 2-8배, duty 급변). 0 이면 전 구간",
)
parser.add_argument("--out_dir", type=str, default=None, help="결과 저장 폴더 (기본: 체크포인트 폴더)")
parser.add_argument("--dump", action="store_true", help="스텝×env 원자료를 raw_records.npz 로 저장")

parser.add_argument("--log_config_path", type=str, default=str(SCRIPT_DIR / "configs" / "logger.yaml"), help="로거 설정 YAML (train.py 와 동일)")
parser.add_argument("--run_tag", type=str, default="", help="실행 구분 태그. 예: p3_norm_off")


AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

check_splint_length_arg(args.splint_length)
if args.warmup_s < 0:
    raise ValueError("--warmup_s 는 0 이상이어야 합니다.")

config = load_config(args)
checkpoint = resolve_checkpoint(args)
# 부상 env 가 있는 phase 는 다섯 조건을 고르게, healthy 는 normal 만
eval_mode = resolve_eval_mode(args, config, default="balanced" if peg_leg_enabled(config) else "normal")
num_envs = args.num_envs if args.num_envs is not None else 40
seed = args.seed if args.seed is not None else config.train.seed
device = args.device


timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
run_tag = args.run_tag.strip()
run_name = f"{timestamp}_{config.phase}_s{seed}" + (f"_{run_tag}" if run_tag else "")

log_dir = SCRIPT_DIR / "logs" / "eval" / run_name
log_dir.mkdir(parents=True, exist_ok=True)

app_logger = create_logger(
    name=run_name,
    log_directory=str(log_dir),
    log_cfgs=read_yaml(args.log_config_path),
)

redirect_python_streams(app_logger)
# 그림 / CSV / npz 도 같은 폴더로. --out_dir 를 주면 그쪽이 우선.
out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else log_dir

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
redirect_python_streams(app_logger) 

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _pain_term(base):
    """학습 보상의 penalty_pain 항 (함수 + yaml 파라미터). 없으면 None (phase 1 등)."""
    try:
        cfg = base.reward_manager.get_term_cfg("penalty_pain")
    except Exception:
        return None
    return cfg.func, dict(cfg.params)


def _head_norm(policy_module) -> dict:
    """phase 3 정책의 출력 정규화 상수 (체크포인트 버퍼). 없으면 항등."""
    def buf(name, default):
        t = getattr(policy_module, name, None)
        return t.detach().cpu().numpy().astype(np.float64) if t is not None else np.asarray(default, np.float64)
    return {
        "L_mean": float(buf("norm_splint_mean", 0.0)), "L_std": float(buf("norm_splint_std", 1.0)),
        "v_mean": buf("norm_vel_mean", np.zeros(3)), "v_std": float(buf("norm_vel_std", 1.0)),
    }


def collect(env, base, policy_fn, policy_module, T: int):
    """T 스텝 동안 지표 원자료를 모은다. 모든 배열의 첫 축은 스텝, 둘째 축은 env."""
    N = base.num_envs
    contacts = base.scene["contact_forces"]
    names = list(contacts.body_names)
    foot_b = torch.tensor([names.index(f"{leg}_foot") for leg in LEGS], device=base.device)
    splint_b = torch.tensor([names.index(f"{leg}_splint") for leg in LEGS], device=base.device)
    calf_b = torch.tensor([names.index(f"{leg}_calf") for leg in LEGS], device=base.device)
    robot = base.scene["robot"]
    has_aux = hasattr(policy_module, "aux_inference")   # phase 3 student 만
    recurrent = getattr(policy_module, "is_recurrent", False)
    pain = _pain_term(base)
    # F_pain 구성은 학습 보상 파라미터를 그대로 따른다 (include_* 가 false 면 해당 항을 0 으로).
    eta = float(pain[1].get("splint_transmission", 0.5)) if pain and pain[1].get("include_splint", True) else 0.0
    w_calf = 1.0 if (pain and pain[1].get("include_calf", True)) else 0.0
    dt = float(base.step_dt)

    rec = {
        "force": np.zeros((T, N, 4), np.float32),   # 다리별 유효 접촉력 (부상 다리 = 부목 끝단)
        "leg": np.zeros((T, N), np.int8),            # -1 정상, 0..3 부상 다리
        "L": np.zeros((T, N), np.float32),           # GT 부목 길이 (정상 = 0)
        "t_reset": np.zeros((T, N), np.float32),     # 정책 호출 시점의 리셋 후 경과 [s]
        "verr": np.zeros((T, N), np.float32),        # |v_cmd,xy - v_xy| (명령은 정책이 본 값)
        "werr": np.zeros((T, N), np.float32),        # |w_cmd,z - w_z|
        "cmd": np.zeros((T, N, 3), np.float32),      # 정책이 본 명령 [vx, vy, wz]
        "fall": np.zeros((T, N), bool),              # env 의 terminated (grace 반영)
        "time_out": np.zeros((T, N), bool),          # env 의 time_out
        "L_hat": np.zeros((T, N), np.float32),       # phase 3: 부목 길이 추정 [m]
        "v_hat": np.zeros((T, N, 3), np.float32),    # phase 3: 속도 추정 [m/s]
        "v_gt": np.zeros((T, N, 3), np.float32),     # privileged base_lin_vel [m/s]
        "pain_F": np.zeros((T, N), np.float32),      # 부상 다리 F_pain [N] (정상 = 0)
        "c_pain": np.zeros((T, N), np.float32),      # 학습 penalty_pain 함수의 C_pain (정상 = 0)
    }
    since_reset = torch.zeros(N, device=base.device)
    obs = env.get_observations()
    with torch.inference_mode():
        for t in range(T):
            leg_pre = base._peg_leg_index.clone()
            rec["leg"][t] = leg_pre.cpu().numpy()
            rec["L"][t] = base._peg_leg_splint_length.cpu().numpy()
            rec["t_reset"][t] = since_reset.cpu().numpy()
            # 정책이 이 스텝에 보는 명령. get_command 는 live 참조라 step 안의 재샘플에 덮이므로 복사한다.
            cmd = base.command_manager.get_command("base_velocity")[:, :3].clone()
            rec["cmd"][t] = cmd.cpu().numpy()
            priv = obs["privileged_obs"]           # [L, vx, vy, vz], 노이즈 없음
            rec["v_gt"][t] = priv[:, 1:4].cpu().numpy()

            actions = policy_fn(obs)
            if has_aux:
                # act_inference 가 남긴 LSTM latent 위에서 보조 헤드를 평가한다 (물리 단위).
                L_hat, v_hat = policy_module.aux_inference()
                rec["L_hat"][t] = L_hat.cpu().numpy()
                rec["v_hat"][t] = v_hat.cpu().numpy()

            obs, _, dones, extras = env.step(actions)
            if recurrent:
                policy_module.reset(dones)
            done_mask = dones.view(-1).bool()
            # 낙상 = env 의 terminated (grace period 반영). time-out 과 같은 스텝이어도 낙상으로 센다.
            rec["fall"][t] = base.reset_terminated.view(-1).cpu().numpy()
            rec["time_out"][t] = base.reset_time_outs.view(-1).cpu().numpy()

            f = contacts.data.net_forces_w
            mag = f[..., 2].abs() if args.force == "z" else f.norm(dim=-1)
            feet = mag[:, foot_b].clone()
            inj = leg_pre >= 0
            if bool(inj.any()):
                rows = torch.nonzero(inj).squeeze(-1)
                feet[rows, leg_pre[rows]] = mag[:, splint_b][rows, leg_pre[rows]]
                # 학습 보상의 통증 하중: 수직력만, 부상 다리의 발 + calf + η·부목
                fz = f[..., 2].abs()
                pain_F = (fz[:, foot_b][rows, leg_pre[rows]]
                          + w_calf * fz[:, calf_b][rows, leg_pre[rows]]
                          + eta * fz[:, splint_b][rows, leg_pre[rows]])
                rec["pain_F"][t, rows.cpu().numpy()] = pain_F.cpu().numpy()
                if pain is not None:
                    rec["c_pain"][t] = pain[0](base, **pain[1]).cpu().numpy()
            rec["force"][t] = feet.cpu().numpy()

            # 이 스텝의 행동이 만든 속도 vs 정책이 본 명령 (학습 보상과 같은 짝)
            rec["verr"][t] = (cmd[:, :2] - robot.data.root_lin_vel_b[:, :2]).norm(dim=-1).cpu().numpy()
            rec["werr"][t] = (cmd[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs().cpu().numpy()

            since_reset += dt
            since_reset[done_mask] = 0.0
            if t % 300 == 0:
                log(f"  step {t}/{T}", flush=True)
    rec["dt"] = dt
    rec["has_aux"] = has_aux
    rec["has_pain"] = pain is not None
    rec["eta"] = eta
    return rec

# main eval function
def summarize(rec, threshold: float, warmup_s: float, norm: dict):

    # TODO: peg leg index 차원 규정이 바뀌어 확인필요
    cond = rec["leg"].astype(int) + 1

    # 각 다리의 힘이 contact_threshold보다 큰지 확인
    contact = rec["force"] > threshold

    # simulation의 한 policy step 시간.
    dt = rec["dt"]

    # reset 직후의 불안정한 구간을 평가에서 제외하기 위한 mask.
    # 리셋이 일어난 그 스텝도 제외한다 — step 안에서 _reset_idx 가 돌아 로봇 상태가 이미
    # 새 에피소드의 랜덤 초기값(reset_base 가 ±0.5 m/s 를 준다)으로 바뀌어 있기 때문이다.
    # 낙상 수 / 에피소드 수 / L̂ 전체 지표는 전 구간(m)을 그대로 쓴다.
    done = rec["fall"] | rec["time_out"]
    warm = (rec["t_reset"] >= warmup_s) & ~done
    rows = []

    # 5가지 조건 Normal / FL / FR / RL / RR
    for g in range(5):

        # 현재 조건에 해당하는 모든 sample을 선택하는 mask.
        m = cond == g                           # (T, N) 전 구간

        # 현재 조건의 전체 sample 개수.
        n = int(m.sum())
        if n == 0:
            continue

        # 현재 조건이면서 동시에 warm-up이 끝난 sample만 선택.
        mw = m & warm                           # 워밍업 제외

        # warm-up을 제외한 실제 평가 sample 개수. 
        nw = int(mw.sum())

        # Duty factor: 접지한 sample 수 / 전체 평가 sample 수 (%)
        duty = [float(contact[..., k][mw].mean()) if nw else float("nan") for k in range(4)]

        # Contact Force
        force = []

        
        for k in range(4):
            # 현재 다리가 접촉하고 있으면서,
            # warm-up이 끝난 현재 조건의 sample만 선택.
            c = contact[..., k] & mw

            # 접촉하고 있을 때의 힘만 평균냄. 
            force.append(float(rec["force"][..., k][c].mean()) if c.any() else 0.0)


        # 시간평균 하중 (접지 여부와 무관, 임계값 영향 없음) 과 네 다리 합 대비 분담률, 공중에 떠 있는 순간의 0 N은 평균에서 제외
        load = [float(rec["force"][..., k][mw].mean()) if nw else float("nan") for k in range(4)]
        
        # load = 전체 시간 기준으로 평균적으로 얼마나 하중을 담당하는가
        # force = 발을 디뎠을 때 얼마나 세게 누르는가
        # FL + FR + RL + RR의 시간 평균 하중을 모두 더함.
        tot = float(np.nansum(load))

        # 각 다리의 load를 전체 다리 load 합으로 나눔.
        share = [l / tot if tot > 0 else float("nan") for l in load]


        row = {
            "group": CONDITION_LABELS[g], "samples": n, "samples_warm": nw,
            # 에피소드 시작 수 (t_reset == 0 인 샘플) — 유효 표본 크기 판단용
            "episodes": int((m & (rec["t_reset"] == 0.0)).sum()),
            **{f"duty_{leg}": duty[k] for k, leg in enumerate(LEGS)},
            **{f"force_{leg}": force[k] for k, leg in enumerate(LEGS)},
            **{f"load_{leg}": load[k] for k, leg in enumerate(LEGS)},
            **{f"share_{leg}": share[k] for k, leg in enumerate(LEGS)},

            # command tracking 성능
            # vel_err_xy = sqrt((vx_cmd - vx_actual)^2 + (vy_cmd - vy_actual)^2)
            # 주의: 딕셔너리 안에 """...""" 를 적으면 다음 키 문자열과 이어 붙어 키 이름이 망가진다.
            "vel_err_xy": float(rec["verr"][mw].mean()) if nw else float("nan"),

            # yaw 추종 성능: yaw_err = |wz_cmd - wz_actual|
            "yaw_err": float(rec["werr"][mw].mean()) if nw else float("nan"),
            "falls": int(rec["fall"][m].sum()),
            "falls_per_min": float(rec["fall"][m].sum() / (n * dt / 60.0)),
        }

        # FL / FR / RL / RR 조건에서만 계산
        if g > 0 and rec["has_pain"]:
            # 부상 다리 통증 하중 [N] 
            row["pain_F_N"] = float(rec["pain_F"][mw].mean()) if nw else float("nan")
            # 실제 학습에서 사용하는 penalty_pain reward 함수가 계산한 pain cost 평균
            row["c_pain"] = float(rec["c_pain"][mw].mean()) if nw else float("nan")

        # aux_inference()가 있는 모델에서만 아래 평가를 수행 (부목길이, 속도 추정)
        if rec["has_aux"]:
            if g > 0:
                # 전 구간 / 수렴 후(리셋 5 s 이후) 두 가지. 짧은 에피소드만 있으면 수렴 후 값은 비어 있다.
                # MAE = mean(|L_hat - L|)
                err_all = np.abs(rec["L_hat"] - rec["L"])[m]

                # 
                row["L_mae_all_m"] = float(err_all.mean())
                row["L_med_all_m"] = float(np.median(err_all))
                # 학습 TB Loss/splint_length 와 같은 z-공간 MSE
                row["L_mse_z"] = float(((err_all / norm["L_std"]) ** 2).mean())

                # reset 후 5초 이상 지난 sample만 선택 -> LSTM이 어느 정도 정보를 축적한 수렴 이후 성능을 보기위함.
                conv = m & (rec["t_reset"] > 5.0)

                # 5초 이상 유지된 sample이 실제로 존재할 때만 계산
                if conv.any():
                    # 수렴 이후의 부목 길이 절대오차.
                    err = np.abs(rec["L_hat"] - rec["L"])[conv]
                    # 평균 절대오차(MAE).
                    row["L_mae_m"] = float(err.mean())
                    # 분포의 가운데(중앙값)와 꼬리(90 분위) — 평균만 보면 소수의 큰 오차에 가린다
                    row["L_med_m"] = float(np.median(err))
                    row["L_mae90_m"] = float(np.quantile(err, 0.9))
                    # 수렴 후 평가에 실제 사용된 sample 수
                    row["L_samples_conv"] = int(conv.sum())

            # 속도 추종 성능
            e = rec["v_hat"][m] - rec["v_gt"][m]                       # (n, 3)
            
            ae = np.abs(e)
            row["v_mae"] = float(ae.mean()) # 축별 |오차| 의 평균 MAE
            for k, ax in enumerate("xyz"):
                row[f"v_mae_{ax}"] = float(ae[:, k].mean())
            # 3D 오차 벡터의 크기 평균 (이전 버전의 v_mae 정의, 축평균 MAE 의 약 2배)
            row["v_err_norm"] = float(np.linalg.norm(e, axis=-1).mean())
            row["v_rmse"] = float(np.sqrt((e ** 2).mean()))
            # 학습 TB Loss/base_lin_vel 과 같은 z-공간 MSE
            row["v_mse_z"] = float(((e / norm["v_std"]) ** 2).mean())

        rows.append(row)
    return rows


def print_tables(rows, warmup_s: float, force_kind: str) -> None:
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

    fdesc = "|Fz|" if force_kind == "z" else "||F||"
    wdesc = f", 리셋 후 {warmup_s:g} s 이후 샘플" if warmup_s > 0 else ""
    table(f"Duty Factor (접지 시간 비율, {fdesc}{wdesc})", "duty", ".3f")
    table(f"Contact Force (N, 접지 중 평균, {fdesc}{wdesc})", "force", ".2f")
    table(f"Load Share (시간평균 {fdesc} 의 네 다리 합 대비 비율, 임계값 무관{wdesc})", "share", ".3f")

    log("\n" + "=" * 84)
    log(f"Tracking / Stability (추적 오차{wdesc}; 낙상은 전 구간)")
    log("=" * 84)
    log(f"{'Group':<8} | {'samples':>8} | {'episodes':>8} | {'|v_xy err| m/s':>14} | {'|yaw err| rad/s':>15} | {'falls':>5} | {'falls/min':>9}")
    log("-" * 84)
    for r in rows:
        log(f"{r['group']:<8} | {r['samples']:>8,} | {r['episodes']:>8} | {r['vel_err_xy']:>14.3f} | "
            f"{r['yaw_err']:>15.3f} | {r['falls']:>5} | {r['falls_per_min']:>9.2f}")

    if any("c_pain" in r for r in rows):
        log("\n" + "=" * 84)
        log("부상 다리 통증 하중 (학습 penalty_pain 과 동일 정의: F_pain = Fz_foot + Fz_calf + η·Fz_splint)")
        log("=" * 84)
        log(f"{'Group':<8} | {'F_pain 평균 N':>13} | {'C_pain 평균':>11}")
        log("-" * 84)
        for r in rows:
            if "c_pain" in r:
                log(f"{r['group']:<8} | {r['pain_F_N']:>13.2f} | {r['c_pain']:>11.4f}")

    if any("v_mae" in r for r in rows):
        log("\n" + "=" * 84)
        log("Phase 3 보조 헤드 (L̂ 는 부상 env 만; '수렴 후' = 리셋 5 s 이후 샘플)")
        log("=" * 84)
        log(f"{'Group':<8} | {'L̂ MAE 전체 m':>13} | {'수렴 후 MAE':>12} | {'수렴 후 med':>12} | {'수렴 후 90%':>12} | "
            f"{'v̂ MAE m/s':>11} | {'‖v̂−v‖ m/s':>11}")
        log("-" * 84)
        for r in rows:
            l0 = f"{r['L_mae_all_m']:.5f}" if "L_mae_all_m" in r else "-"
            l1 = f"{r['L_mae_m']:.5f}" if "L_mae_m" in r else "-"
            l3 = f"{r['L_med_m']:.5f}" if "L_med_m" in r else "-"
            l2 = f"{r['L_mae90_m']:.5f}" if "L_mae90_m" in r else "-"
            log(f"{r['group']:<8} | {l0:>13} | {l1:>12} | {l3:>12} | {l2:>12} | {r['v_mae']:>11.5f} | {r['v_err_norm']:>11.5f}")
        log("-" * 84)
        log("v̂ MAE = 3축 |오차| 평균 (축별 값은 CSV v_mae_x/y/z). ‖v̂−v‖ = 3D 오차 벡터 크기 평균.")
        # 학습 TB 와 같은 혼합비로 맞춘다. splint 손실은 학습에서도 부상 env 만 (마스크) 이지만,
        # 속도 손실은 전 env 평균이라 학습 yaml 의 정상:부상 = 50:50 으로 재가중해야 한다
        # (balanced 평가는 정상 8 / 부상 32 = 20:80 이라 그대로 평균하면 10% 이상 높게 나온다).
        inj = [r for r in rows if "L_mse_z" in r]
        nrm = [r for r in rows if r["group"] == "Normal"]
        parts = []
        if inj:
            parts.append(f"splint_length {np.mean([r['L_mse_z'] for r in inj]):.5f} (부상 env)")
        if inj and nrm:
            v_z = 0.5 * nrm[0]["v_mse_z"] + 0.5 * float(np.mean([r["v_mse_z"] for r in inj]))
            parts.append(f"base_lin_vel {v_z:.5f} (정상:부상 50:50 재가중)")
        elif rows:
            v_z = float(np.mean([r["v_mse_z"] for r in rows]))
            parts.append(f"base_lin_vel {v_z:.5f} ({'정상 env 만' if not inj else '전 조건 평균'})")
        if parts:
            log("z-공간 MSE (학습 TB 와 비교용): " + " | ".join(parts))


def plot_gait(rows, path: Path, force_kind: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(4)
    width = 0.15
    fdesc = "|Fz|" if force_kind == "z" else "||F||"
    for ax, key, ylabel, title in (
        (ax1, "duty", "Duty Factor", f"Duty Factor (Contact Time Ratio, {fdesc})"),
        (ax2, "force", f"Average Force (N, {fdesc})", "Contact Force (injured leg = splint tip)"),
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


def plot_estimation(rec, path: Path) -> bool:
    inj = rec["leg"] >= 0
    if not inj.any():
        return False
    tr = rec["t_reset"][inj]
    errL = np.abs(rec["L_hat"] - rec["L"])[inj]
    L_gt = rec["L"][inj]
    L_hat = rec["L_hat"][inj]
    v_err = np.linalg.norm(rec["v_hat"] - rec["v_gt"], axis=-1)

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
    ax.set_ylabel("|L̂ − L| [m]")
    ax.set_title("Splint length estimation convergence (injured envs)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    conv = tr > 5.0
    sub = np.flatnonzero(conv)[::5]
    ax.scatter(L_gt[sub], L_hat[sub], s=4, alpha=0.25, color=COLORS[0])
    lo, hi = min(L_gt.min(), 0.30), max(L_gt.max(), 0.48)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("GT L [m]")
    ax.set_ylabel("L̂ [m]")
    ax.set_title("L̂ vs GT (converged, >5 s)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    cond = rec["leg"].astype(int) + 1
    for g in range(5):
        m = cond == g
        if m.any():
            ax.hist(v_err[m], bins=40, range=(0, 1.0), histtype="step", lw=1.5,
                    label=CONDITION_LABELS[g], color=COLORS[g], density=True)
    ax.set_xlabel("‖v̂ − v‖ [m/s]")
    ax.set_ylabel("density")
    ax.set_title("Base velocity head error (3D norm)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    env_cfg = build_env_cfg(
        config, num_envs=num_envs, seed=seed, device=device, eval_mode=eval_mode,
        splint_length=args.splint_length, clean=args.clean,
    )

    agent_cfg = build_agent_cfg(config, seed=seed, device=device)

    print_header(config, checkpoint, device, seed, num_envs, eval_mode, env_cfg)

    log(f"[eval] Log dir        : {log_dir}")      # ← 추가
    log(f"[eval] Out dir        : {out_dir}")      # ← 추가

    log(f"[eval] Steps / thresh : {args.steps} / {args.contact_threshold} N "
        f"({'|Fz|' if args.force == 'z' else '||F||'}), warm-up {args.warmup_s:g} s")

    env = wrap_rsl(make_gym_env(config, env_cfg), agent_cfg)
    base = env.unwrapped
    print_conditions(base)
    _, policy_fn, policy_module = make_policy(env, agent_cfg, config, checkpoint, device)
    has_aux = hasattr(policy_module, "aux_inference")
    norm = _head_norm(policy_module)
    log(f"[eval] Aux head       : {'있음 — L̂ / v̂ 로깅' if has_aux else '없음'}")
    if has_aux:
        log(f"[eval] Head de-norm   : L mean {norm['L_mean']:.4f} std {norm['L_std']:.4f} | "
            f"v mean {np.round(norm['v_mean'], 4).tolist()} std {norm['v_std']:.4f} (체크포인트 버퍼)")
    pain = _pain_term(base)
    log(f"[eval] Pain term      : {'있음 — ' + str({k: v for k, v in pain[1].items() if k != 'asset_cfg'}) if pain else '없음'}")

    rec = collect(env, base, policy_fn, policy_module, args.steps)
    rows = summarize(rec, args.contact_threshold, args.warmup_s, norm)
    print_tables(rows, args.warmup_s, args.force)

    out_dir.mkdir(parents=True, exist_ok=True)
    gait_png = out_dir / "gait_analysis.png"
    plot_gait(rows, gait_png, args.force)
    log(f"\n[INFO] 저장: {gait_png}")
    if rec["has_aux"]:
        est_png = out_dir / "estimation_analysis.png"
        if plot_estimation(rec, est_png):
            log(f"[INFO] 저장: {est_png}")
        else:
            log("[INFO] estimation_analysis.png 생략 — 부상 env 가 없습니다")
    csv_path = out_dir / "metrics_summary.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "group", k != "samples", k))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    log(f"[INFO] 저장: {csv_path}")
    if args.dump:
        npz_path = out_dir / "raw_records.npz"
        arrays = {k: v for k, v in rec.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(
            npz_path, **arrays, dt=rec["dt"], eta=rec["eta"],
            L_mean=norm["L_mean"], L_std=norm["L_std"], v_mean=norm["v_mean"], v_std=norm["v_std"],
            contact_threshold=args.contact_threshold, force_kind=args.force, warmup_s=args.warmup_s,
            has_aux=rec["has_aux"], has_pain=rec["has_pain"],
            seed=seed, checkpoint=str(checkpoint),
        )
        log(f"[INFO] 저장: {npz_path}")
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
    logging.shutdown() 
    os._exit(code)
