"""1단계: phase-2 run 디렉토리 하나를 넣으면 phase-3 출력 정규화 파라미터를 YAML 로 뽑는다.

    python compute_output_norm.py <run_dir>
    python compute_output_norm.py <run_dir> --tail_frac 0.2      # 옵션

    <run_dir> = logs/unitree_go1_antalgic/<날짜>_phase2_s42_<tag>   (--debug_obs 로 학습한 run)

run_dir 안에서 자동으로 찾는 것
    obs_debug/obs_raw.csv      학습 중 기록된 관측 (L, base_lin_vel, 검증용 act_hip_*)
    obs_debug/action_raw.csv   정책의 결정론적 평균 mu(s) (탐색 노이즈 이전). action 통계는 전부 여기서
    obs_debug/action.csv       정책이 실제로 낸 action = mu + sigma*eps. 덤프 무결성 검증과 노이즈 크기 확인에만 쓴다
    config.json                peg_leg.splint_length_range (L 이론값과 비교)

정규화 식 (2단계에서 phase 3 loss 가 사용)
    속도    z = (v - mean_j) / std     mean_j: 축별 (3),    std: 3축 pooled 하나
    부목 L  z = (L - mean)   / std     부상 env (L > 0) 만
    action  z = (a - mean_j) / std     mean_j: 관절별 (12), std: 12관절 pooled 하나
    pooled std = sqrt( mean_j Var_j )  (std 의 평균이 아니라 분산의 평균의 제곱근)

action 통계
    phase 3 의 BC target 은 결정론적 mu(s) 이므로 Var(mu) 가 필요하다. action_raw.csv 가 mu 를 직접 담고
    있으므로 학습 마지막 tail_frac 구간(teacher = 최종 checkpoint 이므로 후반만)의 표본 평균/분산을 그대로 쓴다.
    보정·추정 없음.
    2026-09-14 이전 덤프는 action.csv(노이즈 샘플)만 있어 Var(a) - sigma^2 로 Var(mu) 를 추정했다. 그 경로는
    제거했다 — 구버전 run 은 신버전 dumper 로 다시 학습하거나 git 이력의 이전 스크립트를 쓴다.

출력
    <run_dir>/obs_debug/mse_norm.yaml   phase3.yaml 의 train: 아래에 그대로 붙여 넣을 mse_norm 블록 (들여쓰기 2칸 포함)
    화면에 같은 내용 + 검증 정보

주의
    action 의 mean/std 는 env yaml 의 정상:부상 비율과 teacher 에 의존한다. phase-2 run 이 바뀌면 다시 생성할 것.
    L 통계는 L curriculum 이 꺼진 run 에서 균등분포 이론값과 일치해야 한다 (스크립트가 비교해 경고).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import yaml

# ─────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--tail_frac", type=float, default=0.2,
               help="action 통계에 쓸 뒤쪽 구간 비율. teacher 는 최종 checkpoint 이므로 학습 후반만 쓴다")
args = p.parse_args()
if not 0.0 < args.tail_frac <= 1.0:
    raise SystemExit(f"--tail_frac 는 (0, 1] 이어야 합니다: {args.tail_frac}")

run = Path(args.run_dir).resolve()
dbg = run / "obs_debug"
for f in ("obs_raw.csv", "action.csv"):
    if not (dbg / f).exists():
        raise SystemExit(f"{dbg / f} 가 없습니다. --debug_obs 로 학습한 phase-2 run 디렉토리를 넘기세요.")
if not (dbg / "action_raw.csv").exists():
    raise SystemExit(
        f"{dbg / 'action_raw.csv'} 가 없습니다 (2026-09-14 이전 dumper 의 구버전 덤프).\n"
        "  이 스크립트는 노이즈 이전 mu(s) 가 기록된 action_raw.csv 만 지원합니다. "
        "신버전 dumper 로 다시 학습하거나, Var(a) - sigma^2 추정 경로가 있는 이전 스크립트를 git 이력에서 쓰세요."
    )


def _strip_prefix(names: list[str]) -> list[str]:
    """a_hip_FL / mu_hip_FL → hip_FL. 두 action 파일의 접두사를 벗겨 같은 이름으로 맞춘다."""
    return [n.split("_", 1)[1] if "_" in n else n for n in names]


def _load_csv(path: Path):
    with open(path) as f:
        names = f.readline().strip().split(",")
    return names, np.loadtxt(path, delimiter=",", skiprows=1)


# ── 1. CSV 읽기 (필요한 열만) ─────────────────────────────────────────────────
with open(dbg / "obs_raw.csv") as f:
    hdr = f.readline().strip().split(",")
cols = (["step", "env_id", "splint_L", "lin_vel_x", "lin_vel_y", "lin_vel_z"]
        + [c for c in hdr if c.startswith("act_hip_")])          # act_hip_* 4개는 덤프 검증용
obs = np.loadtxt(dbg / "obs_raw.csv", delimiter=",", skiprows=1, usecols=[hdr.index(c) for c in cols])

mu_hdr, mu = _load_csv(dbg / "action_raw.csv")                   # step, env_id, mu_* 12
noisy_hdr, noisy = _load_csv(dbg / "action.csv")                 # step, env_id, a_* 12
act_names = _strip_prefix(mu_hdr[2:])
if act_names != _strip_prefix(noisy_hdr[2:]):
    raise SystemExit(f"action_raw.csv 와 action.csv 의 관절 순서가 다릅니다: {act_names} vs {_strip_prefix(noisy_hdr[2:])}")
for name, arr in (("action_raw.csv", mu), ("action.csv", noisy)):
    if not np.array_equal(obs[:, :2], arr[:, :2]):
        raise SystemExit(f"obs_raw.csv 와 {name} 의 (step, env_id) 순서가 다릅니다.")
A_mu, A_noisy = mu[:, 2:], noisy[:, 2:]
n_act = len(act_names)

step, env_id = obs[:, 0], obs[:, 1].astype(int)
L, V, obs_hip = obs[:, 2], obs[:, 3:6], obs[:, 6:10]
print(f"[입력] {run.name}: {len(step):,} rows, step 0..{int(step.max())}, env {np.unique(env_id).tolist()}")

# ── 2. 덤프 검증 ──────────────────────────────────────────────────────────────
# (a) action(t) 가 다음 스텝 obs 의 act_* 와 같아야 한다 (rollout 행만 기록됐다는 뜻).
#     반드시 노이즈 샘플(action.csv)로 본다 — obs 의 act_* 는 환경에 들어간 노이즈 샘플이라 mu 와는 맞지 않는다.
order = np.lexsort((step, env_id))
same_env_next = (env_id[order][1:] == env_id[order][:-1]) & (step[order][1:] == step[order][:-1] + 1)
hip_idx = [act_names.index(f"hip_{leg}") for leg in ("FL", "FR", "RL", "RR")]
prev_a = A_noisy[order][:-1][same_env_next][:, hip_idx]
next_o = obs_hip[order][1:][same_env_next]
match = np.isclose(prev_a, next_o, atol=1e-4).mean()
print(f"[검증] action(t) == obs.act(t+1) 일치율 (hip 4채널, action.csv): {match:.4f}",
      "" if match > 0.95 else "  ← 경고: rollout 이 아닌 행이 섞여 있음 (구버전 덤프?)")
# (b) action_raw 가 정말 노이즈 '이전' 값인지: eps = a - mu 가 0 이 아니고 평균 0 이어야 한다.
eps = A_noisy - A_mu
eps_std = np.sqrt((eps ** 2).mean(axis=0))
if np.abs(eps).max() < 1e-9:
    raise SystemExit("action_raw.csv 가 action.csv 와 같습니다 — mu 가 아니라 노이즈 샘플이 기록됐습니다 (dumper 확인).")
print(f"[검증] 탐색 노이즈 std(a - mu): pooled {np.sqrt((eps ** 2).mean()):.4f}  (관절별 {np.round(eps_std, 3).tolist()}),  mean {eps.mean():+.4f}",
      "" if abs(eps.mean()) < 0.01 else "  ← 경고: 노이즈 평균이 0 이 아님")

# ── 3. 속도: 축별 mean, 3축 pooled std ────────────────────────────────────────
moving = ~np.all(V == 0.0, axis=1)                     # 리셋 프레임 (v 가 정확히 0) 제외, 약 0.1%
Vm = V[moving]
vel_mean = Vm.mean(axis=0)
vel_std_axis = Vm.std(axis=0)
vel_std = float(np.sqrt(Vm.var(axis=0).mean()))

# ── 4. 부목 길이: 부상 env 만, mean/std. 에피소드 수와 이론값을 함께 본다 ───────
Li = L[L > 0]
n_ep = sum(1 + int(np.count_nonzero(np.diff(L[(env_id == e) & (L > 0)]) != 0))
           for e in np.unique(env_id) if np.any((env_id == e) & (L > 0)))
L_mean, L_std = float(Li.mean()), float(Li.std())
L_theory = None
cfg_path = run / "config.json"
if cfg_path.exists():
    try:
        lo, hi = (float(v) for v in json.loads(cfg_path.read_text())["environment"]["values"]["peg_leg"]["splint_length_range"])
        L_theory = (0.5 * (lo + hi), (hi - lo) / np.sqrt(12.0), lo, hi)
    except (KeyError, TypeError, ValueError):
        pass

# ── 5. action: 학습 후반 구간의 mu 표본 평균/분산 (보정 없음) ───────────────────
tail = step >= step.max() * (1.0 - args.tail_frac)
A = A_mu[tail]
inj = L[tail] > 0
act_mean = A.mean(axis=0)
var_mu = A.var(axis=0)
act_std = float(np.sqrt(var_mu.mean()))
pooled = lambda m: float(np.sqrt(A[m].var(axis=0).mean())) if m.any() else float("nan")
act_std_healthy, act_std_injured = pooled(~inj), pooled(inj)
noisy_std_tail = float(np.sqrt(A_noisy[tail].var(axis=0).mean()))          # 참고: 노이즈 샘플 쪽 pooled std
method_desc = f"action_raw.csv 의 mu 표본 분산 (마지막 {args.tail_frac:.0%}, 보정 없음)"

# ── 6. 화면 출력 ──────────────────────────────────────────────────────────────
print(f"\n[속도]  리셋 프레임 {np.count_nonzero(~moving):,} 행 제외, m/s")
for n, m, s in zip("xyz", vel_mean, vel_std_axis):
    print(f"   v{n}: mean {m:+.4f}  std {s:.4f}")
print(f"   pooled std = {vel_std:.4f}   (평균만 출력하는 예측기의 MSE = {vel_std**2:.4f})")

print(f"\n[부목 길이]  부상 env {len(Li):,} rows, 에피소드 {n_ep} 개, m")
print(f"   mean {L_mean:.4f}  std {L_std:.4f}   (기준 MSE = {L_std**2:.5f})")
if L_theory:
    tm, ts, lo, hi = L_theory
    flag = "" if abs(ts - L_std) / ts < 0.05 and abs(tm - L_mean) < 0.005 else "  ← 경고: 이론값과 5% 이상 차이 (curriculum? 에피소드 부족?)"
    print(f"   이론 U[{lo}, {hi}]: mean {tm:.4f}  std {ts:.4f}{flag}")
if n_ep < 100:
    print("   ← 경고: 에피소드 수가 적어 std 추정이 불안정합니다 (L 은 에피소드 당 1개 표본)")

print(f"\n[action]  마지막 {args.tail_frac:.0%} = step ≥ {int(step[tail].min())}, {tail.sum():,} rows")
print(f"   방법: {method_desc}")
print(f"   {'joint':10s} {'mean':>8s} {'std(mu)':>8s} {'std(a)':>8s} {'std(a-mu)':>10s}")
for j, n in enumerate(act_names):
    print(f"   {n:10s} {act_mean[j]:+8.3f} {np.sqrt(var_mu[j]):8.3f} {np.sqrt(A_noisy[tail][:, j].var()):8.3f} {np.sqrt((eps[tail][:, j] ** 2).mean()):10.3f}")
print(f"   pooled std = {act_std:.4f}  (노이즈 샘플 쪽 {noisy_std_tail:.4f}; 정상 env {act_std_healthy:.3f} / 부상 env {act_std_injured:.3f})")
print(f"   (기준 MSE = {act_std**2:.4f})")

# ── 7. YAML 블록 생성 (phase3.yaml 의 train.mse_norm 에 그대로 붙여 넣는다) ─────────────
fmt = lambda xs, nd: "[" + ", ".join(f"{x:.{nd}f}" for x in xs) + "]"
yaml_text = f"""  # 출력 정규화 상수. compute_output_norm.py 가 생성 — 손으로 고치지 말고 다시 생성할 것.
  #   source   : {run.name}
  #   generated: {dt.datetime.now():%Y-%m-%d %H:%M}   rows {len(step):,}   action tail {args.tail_frac:.0%}
  #   action   : {method_desc}
  #   z = (y - mean) / std.  mean 은 차원별, std 는 head 당 pooled 하나 = sqrt(mean_j Var_j)
  mse_norm:
    enable: true

    action_mean: {fmt(act_mean, 3)}   # 관절 순서 hip FL FR RL RR, thigh FL FR RL RR, calf FL FR RL RR
    action_pstd: {act_std:.4f}                   # 12관절 pooled, 노이즈 이전 mu 기준 (정상 {act_std_healthy:.3f} / 부상 {act_std_injured:.3f})

    splint_mean: {L_mean:.4f}                    # 부상 env 만 [m]
    splint_std: {L_std:.4f}

    vel_mean: {fmt(vel_mean, 4)}          # 축 순서 x y z [m/s]
    vel_pstd: {vel_std:.4f}                      # 3축 pooled (축별 {fmt(vel_std_axis, 3)})
"""
parsed = yaml.safe_load(yaml_text)["mse_norm"]            # 붙여 넣기 전에 파싱 가능한지 확인
assert len(parsed["action_mean"]) == n_act and len(parsed["vel_mean"]) == 3 and parsed["enable"] is True
out = dbg / "mse_norm.yaml"
out.write_text(yaml_text)
print("\n" + "=" * 100 + f"\n{yaml_text}" + "=" * 100 + f"\n저장: {out}")
