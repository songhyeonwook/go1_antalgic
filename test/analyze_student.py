"""phase 3 student 를 분석한다 (Isaac 불필요 — rollout_dump 의 .npz + checkpoint).

analyze_dump.py 가 '관측에서 L 을 추정할 수 있는가'를 물었다면, 여기서는
'student 가 실제로 그것을 배웠는가'를 묻는다:

  [S0] 재생 검증 — 덤프된 obs 시퀀스를 student LSTM 에 오프라인 재생해
       hidden state 를 복원하고, 재생 action 이 덤프된 action 과 일치하는지 확인
       (일치해야 이후 latent 분석이 실제 rollout 의 내부 상태를 본 것)
  [S1] latent probe: 부목 길이 L — LSTM hidden(256) → L 선형 probe.
       리셋 후 경과 시간별 수렴 곡선 포함 (RLS [C] 와 비교용)
  [S2] latent probe: 마찰 μ — 동일 프로토콜
  [S3] latent probe: 부상 다리 분류 (Normal/FL/FR/RL/RR 5-way)
  [S4] latent probe: base_lin_vel (privileged 3차원)
  [S5] distillation 충실도 — teacher(61) vs student action 일치도 (그룹/L 구간별)

probe 는 env 단위 held-out split (관측 시퀀스 자체가 env 별로 상관되므로
프레임 단위 split 은 낙관 편향). LSTM 초기 상태를 모르는 첫 세그먼트
(첫 done 이전)는 전부 제외한다.

    PYTHONPATH= python3 analyze_student.py dumps/p3_final_balanced.npz
    (checkpoint 는 덤프 meta 에서 자동으로 읽음 — --checkpoint 로 덮어쓰기 가능)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

LEGS = ("FL", "FR", "RL", "RR")
GROUP_NAMES = {-1: "Normal", 0: "FL Peg", 1: "FR Peg", 2: "RL Peg", 3: "RR Peg"}
TIME_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9))


def build_student(sd: dict, device: str):
    lstm = torch.nn.LSTM(51, 256)
    lstm.load_state_dict({k.removeprefix("memory_s.rnn."): v
                          for k, v in sd.items() if k.startswith("memory_s.rnn.")})
    mlp = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 12),
    )
    mlp.load_state_dict({k.removeprefix("student."): v
                         for k, v in sd.items() if k.startswith("student.")})
    # P3-aux-* 체크포인트의 보조 예측 헤드 (없으면 None — 구버전 호환)
    aux = None
    if any(k.startswith("aux_head.") for k in sd):
        aux = torch.nn.Linear(256, sd["aux_head.weight"].shape[0])
        aux.load_state_dict({k.removeprefix("aux_head."): v
                             for k, v in sd.items() if k.startswith("aux_head.")})
        aux = aux.to(device).eval()
    return lstm.to(device).eval(), mlp.to(device).eval(), aux


def build_teacher(sd: dict, device: str):
    mlp = torch.nn.Sequential(
        torch.nn.Linear(61, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 12),
    )
    mlp.load_state_dict({k.removeprefix("teacher."): v
                         for k, v in sd.items() if k.startswith("teacher.")})
    return mlp.to(device).eval()


def replay_lstm(lstm, obs, dones, device):
    """rollout_dump 의 루프 순서를 그대로 재현한다:
    obs[t]·dones[t] 기록 → reset(dones[t]) → policy 가 obs[t] 소비.
    따라서 obs[t] 를 넣기 전에 dones[t] 로 hidden 을 리셋한다."""
    T, N, _ = obs.shape
    h = torch.zeros(1, N, 256, device=device)
    c = torch.zeros(1, N, 256, device=device)
    lat = torch.empty(T, N, 256)
    obs_t = torch.tensor(obs, dtype=torch.float32)
    dones_t = torch.tensor(dones)
    with torch.no_grad():
        for t in range(T):
            m = dones_t[t]
            if m.any():
                h[:, m] = 0.0
                c[:, m] = 0.0
            out, (h, c) = lstm(obs_t[t:t + 1].to(device), (h, c))
            lat[t] = out[0].cpu()
    return lat.numpy()


def ridge_probe(X_tr, y_tr, X_te, lam=1.0):
    """표준화 후 closed-form ridge. 반환: test 예측."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    Xn = (X_tr - mu) / sd
    Xn = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    A = Xn.T @ Xn + lam * np.eye(Xn.shape[1])
    w = np.linalg.solve(A, Xn.T @ y_tr)
    Xt = np.concatenate([(X_te - mu) / sd, np.ones((len(X_te), 1))], 1)
    return Xt @ w


def report_probe(name, y_te, pred, t_reset_te, unit_scale, unit):
    err = np.abs(pred - y_te)
    r2 = 1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
    print(f"  {name}: R² = {r2:.3f}, MAE median {np.median(err)*unit_scale:.1f} {unit}, "
          f"90% {np.quantile(err, 0.9)*unit_scale:.1f} {unit} (test {len(y_te)})")
    for lo, hi in TIME_BINS:
        m = (t_reset_te >= lo) & (t_reset_te < hi)
        if m.sum() >= 20:
            tag = f"{lo:g}-{hi:g}s" if hi < 1e8 else f">{lo:g}s"
            print(f"    리셋 후 {tag:>7}: MAE median {np.median(err[m])*unit_scale:.1f} {unit} "
                  f"(n={int(m.sum())})")


def main():
    ap = argparse.ArgumentParser(description="phase 3 student latent/충실도 분석")
    ap.add_argument("dump", nargs="?",
                    default=str(Path(__file__).parent / "dumps" / "p3_final_balanced.npz"))
    ap.add_argument("--checkpoint", default=None, help="기본: 덤프 meta 의 checkpoint")
    args = ap.parse_args()

    npz = np.load(args.dump)
    meta = json.loads(str(npz["meta"]))
    d = {k: npz[k] for k in npz.files if k != "meta"}
    dt = meta["step_dt"]
    T, N = d["gt_leg"].shape

    ckpt_path = args.checkpoint or meta["checkpoint"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    lstm, student_mlp, aux_head = build_student(sd, device)
    teacher_mlp = build_teacher(sd, device)

    print("=" * 78)
    print(f"덤프: {args.dump}")
    print(f"체크포인트: {ckpt_path}")

    # ── LSTM 재생 ──
    lat = replay_lstm(lstm, d["obs_policy"], d["dones"], device)   # (T, N, 256)

    # 유효 구간: 첫 done 이후만 (그 전엔 rollout 의 hidden 초기값을 모름)
    # + 리셋 후 경과 시간
    valid = np.zeros((T, N), dtype=bool)
    t_reset = np.full((T, N), np.nan)
    for n in range(N):
        done_ts = np.where(d["dones"][:, n])[0]
        for i, s in enumerate(done_ts):
            e = done_ts[i + 1] if i + 1 < len(done_ts) else T
            valid[s:e, n] = True
            t_reset[s:e, n] = (np.arange(s, e) - s) * dt

    # ── [S0] 재생 검증 ──
    # action[t] = policy(obs[t-1]) 이므로 재생 action 은 한 스텝 밀려 비교
    print("=" * 78)
    print("[S0] 재생 검증 (재생 student action vs 덤프 action)")
    with torch.no_grad():
        act_replay = student_mlp(torch.tensor(lat, dtype=torch.float32).to(device)).cpu().numpy()
    diff = np.abs(act_replay[:-1][valid[:-1]] - d["action"][1:][valid[:-1]])
    print(f"  |Δaction|: mean {diff.mean():.2e}, max {diff.max():.2e} "
          f"({'OK — latent 는 실제 rollout 내부 상태' if diff.max() < 1e-3 else 'FAIL — 정렬 확인 필요'})")

    # ── probe 데이터 구성 (env held-out, 그룹별 층화) ──
    inj = d["gt_leg"] >= 0
    rng = np.random.default_rng(0)
    rep = np.empty(N, dtype=int)
    for n in range(N):
        vals, counts = np.unique(d["gt_leg"][:, n], return_counts=True)
        rep[n] = int(vals[counts.argmax()])
    # 각 조건(Normal/FL/FR/RL/RR)에서 최소 1개 이상 test 로 확보
    test_envs = set()
    for g in (-1, 0, 1, 2, 3):
        envs_g = np.where(rep == g)[0]
        if len(envs_g):
            k = max(1, len(envs_g) // 5)
            test_envs |= set(rng.choice(envs_g, size=k, replace=False))
    is_test_env = np.zeros(N, dtype=bool)
    is_test_env[list(test_envs)] = True
    print(f"  probe split: train {int((~is_test_env).sum())} env / test {int(is_test_env.sum())} env")

    sub = valid.copy()
    sub[::2] = sub[::2] & False  # stride 2 서브샘플 (시간 상관 완화)
    ti, ni = np.where(sub & inj)
    X_all = lat[ti, ni]
    te_m = is_test_env[ni]
    tr_m = ~te_m
    t_reset_te = t_reset[ti, ni][te_m]

    # ── [S1] L probe ──
    print("=" * 78)
    print("[S1] latent → 부목 길이 L (부상 env)")
    y = d["gt_L"][ti, ni]
    pred = ridge_probe(X_all[tr_m], y[tr_m], X_all[te_m])
    report_probe("선형 probe", y[te_m], pred, t_reset_te, 1000, "mm")
    print(f"  (비교: 관측 이력 MLP {'/'.join(['analyze_dump [D]'])} ≈ 9 mm, "
          f"오프라인 RLS ≈ 0.5 mm)")

    # ── [S2] μ probe ──
    print("=" * 78)
    print("[S2] latent → 마찰 μ (부상 env)")
    y = d["gt_mu"][ti, ni]
    base = np.abs(y[te_m] - np.median(y[tr_m])).mean()
    pred = ridge_probe(X_all[tr_m], y[tr_m], X_all[te_m])
    report_probe("선형 probe", y[te_m], pred, t_reset_te, 1000, "e-3")
    print(f"  (상수 예측 MAE = {base*1000:.0f}e-3)")

    # ── [S3] 부상 다리 분류 ──
    print("=" * 78)
    print("[S3] latent → 부상 다리 5-way 분류 (Normal/FL/FR/RL/RR)")
    ti2, ni2 = np.where(sub)
    Xc = torch.tensor(lat[ti2, ni2], dtype=torch.float32)
    yc = torch.tensor(d["gt_leg"][ti2, ni2].astype(np.int64) + 1)  # 0=Normal
    te_c = torch.tensor(is_test_env[ni2])
    W = torch.nn.Linear(256, 5).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=1e-2)
    Xtr, ytr = Xc[~te_c].to(device), yc[~te_c].to(device)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(W(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_c = W(Xc[te_c].to(device)).argmax(-1).cpu()
    acc = (pred_c == yc[te_c]).float().mean()
    print(f"  선형 probe 정확도: {acc*100:.1f}% (test {int(te_c.sum())}, chance 20%)")
    for g in range(5):
        m = yc[te_c] == g
        if m.any():
            a = (pred_c[m] == g).float().mean()
            print(f"    {GROUP_NAMES[g-1]:<7}: {a*100:5.1f}%")

    # ── [S4] base_lin_vel probe ──
    print("=" * 78)
    print("[S4] latent → base_lin_vel (privileged, 전체 env)")
    y3 = d["obs_privileged"][ti2, ni2][:, 7:10]
    te2 = is_test_env[ni2]
    r2s = []
    for a in range(3):
        pred = ridge_probe(lat[ti2, ni2][~te2], y3[~te2, a], lat[ti2, ni2][te2])
        r2s.append(1 - ((y3[te2, a] - pred) ** 2).sum()
                   / ((y3[te2, a] - y3[te2, a].mean()) ** 2).sum())
    print(f"  R²: vx {r2s[0]:.3f}, vy {r2s[1]:.3f}, vz {r2s[2]:.3f}")

    # ── [S5] distillation 충실도 ──
    print("=" * 78)
    print("[S5] teacher vs student action (유효 구간)")
    with torch.no_grad():
        obs_t = torch.tensor(
            np.concatenate([d["obs_policy"], d["obs_privileged"]], -1),
            dtype=torch.float32)
        act_teacher = teacher_mlp(obs_t.to(device)).cpu().numpy()
    # student 재생 action 과 비교 (같은 시점 obs 기준)
    da = np.abs(act_replay - act_teacher)
    print(f"  {'Group':<10} | {'mean|Δa| rad':^12} | {'RMSE rad':^9} |")
    print("  " + "-" * 40)
    for g in (-1, 0, 1, 2, 3):
        envs = np.where(rep == g)[0]
        if len(envs) == 0:
            continue
        m = valid[:, envs]
        dg = da[:, envs][m]
        print(f"  {GROUP_NAMES[g]:<10} | {dg.mean():^12.4f} | {np.sqrt((dg**2).mean()):^9.4f} |")
    print("  L 구간별 (부상 env):")
    gl = d["gt_L"]
    for lo, hi in ((0.33, 0.37), (0.37, 0.41), (0.41, 0.45)):
        m = (gl >= lo) & (gl < hi) & inj & valid
        if m.any():
            print(f"    L∈[{lo:.2f},{hi:.2f}): mean|Δa| {da[m].mean():.4f} rad")

    # ── [S6] aux head 직접 평가 (P3-aux-* 체크포인트만) ──
    print("=" * 78)
    if aux_head is None:
        print("[S6] aux head 없음 — 구버전 체크포인트 (probe 결과만 유효)")
    else:
        print("[S6] aux head [L̂, μ̂] 평가 (부상 env, 유효 구간 전체)")
        with torch.no_grad():
            pred = aux_head(
                torch.tensor(lat[ti, ni], dtype=torch.float32).to(device)
            ).cpu().numpy()
        t_reset_all = t_reset[ti, ni]
        L_hat = pred[:, 0] * 0.06 + 0.39
        mu_hat = pred[:, 1] * 0.5 + 1.0
        gt_L_s = d["gt_L"][ti, ni]
        gt_mu_s = d["gt_mu"][ti, ni]
        report_probe("aux L̂  ", gt_L_s, L_hat, t_reset_all, 1000, "mm")
        # live RLS 채널과 비교 (obs 49 = L̂_norm — prior 상수면 오차가 고정값)
        rls_L = d["obs_policy"][ti, ni, 49] * 0.06 + 0.39
        err_rls = np.abs(rls_L - gt_L_s)
        print(f"  (비교) live RLS 채널 L̂: MAE median {np.median(err_rls)*1000:.1f} mm, "
              f"90% {np.quantile(err_rls, 0.9)*1000:.1f} mm")
        report_probe("aux μ̂  ", gt_mu_s, mu_hat, t_reset_all, 1000, "e-3")
        base = np.abs(gt_mu_s - np.median(gt_mu_s)).mean()
        print(f"  (비교) 상수 예측 μ MAE = {base*1000:.0f}e-3, "
              f"물리특징 회귀 상한 ≈ 140e-3")
    print("=" * 78)


if __name__ == "__main__":
    main()
