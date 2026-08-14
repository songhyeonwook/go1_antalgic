"""rollout_dump.py 의 .npz 에서 평가 지표를 뽑는다 (Isaac 불필요).

analyze_dump.py 가 다루지 않는 세 가지:
  [E] play_result.py 스타일 보행 지표 — 그룹(Normal/FL/FR/RL/RR)별 다리별
      duty factor / 접촉력. 부상 다리는 발이 접히므로 부목 접촉으로 대체 (*표시)
  [F] 마찰(μ) 예측 — 단일 프레임 ridge(누설 검사) + 이력 MLP baseline
      (analyze_dump [D] 와 동일 프로토콜, 타깃만 gt_mu)
  [G] 명령 추종 성능 — 그룹별 |v_xy 오차|, |yaw 오차|, 몸통 높이

    PYTHONPATH= python3 analyze_dump_perf.py dumps/p3_final_balanced.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
CONTACT_N = 5.0  # analyze_dump.py 와 동일한 접지 판정 힘


def quat_to_rot(q):
    """(..., 4) wxyz → (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def env_groups(gt_leg):
    """env 별 대표 조건 (에피소드 전체 최빈값): -1=Normal, 0..3=FL/FR/RL/RR."""
    T, N = gt_leg.shape
    rep = np.empty(N, dtype=int)
    for n in range(N):
        vals, counts = np.unique(gt_leg[:, n], return_counts=True)
        rep[n] = int(vals[counts.argmax()])
    return rep


def main(path: str):
    npz = np.load(path)
    meta = json.loads(str(npz["meta"]))
    d = {k: npz[k] for k in npz.files if k != "meta"}
    dt = meta["step_dt"]
    T, N = d["gt_leg"].shape

    gt_leg, gt_L, gt_mu = d["gt_leg"], d["gt_L"], d["gt_mu"]
    f_ft = np.linalg.norm(d["contact_foot"], axis=-1)      # (T, N, 4)
    f_sp = np.linalg.norm(d["contact_splint"], axis=-1)
    rep = env_groups(gt_leg)
    group_names = {-1: "Normal", 0: "FL Peg", 1: "FR Peg", 2: "RL Peg", 3: "RR Peg"}

    print("=" * 78)
    print(f"덤프: {path}  (T={T}, N={N}, dt={dt:.3f}s, 조건={meta['condition']})")
    counts = {g: int((rep == g).sum()) for g in (-1, 0, 1, 2, 3)}
    print("  그룹 구성:", ", ".join(f"{group_names[g]} {c}개" for g, c in counts.items()))

    # ── [E] play_result 스타일 보행 지표 ──
    print("=" * 78)
    print(f"[E] 그룹별 duty factor / 접촉력 (임계 {CONTACT_N:.0f} N, 부상 다리는 부목 접촉 *)")
    header = f"  {'Group':<10} | " + " | ".join(f"{leg:^13}" for leg in LEGS)
    print(header + " |")
    print("  " + "-" * (len(header) + 1))
    for g in (-1, 0, 1, 2, 3):
        envs = np.where(rep == g)[0]
        if len(envs) == 0:
            continue
        cells = []
        for k in range(4):
            # 부상 다리 열은 부목 접촉으로 대체 (발은 접혀서 항상 스윙)
            src = f_sp if (g == k) else f_ft
            force = src[:, envs, k]                      # (T, n_envs)
            contact = force > CONTACT_N
            df = contact.mean()
            avg_f = force[contact].mean() if contact.any() else 0.0
            mark = "*" if g == k else " "
            cells.append(f"{df:.2f} {avg_f:6.1f}N{mark}")
        print(f"  {group_names[g]:<10} | " + " | ".join(cells) + " |")
    print("  (각 셀: duty factor / 접지 중 평균 힘)")

    # ── [F] 마찰(μ) 예측 ──
    print("=" * 78)
    print("[F] 마찰 μ 예측 (부상 env, GT μ∈[0.5,1.5] 균등)")
    inj_mask = gt_leg >= 0
    ti, ni = np.where(inj_mask)
    obs = d["obs_policy"][ti, ni]
    y_mu = gt_mu[ti, ni]
    print(f"  GT μ: mean {y_mu.mean():.3f}, std {y_mu.std():.3f} "
          f"(상수 예측 MAE = {np.abs(y_mu - np.median(y_mu)).mean()*1e3:.0f}e-3)")

    X = np.concatenate([obs, np.ones((len(obs), 1))], 1)
    w = np.linalg.lstsq(X.astype(np.float64), y_mu, rcond=1e-6)[0]
    pred = X @ w
    r2 = 1 - ((y_mu - pred) ** 2).sum() / ((y_mu - y_mu.mean()) ** 2).sum()
    print(f"  단일 프레임 ridge R² = {r2:.3f} (높으면 정적 누설 의심)")

    try:
        import torch
        H = 25
        env_ids = np.where(inj_mask.any(0))[0]
        rng = np.random.default_rng(0)
        test_envs = set(rng.choice(env_ids, size=max(4, len(env_ids) // 5), replace=False))
        Xs, Ys_mu, Ys_L, is_test = [], [], [], []
        for n in env_ids:
            valid = inj_mask[:, n]
            for t in range(H, T, 2):
                if valid[t - H:t].all() and not d["dones"][t - H:t - 1, n].any():
                    Xs.append(d["obs_policy"][t - H:t, n].reshape(-1))
                    Ys_mu.append(gt_mu[t, n])
                    Ys_L.append(gt_L[t, n])
                    is_test.append(n in test_envs)
        Xs = torch.tensor(np.array(Xs), dtype=torch.float32)
        is_test = np.array(is_test)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tr, te = torch.tensor(~is_test), torch.tensor(is_test)

        def fit_mlp(Y):
            Y = torch.tensor(np.array(Y), dtype=torch.float32)
            net = torch.nn.Sequential(
                torch.nn.Linear(Xs.shape[1], 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 1),
            ).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            Xtr, Ytr = Xs[tr].to(dev), Y[tr].to(dev)
            for _ in range(60):
                perm = torch.randperm(len(Xtr), device=dev)
                for i in range(0, len(Xtr), 4096):
                    idx = perm[i:i + 4096]
                    loss = torch.nn.functional.mse_loss(net(Xtr[idx]).squeeze(-1), Ytr[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pe = net(Xs[te].to(dev)).squeeze(-1).cpu().numpy()
            return np.abs(pe - Y[te].numpy())

        mae_mu = fit_mlp(Ys_mu)
        print(f"  이력 MLP (창 {H}step={H*dt:.1f}s, held-out env): "
              f"μ MAE median {np.median(mae_mu):.3f}, 90% {np.quantile(mae_mu, 0.9):.3f} "
              f"(샘플 {int(te.sum())})")
        mae_L = fit_mlp(Ys_L)
        print(f"  (참고) 동일 프로토콜 L MAE: median {np.median(mae_L)*1000:.1f} mm, "
              f"90% {np.quantile(mae_L, 0.9)*1000:.1f} mm")
    except Exception as exc:  # noqa: BLE001
        print(f"  MLP baseline 생략: {type(exc).__name__}: {exc}")

    # RLS 관측 채널이 실제로 갱신됐는지 (prior 상수면 std=0)
    rls_ch = d["obs_policy"][..., 49:51]
    print(f"  rls_estimate 채널 std: L̂_norm {rls_ch[..., 0].std():.4f}, "
          f"√P_norm {rls_ch[..., 1].std():.4f} (0 이면 prior 상수)")

    # ── [G] 명령 추종 성능 ──
    print("=" * 78)
    print("[G] 명령 추종 성능 (그룹별)")
    root = d["root_state"]                                 # (T, N, 13)
    Rw = quat_to_rot(root[..., 3:7])
    v_b = np.einsum("tnji,tnj->tni", Rw, root[..., 7:10])   # world→base
    w_b = np.einsum("tnji,tnj->tni", Rw, root[..., 10:13])
    cmd = d["commands"]                                     # (T, N, 3)
    err_xy = np.linalg.norm(v_b[..., :2] - cmd[..., :2], axis=-1)
    err_yaw = np.abs(w_b[..., 2] - cmd[..., 2])
    height = root[..., 2]
    print(f"  {'Group':<10} | {'|v_xy err| m/s':^14} | {'|yaw err| rad/s':^15} | {'높이 m':^8} |")
    print("  " + "-" * 60)
    for g in (-1, 0, 1, 2, 3):
        envs = np.where(rep == g)[0]
        if len(envs) == 0:
            continue
        print(f"  {group_names[g]:<10} | {err_xy[:, envs].mean():^14.3f} "
              f"| {err_yaw[:, envs].mean():^15.3f} | {height[:, envs].mean():^8.3f} |")
    n_falls = int(d["dones"].sum())
    ep_len = T * N / max(n_falls + N, N)  # 대략적 평균 에피소드 길이
    print(f"  전체 done 수: {n_falls} (timeout 포함), 평균 에피소드 ≈ {ep_len * dt:.1f}s")
    # 부목 길이별 성능 (부상 env)
    print("  L 구간별 (부상 env):")
    for lo, hi in ((0.33, 0.37), (0.37, 0.41), (0.41, 0.45)):
        m = (gt_L >= lo) & (gt_L < hi) & inj_mask
        if m.any():
            print(f"    L∈[{lo:.2f},{hi:.2f}): |v_xy err| {err_xy[m].mean():.3f} m/s, "
                  f"|yaw err| {err_yaw[m].mean():.3f} rad/s, 높이 {height[m].mean():.3f} m")
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(Path(__file__).parent / "dumps" / "p3_final_balanced.npz"))
