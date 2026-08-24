"""GT 부목 길이 L 민감도 — teacher 가 privileged L 채널을 실제로 쓰는가.

μ 비식별성 검증(`mu_robustness_report.py`)과 같은 질문을 L 에 대해 던진다:
teacher 입력의 GT L 채널만 흔들었을 때 (물리는 그대로) 행동이 얼마나 바뀌는가.

방법 — `analyze_student.py` [S0] 과 동일한 오프라인 재생:
  a_replay[t] = actor(concat(obs_policy[t], obs_privileged[t]))  ↔  dump action[t+1]
teacher 는 feed-forward MLP 라 hidden state 가 없어 재생이 정확히 일치해야 한다.

privileged 레이아웃: onehot(5) L(1) mu(1) lin_vel(3)  → L 은 index 5 (raw m).
actor 입력 = policy(51) ‖ privileged(10) = 61  → 흔들 채널은 index 56.

    /home/shw/miniconda3/envs/isaac/bin/python l_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DUMP = HERE / "dumps" / "p2_pain2_final.npz"
CKPT = Path(
    "/home/shw/go1_lod/scripts/rsl_rl/logs/unitree_go1_antalgic/backup/antalgic_P2/"
    "2026_08_16_18_10_21_phase2_s42_P2-splint-004-0815/model_10998.pt"
)

POLICY_DIM = 51
L_IDX = POLICY_DIM + 5          # GT 부목 길이 (m)
MU_IDX = POLICY_DIM + 6         # GT 부목 끝단 마찰
VY_IDX = POLICY_DIM + 8         # lin_vel y (스케일 참조용 대조)

L_RANGE = (0.33, 0.45)          # splint_length_range (antalgic.yaml)
MU_RANGE = (0.5, 1.5)           # foot_friction_range
L_PRIOR = 0.39                  # mdp/rls.py RLS_L_PRIOR


BK = Path("/home/shw/go1_lod/scripts/rsl_rl/logs/unitree_go1_antalgic/backup")
TEACHERS = [
    ("antalgic", "p2_pain2_final.npz",
     BK / "antalgic_P2/2026_08_16_18_10_21_phase2_s42_P2-splint-004-0815/model_10998.pt"),
    ("fault_tol", "p2_ft_final.npz",
     BK / "FT_P2/2026_08_15_19_45_20_phase2_s42_P2-fault-tolerant-001/model_10998.pt"),
    ("symmetric", "p2_sym2_final.npz",
     BK / "SYM_P2/2026_08_16_02_55_30_phase2_s42_P2-symmetric-002/model_10998.pt"),
]


def build_head(ckpt: Path, prefix: str) -> nn.Sequential:
    net, _ = build_actor(ckpt, prefix)
    return net


def build_actor(ckpt: Path, prefix: str = "actor") -> tuple[nn.Sequential, int]:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    w = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
    idx = sorted({int(k.split(".")[0]) for k in w})
    layers: list[nn.Module] = []
    for i, li in enumerate(idx):
        lin = nn.Linear(w[f"{li}.weight"].shape[1], w[f"{li}.weight"].shape[0])
        lin.weight.data, lin.bias.data = w[f"{li}.weight"], w[f"{li}.bias"]
        layers.append(lin)
        if i < len(idx) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers).eval()
    return net, w[f"{idx[0]}.weight"].shape[1]


@torch.no_grad()
def act(net: nn.Sequential, obs: np.ndarray) -> np.ndarray:
    return net(torch.from_numpy(obs.astype(np.float32))).numpy()


def main() -> None:
    z = np.load(DUMP)
    d = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"]))

    net, in_dim = build_actor(CKPT)
    obs = np.concatenate([d["obs_policy"], d["obs_privileged"]], axis=-1)  # (T,N,61)
    T, N, D = obs.shape
    assert D == in_dim, f"obs {D} != actor input {in_dim}"

    print(f"[cfg] dump={DUMP.name}  T={T} N={N} obs={D}")
    print(f"[cfg] teacher={meta['checkpoint'].split('/')[-2]}/{meta['checkpoint'].split('/')[-1]}")
    print(f"[cfg] privileged layout={meta['obs_privileged_layout']}\n")

    # ── 유효 구간: 첫 done 이후 (rollout_dump 워밍업 이후에도 초기 전이 배제) ──
    dones = d["dones"]
    valid = np.zeros((T, N), bool)
    for n in range(N):
        first = np.argmax(dones[:, n]) if dones[:, n].any() else 0
        valid[first + 1:, n] = True
    valid[-1] = False                       # action[t+1] 정렬용 마지막 스텝 제외

    gt = d["gt_leg"]
    injured = gt >= 0
    inj = valid & injured
    nor = valid & ~injured
    print(f"[data] 유효 샘플: 부상 {inj.sum():,}  정상 {nor.sum():,}")

    # ── [S0] 재생 검증 ──
    base = act(net, obs.reshape(-1, D)).reshape(T, N, -1)
    ref = d["action"]
    err = np.abs(base[:-1][valid[:-1]] - ref[1:][valid[:-1]])
    print(f"[S0] 재생 검증 |Δaction| mean={err.mean():.3e} max={err.max():.3e} "
          f"({'OK' if err.max() < 1e-3 else 'FAIL — 정렬 확인 필요'})\n")

    a_std = ref[1:][valid[:-1]].std()
    print(f"[scale] 덤프 action std = {a_std:.4f} rad  "
          f"(|Δa| 를 이 값 대비 %로도 표기)\n")

    def sens(idx: int, setter, mask: np.ndarray) -> tuple[float, float, float]:
        """채널 idx 를 setter 로 바꾼 뒤 |Δa| (mean, p95, max) 반환."""
        pert = obs.copy()
        pert[..., idx] = setter(obs[..., idx])
        a = act(net, pert.reshape(-1, D)).reshape(T, N, -1)
        dv = np.abs(a - base)[mask]
        return dv.mean(), np.percentile(dv, 95), dv.max()

    def row(label, idx, setter, mask=inj):
        m, p95, mx = sens(idx, setter, mask)
        print(f"  {label:<34s} {m:8.4f}  {100*m/a_std:6.1f}%  {p95:8.4f}  {mx:8.4f}")

    hdr = f"  {'조건':<34s} {'mean|Δa|':>8s}  {'/std':>6s}  {'p95':>8s}  {'max':>8s}"

    print("── [A] L 채널 섭동 (부상 env, 물리는 그대로) ──")
    print(hdr)
    for dl in (-0.06, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.06):
        row(f"L += {dl:+.2f} m", L_IDX, lambda v, dl=dl: v + dl)
    row(f"L → prior 상수 {L_PRIOR}", L_IDX, lambda v: np.full_like(v, L_PRIOR))
    row("L → 0 (정상 sentinel)", L_IDX, lambda v: np.zeros_like(v))
    print()

    print("── [B] 학습 범위 전폭 스윙 (min→max, 같은 조건에서 μ 와 비교) ──")
    print(hdr)
    for name, idx, (lo, hi) in (("L", L_IDX, L_RANGE), ("mu", MU_IDX, MU_RANGE)):
        pl = obs.copy(); pl[..., idx] = lo
        ph = obs.copy(); ph[..., idx] = hi
        al = act(net, pl.reshape(-1, D)).reshape(T, N, -1)
        ah = act(net, ph.reshape(-1, D)).reshape(T, N, -1)
        dv = np.abs(ah - al)[inj]
        print(f"  {name} {lo} → {hi:<25.2f} {dv.mean():8.4f}  "
              f"{100*dv.mean()/a_std:6.1f}%  {np.percentile(dv,95):8.4f}  {dv.max():8.4f}")
    print()

    print("── [C] 대조: 다른 privileged 채널 (스케일 기준) ──")
    print(hdr)
    row("mu += +0.5", MU_IDX, lambda v: v + 0.5)
    row("lin_vel_y += +0.2 m/s", VY_IDX, lambda v: v + 0.2)
    print()

    # ── 관절별 분해: L 전폭 스윙이 어느 관절을 움직이나 ──
    pl = obs.copy(); pl[..., L_IDX] = L_RANGE[0]
    ph = obs.copy(); ph[..., L_IDX] = L_RANGE[1]
    dv = np.abs(act(net, ph.reshape(-1, D)) - act(net, pl.reshape(-1, D))).reshape(T, N, -1)
    LEGS = ("FL", "FR", "RL", "RR")
    names = [f"{l}_{j}" for j in ("hip", "thigh", "calf") for l in LEGS]
    print("── [D] L 전폭 스윙의 관절별 |Δa| (부상 env) ──")
    per = dv[inj].mean(0)
    order = np.argsort(-per)
    for i in order:
        print(f"  {names[i]:<10s} {per[i]:7.4f} rad")
    print()

    # ── 부상 다리별 분해 ──
    print("── [E] 부상 다리별 L 전폭 스윙 |Δa| ──")
    for k, leg in enumerate(LEGS):
        m = valid & (gt == k)
        if m.sum() == 0:
            continue
        print(f"  {leg}  n={m.sum():6,}  mean|Δa|={dv[m].mean():.4f} rad  "
              f"({100*dv[m].mean()/a_std:.1f}% of std)")
    m = nor
    print(f"  정상 n={m.sum():6,}  mean|Δa|={dv[m].mean():.4f} rad  "
          f"({100*dv[m].mean()/a_std:.1f}% of std)")
    print()

    # ── [F] 다른 teacher 체크포인트에서도 같은가 (3-paradigm arm) ──
    print("── [F] teacher 별 전폭 스윙 민감도 (부상 env, mean|Δa| / action std) ──")
    print(f"  {'teacher':<14s} {'L 0.33→0.45':>14s} {'mu 0.5→1.5':>13s} "
          f"{'vy +0.2':>10s} {'critic ΔV(L)':>13s}")
    for tag, dump_name, ck in TEACHERS:
        dp, cp = HERE / "dumps" / dump_name, Path(ck)
        if not dp.exists() or not cp.exists():
            print(f"  {tag:<14s} 누락 (dump={dp.exists()} ckpt={cp.exists()})")
            continue
        zz = np.load(dp)
        dd = {k: zz[k] for k in zz.files if k != "meta"}
        o = np.concatenate([dd["obs_policy"], dd["obs_privileged"]], -1)
        Tx, Nx, Dx = o.shape
        g = dd["gt_leg"] >= 0
        v = np.zeros((Tx, Nx), bool)
        for n in range(Nx):
            f0 = np.argmax(dd["dones"][:, n]) if dd["dones"][:, n].any() else 0
            v[f0 + 1:, n] = True
        v[-1] = False
        mk = v & g
        anet, _ = build_actor(cp)
        cnet = build_head(cp, "critic")
        std = dd["action"][1:][v[:-1]].std()

        def swing(idx, lo, hi, net=anet):
            a_ = o.copy(); a_[..., idx] = lo
            b_ = o.copy(); b_[..., idx] = hi
            x = act(net, b_.reshape(-1, Dx)) - act(net, a_.reshape(-1, Dx))
            return np.abs(x).reshape(Tx, Nx, -1)[mk].mean()

        vy = o.copy(); vy[..., VY_IDX] += 0.2
        dvy = np.abs(act(anet, vy.reshape(-1, Dx)).reshape(Tx, Nx, -1) -
                     act(anet, o.reshape(-1, Dx)).reshape(Tx, Nx, -1))[mk].mean()
        v_base = act(cnet, o.reshape(-1, Dx)).reshape(Tx, Nx)
        dV = swing(L_IDX, *L_RANGE, net=cnet)
        print(f"  {tag:<14s} {100*swing(L_IDX,*L_RANGE)/std:13.1f}% "
              f"{100*swing(MU_IDX,*MU_RANGE)/std:12.1f}% {100*dvy/std:9.1f}% "
              f"{dV:9.3f} ({100*dV/np.abs(v_base[mk]).mean():.1f}%)")
    print()
    rls_comparison(net, obs, inj, a_std, T, N, D, base)


def rls_comparison(net, obs, inj, std, T, N, D, base) -> None:
    """[G] 같은 물리 범위를 GT L(privileged) 대신 RLS 채널(policy)로 흔들면?

    teacher 는 obs_groups 상 policy 그룹도 받으므로 RLS 추정 L̂ 을 이미 본다.
    GT L 무반응이 '길이가 무용'인지 '중복이라 무시'인지는 이 비교로만 갈린다.
    """
    RLS_L, RLS_P = 49, 50      # policy: rls_estimate = [L_hat_norm, sqrtP_norm]

    def swing(idx, lo, hi):
        a = obs.copy(); a[..., idx] = lo
        b = obs.copy(); b[..., idx] = hi
        x = np.abs(act(net, b.reshape(-1, D)) - act(net, a.reshape(-1, D)))
        return x.reshape(T, N, -1)[inj].mean()

    print("── [G] 정보원 비교: RLS 추정 채널 vs privileged GT (부상 env) ──")
    print(f"  {'채널':<42s} {'mean|Δa|':>9s} {'/std':>7s}")
    for label, idx, lo, hi in (
        ("RLS L̂  (policy[49])  0.33→0.45 상당", RLS_L, -1.0, 1.0),
        ("RLS √P (policy[50])  0→1 (확신→prior)", RLS_P, 0.0, 1.0),
        ("GT L   (privileged[5]) 0.33→0.45", L_IDX, *L_RANGE),
        ("mu     (privileged[6]) 0.5→1.5", MU_IDX, *MU_RANGE),
    ):
        v = swing(idx, lo, hi)
        print(f"  {label:<42s} {v:9.4f} {100*v/std:6.1f}%")

    b = obs.copy(); b[..., RLS_L] = 0.0; b[..., RLS_P] = 1.0
    dv = np.abs(act(net, b.reshape(-1, D)).reshape(T, N, -1) - base)
    print(f"  {'RLS 채널 차단(prior 고정)':<42s} {dv[inj].mean():9.4f} "
          f"{100*dv[inj].mean()/std:6.1f}%")
    lh = obs[..., RLS_L][inj]
    print(f"\n  덤프 내 L̂_norm 실측: p5={np.percentile(lh,5):+.2f} "
          f"p95={np.percentile(lh,95):+.2f} → L̂ ≈ "
          f"[{0.39+0.06*np.percentile(lh,5):.3f}, {0.39+0.06*np.percentile(lh,95):.3f}] m "
          f"(섭동 범위가 실측 범위와 일치)")


if __name__ == "__main__":
    main()
