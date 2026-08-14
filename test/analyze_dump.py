"""rollout_dump.py 가 만든 .npz 를 분석한다 (Isaac 불필요 — numpy/torch만).

phase 3 설계 결정을 위한 4개 분석:
  [A] 보행 형태 — 부목 duty factor, 접촉력, L별 통계 (RLS 등식 공급량)
  [B] 토크 잔차 게이트 — |τ_hip|/|τ_thigh| 만으로 부목 접지를 감지할 수 있는가
      (GT 접촉으로 ROC/정밀도 채점 — 실기에서 쓸 게이트의 sim 상한선)
  [C] 오프라인 RLS — 착지 기하구속으로 L̂ 수렴 확인
      (oracle 게이트 = 상한, torque 게이트 = 실기 근사)
  [D] 학습 기반 비교 — 단일 프레임 ridge(정적 누설 검사) + 이력 MLP baseline

FK 는 GT body 위치(pos_*_w)로 자체 검증한다. GT 는 검증에만 쓰고
추정기 입력으로는 절대 쓰지 않는다.

    PYTHONPATH= python3 analyze_dump.py dumps/p2_final_balanced.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")

# ── Go1 기구 상수 (URDF) ────────────────────────────────────────────────
HIP_X, HIP_Y = 0.1881, 0.04675      # trunk → hip joint
THIGH_Y = 0.08                       # hip → thigh joint (측방)
THIGH_LEN = 0.213                    # thigh → calf joint
CALF_LEN = 0.213                     # calf → foot
LEG_SIDE = {"FL": +1, "FR": -1, "RL": +1, "RR": -1}
LEG_FRONT = {"FL": +1, "FR": +1, "RL": -1, "RR": -1}
# 부목 (splint/usd_builder.py 와 동일)
SPLINT_LATERAL = 0.055
SPLINT_PITCH = 0.750                 # thigh 프레임에서 -Z 에서 +X 로 기운 각

CONTACT_N = 5.0                      # 접지 판정 힘 (라벨용)
STANCE_N = 20.0                      # RLS 게이트용 확실한 스탠스


def rx(t):
    c, s = np.cos(t), np.sin(t)
    o, z = np.ones_like(t), np.zeros_like(t)
    return np.stack([
        np.stack([o, z, z], -1),
        np.stack([z, c, -s], -1),
        np.stack([z, s, c], -1),
    ], -2)


def ry(t):
    c, s = np.cos(t), np.sin(t)
    o, z = np.ones_like(t), np.zeros_like(t)
    return np.stack([
        np.stack([c, z, s], -1),
        np.stack([z, o, z], -1),
        np.stack([-s, z, c], -1),
    ], -2)


def quat_to_rot(q):
    """(..., 4) wxyz → (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def leg_fk_base(q_hip, q_thigh, q_calf, leg):
    """base 프레임 FK. 반환: p_foot, p_thigh_origin, R_thigh (모두 base 프레임)."""
    side, front = LEG_SIDE[leg], LEG_FRONT[leg]
    p_hip = np.stack([
        np.full_like(q_hip, front * HIP_X),
        np.full_like(q_hip, side * HIP_Y),
        np.zeros_like(q_hip),
    ], -1)
    R1 = rx(q_hip)
    p_thigh = p_hip + (R1 @ np.array([0.0, side * THIGH_Y, 0.0]))
    R2 = R1 @ ry(q_thigh)
    p_calf = p_thigh + (R2 @ np.array([0.0, 0.0, -THIGH_LEN]))
    R3 = R2 @ ry(q_calf)
    p_foot = p_calf + (R3 @ np.array([0.0, 0.0, -CALF_LEN]))
    return p_foot, p_thigh, R2


def splint_ab(p_thigh, R2, leg):
    """부목 끝단 p_tip = a + b·L 의 a, b (base 프레임)."""
    side = LEG_SIDE[leg]
    anchor = np.array([0.0, side * SPLINT_LATERAL, 0.0])
    d = np.array([np.sin(SPLINT_PITCH), 0.0, -np.cos(SPLINT_PITCH)])
    a = p_thigh + (R2 @ anchor)
    b = R2 @ d
    return a, b


def auc_score(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, len(score) + 1)
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (rank[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main(path: str):
    npz = np.load(path)
    meta = json.loads(str(npz["meta"]))
    # ⚠️ NpzFile 은 lazy — 루프 안에서 d[key] 접근 시마다 배열 전체를 다시
    # 압축 해제한다 (실측: part D 가 30분+ 로 폭발). 전부 메모리에 1회 적재.
    d = {k: npz[k] for k in npz.files if k != "meta"}
    dt = meta["step_dt"]
    T, N = d["gt_leg"].shape
    joint_names = meta["joint_names"]
    leg_joint_names = meta["leg_joint_names"]

    jp = d["joint_pos"]                      # (T, N, 16)
    q = {leg: {j: jp[..., joint_names.index(f"{leg}_{j}_joint")]
               for j in ("hip", "thigh", "calf")} for leg in LEGS}
    tau_hip = {leg: d["applied_torque_leg"][..., leg_joint_names.index(f"{leg}_hip_joint")]
               for leg in LEGS}
    tau_thigh = {leg: d["applied_torque_leg"][..., leg_joint_names.index(f"{leg}_thigh_joint")]
                 for leg in LEGS}

    gt_leg, gt_L = d["gt_leg"], d["gt_L"]
    f_sp = np.linalg.norm(d["contact_splint"], axis=-1)   # (T, N, 4)
    f_ft = np.linalg.norm(d["contact_foot"], axis=-1)
    grav = d["projected_gravity"]                          # (T, N, 3) 단위벡터(아래)
    ghat = grav / np.linalg.norm(grav, axis=-1, keepdims=True)

    # ── FK 계산 (전 다리) ──
    p_foot_b = np.zeros((T, N, 4, 3))
    a_b = np.zeros((T, N, 4, 3))
    b_b = np.zeros((T, N, 4, 3))
    for k, leg in enumerate(LEGS):
        pf, pt, R2 = leg_fk_base(q[leg]["hip"], q[leg]["thigh"], q[leg]["calf"], leg)
        p_foot_b[:, :, k] = pf
        a, b = splint_ab(pt, R2, leg)
        a_b[:, :, k] = a
        b_b[:, :, k] = b

    # ── [0] FK 자체 검증 (GT world 위치 대비) ──
    print("=" * 72)
    print("[0] FK 검증 (GT body 위치 대비, base→world 변환 후)")
    root = d["root_state"]                                 # (T, N, 13)
    Rw = quat_to_rot(root[..., 3:7])                       # (T, N, 3, 3)
    pw = root[..., :3]
    foot_w_fk = pw[:, :, None] + np.einsum("tnij,tnkj->tnki", Rw, p_foot_b)
    err_foot = np.linalg.norm(foot_w_fk - d["pos_feet_w"], axis=-1)
    print(f"  발 FK 오차:    mean {err_foot.mean()*1000:.2f} mm, max {err_foot.max()*1000:.2f} mm")
    # 부목 끝단: 부상 env 의 자기 다리만 (GT L 사용 — 검증 전용)
    inj_mask = gt_leg >= 0
    ti, ni = np.where(inj_mask)
    ki = gt_leg[ti, ni]
    tip_b = a_b[ti, ni, ki] + b_b[ti, ni, ki] * gt_L[ti, ni, None]
    tip_w_fk = pw[ti, ni] + np.einsum("mij,mj->mi", Rw[ti, ni], tip_b)
    err_tip = np.linalg.norm(tip_w_fk - d["pos_splint_w"][ti, ni, ki], axis=-1)
    print(f"  부목 끝단 오차: mean {err_tip.mean()*1000:.2f} mm, max {err_tip.max()*1000:.2f} mm")
    fk_ok = err_foot.mean() < 0.01 and err_tip.mean() < 0.02
    print(f"  → {'OK' if fk_ok else 'FAIL — 이후 결과 신뢰 불가'}")

    # 접촉 반경 (데이터에서 추정): 스탠스 중 world z
    st_ft = f_ft > STANCE_N
    r_foot = float(np.median(d["pos_feet_w"][..., 2][st_ft]))
    own_sp = f_sp[ti, ni, ki]
    r_tip = float(np.median(d["pos_splint_w"][ti, ni, ki][own_sp > STANCE_N, 2]))
    print(f"  접촉 반경(실측): 발 {r_foot*1000:.1f} mm, 부목 끝단 {r_tip*1000:.1f} mm")

    # ── [A] 보행 형태 ──
    print("=" * 72)
    print("[A] 보행 형태 (부상 env)")
    own_contact = own_sp > CONTACT_N
    print(f"  부목 duty factor: {own_contact.mean():.2f}")
    print(f"  부목 접촉력 (접지 중): mean {own_sp[own_contact].mean():.1f} N")
    # 부상 무릎(calf joint) 높이 — "무릎 끌기(stump-dragging)" exploit 감시.
    # pain(부목) 추가 후 무통 지지점을 찾는 유인이 생기므로, 무릎이 지면으로
    # 내려오면 calf pain 이 그걸 막는 중이라는 직접 증거가 된다.
    knee_b = np.zeros((T, N, 4, 3))
    for k, leg in enumerate(LEGS):
        _, pt, R2 = leg_fk_base(q[leg]["hip"], q[leg]["thigh"], q[leg]["calf"], leg)
        knee_b[:, :, k] = pt + (R2 @ np.array([0.0, 0.0, -THIGH_LEN]))
    knee_w_z = (pw[:, :, None] + np.einsum("tnij,tnkj->tnki", Rw, knee_b))[..., 2]
    kz = knee_w_z[ti, ni, ki]
    print(f"  부상 무릎 높이: median {np.median(kz)*100:.1f} cm, 최저 {kz.min()*100:.1f} cm, "
          f"<5cm 비율 {np.mean(kz < 0.05)*100:.2f}%  (내려오면 무릎끌기 exploit 신호)")
    # 착지(에지) 빈도
    own_c_tn = np.zeros((T, N), dtype=bool)
    own_c_tn[ti, ni] = own_contact
    touchdown = own_c_tn[1:] & ~own_c_tn[:-1]
    inj_env = inj_mask.any(0)
    td_rate = touchdown[:, inj_env].sum(0).mean() / (T * dt)
    print(f"  착지 빈도: {td_rate:.2f} 회/s (부상 env 평균) — RLS 등식 공급량")
    for lo, hi in ((0.33, 0.37), (0.37, 0.41), (0.41, 0.45)):
        m = (gt_L[ti, ni] >= lo) & (gt_L[ti, ni] < hi)
        if m.any():
            print(f"  L∈[{lo:.2f},{hi:.2f}): duty {own_contact[m].mean():.2f}, "
                  f"힘 {own_sp[m][own_contact[m]].mean() if own_contact[m].any() else 0:.1f} N")

    # ── [B] 토크 게이트 ──
    print("=" * 72)
    print("[B] 토크 접지 감지 (부상 다리, 실기 신호만)")
    tau_h = np.abs(np.stack([tau_hip[leg] for leg in LEGS], -1)[ti, ni, ki])
    tau_t = np.abs(np.stack([tau_thigh[leg] for leg in LEGS], -1)[ti, ni, ki])
    label = own_sp > CONTACT_N
    for name, score in (("|τ_hip|", tau_h), ("|τ_thigh|", tau_t),
                        ("결합 √(τ_hip²+τ_thigh²)", np.hypot(tau_h, tau_t))):
        print(f"  {name:24s} AUC = {auc_score(score, label):.3f}")
    score = np.hypot(tau_h, tau_t)
    ths = np.quantile(score, np.linspace(0.05, 0.99, 120))
    best = None
    for th in ths:
        pred = score > th
        tp = (pred & label).sum()
        prec = tp / max(pred.sum(), 1)
        rec = tp / max(label.sum(), 1)
        if prec >= 0.90 and (best is None or rec > best[2]):
            best = (th, prec, rec)
    if best:
        print(f"  운영점(정밀도≥0.90): th={best[0]:.2f} N·m → 정밀도 {best[1]:.2f}, 재현율 {best[2]:.2f}")
        gate_th = best[0]
    else:
        gate_th = float(np.quantile(score, 0.7))
        print(f"  정밀도 0.90 달성 불가 — 임시 th={gate_th:.2f}")

    # ── [C] 오프라인 RLS ──
    print("=" * 72)
    print("[C] 오프라인 RLS (L̂ 수렴)")
    L_PRIOR, P0, R_NOISE = 0.39, 0.06 ** 2, 0.005 ** 2
    c0 = r_tip - r_foot   # 끝단/발 반경 차 (실측)

    def run_rls(use_torque_gate: bool):
        errs_final, errs_t, n_upd = [], {1.0: [], 2.0: [], 5.0: []}, []
        for n in range(N):
            # 에피소드 분할: done 스텝의 기록은 리셋 이후(새 에피소드) 값이므로
            # done 스텝이 새 세그먼트의 시작이다 (off-by-one 주의 — 실측 확인)
            done_ts = [t for t in range(T) if d["dones"][t, n]]
            seg_starts = [0] + done_ts
            seg_ends = [t - 1 for t in done_ts] + [T - 1]
            for s, e in zip(seg_starts, seg_ends):
                if e - s < 150:
                    continue
                k = int(gt_leg[min(s + 10, e), n])
                if k < 0:
                    continue
                L_true = float(gt_L[min(s + 10, e), n])
                L_hat, P, upd = L_PRIOR, P0, 0
                for t in range(s + 5, e, 3):   # 3-step 서브샘플 (상관 노이즈 완화)
                    if use_torque_gate:
                        sc = float(np.hypot(abs(tau_hip[LEGS[k]][t, n]),
                                            abs(tau_thigh[LEGS[k]][t, n])))
                        splint_stance = sc > gate_th
                    else:
                        splint_stance = f_sp[t, n, k] > STANCE_N
                    if not splint_stance:
                        continue
                    # 건강한 스탠스 발 (부상 다리 제외, 가장 힘 큰 발)
                    ft = f_ft[t, n].copy()
                    ft[k] = 0.0
                    j = int(ft.argmax())
                    if ft[j] < STANCE_N:
                        continue
                    g = ghat[t, n]
                    # ĝ(하향 단위벡터) 투영: 끝단 깊이 ĝ·(a+bL) = 발 깊이 − c0
                    #   → (ĝ·b)·L = ĝ·(p_foot − a) − c0,  coef = ĝ·b > 0
                    coef = float(g @ b_b[t, n, k])
                    if coef < 0.3:
                        continue  # 부목 축이 수평에 가까움 — 정보 없음
                    y = float(g @ (p_foot_b[t, n, j] - a_b[t, n, k])) - c0
                    K = P * coef / (coef * coef * P + R_NOISE)
                    innov = y - coef * L_hat
                    if abs(innov) > 0.08:   # innovation 게이트 (미끄러짐/오검출 기각)
                        continue
                    L_hat += K * innov
                    P *= (1 - K * coef)
                    upd += 1
                    t_ep = (t - s) * dt
                    for tk in errs_t:
                        if abs(t_ep - tk) < 1.5 * dt * 3:
                            errs_t[tk].append(abs(L_hat - L_true))
                if upd >= 5:
                    errs_final.append(abs(L_hat - L_true))
                    n_upd.append(upd)
        return errs_final, errs_t, n_upd

    for gate_name, use_tq in (("oracle 게이트 (GT 접촉 — 상한)", False),
                              ("torque 게이트 (실기 근사)", True)):
        ef, et, nu = run_rls(use_tq)
        ef = np.array(ef)
        print(f"  {gate_name}: 에피소드 {len(ef)}개, 평균 갱신 {np.mean(nu):.0f}회")
        if len(ef):
            print(f"    최종 |L̂−L|: median {np.median(ef)*1000:.1f} mm, "
                  f"90% {np.quantile(ef, 0.9)*1000:.1f} mm")
            for tk, v in et.items():
                if v:
                    print(f"    {tk:.0f}s 시점 오차: median {np.median(v)*1000:.1f} mm (n={len(v)})")

    # ── [D] 학습 기반 비교 ──
    print("=" * 72)
    print("[D] 정적 누설 검사 + 이력 MLP baseline (부상 env)")
    obs = d["obs_policy"][ti, ni]        # (M, 51)
    y_all = gt_L[ti, ni]
    X = np.concatenate([obs, np.ones((len(obs), 1))], 1)
    w = np.linalg.lstsq(X.astype(np.float64), y_all, rcond=1e-6)[0]
    pred = X @ w
    ss = 1 - ((y_all - pred) ** 2).sum() / ((y_all - y_all.mean()) ** 2).sum()
    print(f"  단일 프레임 ridge R² = {ss:.3f}  (기존 IK 모델은 ~0.998 이었음 — 낮을수록"
          " '동역학을 봐야만 아는 문제')")

    try:
        import torch
        H = 25
        env_ids = np.where(inj_mask.any(0))[0]
        rng = np.random.default_rng(0)
        test_envs = set(rng.choice(env_ids, size=max(4, len(env_ids) // 5), replace=False))
        Xs, Ys, is_test = [], [], []
        for n in env_ids:
            valid = inj_mask[:, n]
            for t in range(H, T, 2):
                if valid[t - H:t].all() and not d["dones"][t - H:t - 1, n].any():
                    Xs.append(d["obs_policy"][t - H:t, n].reshape(-1))
                    Ys.append(gt_L[t, n])
                    is_test.append(n in test_envs)
        Xs = torch.tensor(np.array(Xs), dtype=torch.float32)
        Ys = torch.tensor(np.array(Ys), dtype=torch.float32)
        is_test = np.array(is_test)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tr, te = torch.tensor(~is_test), torch.tensor(is_test)
        net = torch.nn.Sequential(
            torch.nn.Linear(Xs.shape[1], 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 1),
        ).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xtr, Ytr = Xs[tr].to(dev), Ys[tr].to(dev)
        for ep in range(60):
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), 4096):
                idx = perm[i:i + 4096]
                loss = torch.nn.functional.mse_loss(net(Xtr[idx]).squeeze(-1), Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pe = net(Xs[te].to(dev)).squeeze(-1).cpu().numpy()
        mae = np.abs(pe - Ys[te].numpy())
        print(f"  이력 MLP (창 {H}step={H*dt:.1f}s, held-out env): "
              f"MAE median {np.median(mae)*1000:.1f} mm, 90% {np.quantile(mae, 0.9)*1000:.1f} mm "
              f"(샘플 {int(te.sum())})")
    except Exception as exc:  # noqa: BLE001
        print(f"  MLP baseline 생략: {type(exc).__name__}: {exc}")

    print("=" * 72)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(Path(__file__).parent / "dumps" / "p2_final_balanced.npz"))
