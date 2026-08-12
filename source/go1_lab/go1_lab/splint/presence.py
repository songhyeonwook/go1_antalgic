from __future__ import annotations

import torch

LEGS = ("FL", "FR", "RL", "RR")

# 링크를 지울 수 없으므로 질량은 0 대신 무시 가능한 값을 준다.
# (articulation 링크의 정확한 0 질량은 PhysX 에서 유효하지 않다)
TINY_MASS = 1.0e-4


def _visual_paths(env_path: str, leg: str, attach: str) -> list[str]:
    """한 다리의 부목 관련 시각 prim 경로 (링크 + 부착 링크 위의 요소)."""
    return [
        f"{env_path}/Robot/{leg}_splint",
        f"{env_path}/Robot/{leg}_{attach}/{leg}_splint_sleeve_visual",
        f"{env_path}/Robot/{leg}_{attach}/{leg}_splint_cuff_visual",
    ]


def set_splint_presence(
    scene,
    robot,
    injured_leg: torch.Tensor,
    attach: str,
    env_ids: torch.Tensor | None = None,
    tiny: float = TINY_MASS,
) -> dict[str, int]:
    """부목을 부상 다리에만 존재하게 만든다 (렌더 + 질량 + 콜라이더).

    리셋마다 호출할 수 있다 — 세 층 모두 양방향 토글된다.

    Args:
        scene: InteractiveScene
        robot: Articulation
        injured_leg: **(num_envs,)** long, 항상 전체 길이.
            -1 = 부목 없음, 0..3 = LEGS 인덱스.
        attach: "thigh" | "hip" — 부목이 붙은 링크 (시각 prim 경로 결정용).
        env_ids: 갱신할 env 인덱스. None 이면 전체.
            리셋 이벤트에서는 리셋된 env 만 넘겨 비용을 줄인다.

    Returns:
        {"shown", "hidden", "enabled", "disabled"} 카운트.
    """
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    num_envs = int(injured_leg.shape[0])
    if env_ids is None:
        sel = list(range(num_envs))
    else:
        sel = [int(v) for v in env_ids.detach().cpu().tolist()]

    idx = injured_leg.detach().cpu().tolist()
    stat = {"shown": 0, "hidden": 0, "enabled": 0, "disabled": 0}

    for env_i in sel:
        env_path = scene.env_prim_paths[env_i]
        for k, leg in enumerate(LEGS):
            present = idx[env_i] == k

            # ── 렌더 ──
            for p in _visual_paths(env_path, leg, attach):
                prim = stage.GetPrimAtPath(p)
                if not (prim and prim.IsValid()):
                    continue
                im = UsdGeom.Imageable(prim)
                if present:
                    im.MakeVisible()
                else:
                    im.MakeInvisible()
            stat["shown" if present else "hidden"] += 1

            # ── 접촉 ──
            prim = stage.GetPrimAtPath(f"{env_path}/Robot/{leg}_splint/collision")
            if prim and prim.IsValid():
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(bool(present))
                stat["enabled" if present else "disabled"] += 1

    # ── 동역학 ──
    # get/set_masses 는 (num_envs, num_bodies) 전체 텐서를 쓰므로, 갱신 대상
    # 행만 고쳐서 되돌려 준다 (set 은 env_ids 로 대상 env 만 반영).
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    idx_t = injured_leg.detach().cpu()
    sel_t = torch.tensor(sel, dtype=torch.long)

    for k, leg in enumerate(LEGS):
        b = robot.body_names.index(f"{leg}_splint")
        default_m = robot.data.default_mass[:, b].cpu()
        default_i = robot.data.default_inertia[:, b].cpu()
        present_rows = sel_t[idx_t[sel_t] == k]
        absent_rows = sel_t[idx_t[sel_t] != k]
        if present_rows.numel():
            masses[present_rows, b] = default_m[present_rows]
            inertias[present_rows, b] = default_i[present_rows]
        if absent_rows.numel():
            ratio = float(tiny) / default_m[absent_rows]
            masses[absent_rows, b] = float(tiny)
            inertias[absent_rows, b] = default_i[absent_rows] * ratio[:, None]

    robot.root_physx_view.set_masses(masses, sel_t)
    robot.root_physx_view.set_inertias(inertias, sel_t)
    return stat
