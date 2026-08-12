"""부목을 다리별로 '있게/없게' 만드는 런타임 헬퍼 (test/ 스파이크에서 검증 후 승격).

⚠️ 전제: 부목 링크 자체는 지울 수 없다.
   Isaac Lab 의 Articulation 텐서 API 는 모든 env 의 링크/관절 수와 이름이
   같아야 성립한다. MultiUsdFileCfg 로 변형을 나눠도 이 제약은 동일하므로,
   "정상 조건에도 부목 링크가 존재해야 한다"는 피할 수 없다.

그래서 없애는 것은 링크가 아니라 그 링크의 '효과'다. 세 층 모두 런타임에
양방향 토글된다 (test/spike_runtime_collider_toggle.py 로 실측 확인):

  1) 렌더      visibility          → 순수 렌더링 속성. 물리 무관.
  2) 동역학    mass/inertia        → 중력/관성 기여.
  3) 접촉      collisionEnabled    → 접촉 이벤트 자체.
                                     ★ replicate_physics=False 필수.
                                       True 면 프로토타입 하나가 복제되어
                                       env 별 차이를 만들 수 없다.

남는 것(못 없애는 것):
  - prismatic 관절 자유도: joint_pos/joint_vel 텐서에 계속 존재
    → observation / action term 에서 이름으로 제외해야 한다.
  - ContactSensor 의 body 채널: 계속 존재 (값은 0).
  - PhysX 가 그 DOF 를 계속 푼다 (비용은 미미).
"""

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
