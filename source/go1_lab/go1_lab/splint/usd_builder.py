"""다리 전체 부목(HKAFO 형) USD 변형기.

    골반 옆 부착점 ──[prismatic]── {leg}_splint (막대 + 끝단 접촉구)

부목이 다리 옆을 따라 내려와 발보다 아래에서 접지한다. 부상 다리는 부목으로
걷고 발은 뜬 채로 매달린다. 부목 길이 L_splint = prismatic 관절 위치이며,
어떤 관절 인코더에도 나타나지 않으므로 접촉으로만 관측 가능하다.

부착 링크는 두 가지를 지원한다 (연구 설계상 의미가 다르다):

  attach="thigh"  부목이 hip abduction + thigh 굴곡을 따라 돈다.
                  calf 만 잠긴다 (현재 리포와 동일). 부상 다리가 전후로
                  스윙할 수 있어 보행이 가능하다. 무릎 고정 다리보조기.

  attach="hip"    부목이 hip abduction 만 따라 돈다. thigh 굴곡을 따라가지
                  못하므로 thigh 관절도 함께 잠가야 물리적으로 일관된다.
                  전후 스윙이 불가능한 강체 지팡이(진짜 peg leg).

⚠️ pxr 의존은 함수 내부 lazy import — 함수 호출은 반드시 SimulationApp
   인스턴스화 이후에 할 것. (test/ 스파이크에서 검증 후 승격된 모듈.)
"""

from __future__ import annotations

import math

LEGS = ("FL", "FR", "RL", "RR")

# 다리별 좌/우 부호 (URDF: FL/RL 은 +y, FR/RR 은 -y)
LEG_SIDE = {"FL": +1.0, "FR": -1.0, "RL": +1.0, "RR": -1.0}

# URDF 기하
THIGH_JOINT_Y = 0.08  # hip 프레임에서 thigh joint 까지의 측방 오프셋
GO1_THIGH_LENGTH = 0.213
GO1_CALF_LENGTH = 0.213

# 부목이 다리 메시와 겹치지 않도록 바깥쪽으로 띄우는 거리
SPLINT_LATERAL = 0.055

# 부착 링크(thigh) 프레임에서 부목이 향하는 방향의 피치(Y 축, rad).
# 0 이면 부목이 thigh 의 -Z 를 그대로 따라가는데, nominal 자세에서 thigh 는
# 0.8 rad 굽어 있으므로 부목이 앞아래로 45° 기울어 수직으로 서지 못한다.
#
# nominal(thigh=0.8, calf=-1.5)에서 thigh joint → 발 벡터를 thigh 프레임에서 풀면
#   (0,0,-0.213) + R_y(-1.5)·(0,0,-0.213) = (0.2124, 0, -0.2281),  |·| = 0.3116
# 즉 -Z 에서 +X 쪽으로 0.750 rad 기울어 있다. 이 각도를 부목에 미리 주면
# nominal 자세에서 부목이 다리 라인(≈수직)과 나란해지고, 이후 thigh 가 움직이면
# 부목도 함께 앞뒤로 스윙한다 — 부목으로 걷기 위해 필요한 성질.
SPLINT_PITCH = 0.750
# nominal 자세에서 thigh joint 로부터 발까지의 거리 (부목 길이 하한 판단용)
NOMINAL_LEG_REACH = 0.3116

SPLINT_TIP_RADIUS = 0.018  # 끝단 접촉구 (지면을 딛는 부분)
SPLINT_BAR_RADIUS = 0.010  # 막대 (시각 전용)
SPLINT_MASS = 0.15

# 부목 길이 = 부착점에서 끝단까지. 서 있을 때 thigh joint 는 지면 위 ~0.30 m.
SPLINT_MIN = 0.20  # 주차(= 정상 다리). 끝단이 지면 위에 떠서 접지하지 않는다.
# ⚠️ 이 값은 '설계 상한'이다. 학습에서 뽑는 L 범위가 이보다 크면 USD 관절 상한에
#    조용히 clip 된다 (per-env limit 을 따로 쓰지 않는 경우). 실측으로 확인된
#    함정이므로 L 샘플 범위보다 반드시 여유 있게 잡을 것.
SPLINT_MAX = 0.48

# 시각 요소: 안쪽 막대(부목 링크) + 바깥 슬리브(부착 링크) + 골반 브래킷.
# 슬리브는 anchor(=thigh joint, 골반 옆)에서 시작해 막대가 드러나는 지점까지를
# 덮는다. 슬리브를 얇은 회색으로 두면 흰 로봇에 묻혀 "무릎에서 봉이 나온"
# 것처럼 보이므로, 막대와 같은 계열 색으로 굵게 그린다.
SPLINT_BAR_LEN = 0.22
SPLINT_SLEEVE_LEN = 0.22
SPLINT_SLEEVE_RADIUS = 0.017

# 골반 부착 브래킷: hip 관절에서 부목 축까지 옆으로 뻗는 짧은 막대.
# 부목이 어디에 볼트로 물려 있는지를 눈으로 보여주는 용도.
SPLINT_CUFF_RADIUS = 0.013

COLOR_BAR = (0.90, 0.38, 0.10)
COLOR_SLEEVE = (0.72, 0.28, 0.06)
COLOR_CUFF = (0.30, 0.32, 0.36)


def build_splint_usd(
    src_usd: str,
    dst_usd: str,
    *,
    attach: str = "thigh",
    length_min: float = SPLINT_MIN,
    length_max: float = SPLINT_MAX,
    root_name: str = "go1_splint",
) -> str:
    """src_usd 를 reference 로 물고 4 개의 부목 링크 + prismatic 관절을 추가한다.

    reference 방식이라 원본 USD 는 읽기만 하고 절대 수정하지 않는다.

    Args:
        attach: "thigh" 또는 "hip". 부목을 어느 링크에 붙일지.

    Returns:
        dst_usd 경로.
    """
    if attach not in ("thigh", "hip"):
        raise ValueError(f"attach 는 'thigh' 또는 'hip' 이어야 합니다: {attach!r}")

    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    src_stage = Usd.Stage.Open(src_usd)
    if src_stage is None:
        raise RuntimeError(f"원본 USD 를 열 수 없습니다: {src_usd}")
    src_default = src_stage.GetDefaultPrim()
    if not src_default:
        raise RuntimeError(f"원본 USD 에 defaultPrim 이 없습니다: {src_usd}")

    parent_names = {f"{leg}_{attach}" for leg in LEGS}
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    parent_world: dict[str, Gf.Matrix4d] = {}
    for prim in src_stage.Traverse():
        if prim.GetName() in parent_names:
            parent_world[prim.GetName()] = xf_cache.GetLocalToWorldTransform(prim)
    missing = sorted(parent_names - set(parent_world))
    if missing:
        raise RuntimeError(f"원본 USD 에서 부착 링크를 찾지 못함: {missing}")

    root_world_inv = xf_cache.GetLocalToWorldTransform(src_default).GetInverse()

    stage = Usd.Stage.CreateNew(dst_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{root_name}"))
    root.GetPrim().GetReferences().AddReference(src_usd)
    stage.SetDefaultPrim(root.GetPrim())

    # 부목이 향하는 방향 d 를 만드는 회전.
    #   flip  : X 축 180° → 관절 +Z 가 부착 링크의 -Z(아래)를 향한다.
    #           덕분에 관절 위치가 곧 부목 길이(양수)가 된다.
    #   pitch : Y 축 회전 → nominal 자세에서 부목이 다리 라인과 나란해진다.
    #           (hip 부착은 thigh 굴곡을 안 따라가므로 피치를 주지 않는다)
    phi = -float(SPLINT_PITCH) if attach == "thigh" else 0.0
    m_flip = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0))
    m_pitch = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(phi))
    )
    # USD 는 행벡터 규약(v' = v·M)이라 "flip 먼저, 그 다음 pitch" = m_flip * m_pitch
    rot_joint0 = Gf.Quatf((m_flip * m_pitch).ExtractRotationQuat())
    rot_joint1 = Gf.Quatf(m_flip.ExtractRotationQuat())

    # 부목 방향 단위벡터 d (부착 링크 프레임). 관절 +Z 가 가리키는 방향.
    d = (m_flip * m_pitch).TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))

    for leg in LEGS:
        side = LEG_SIDE[leg]
        parent_path = f"/{root_name}/{leg}_{attach}"
        splint_path = f"/{root_name}/{leg}_splint"

        # 부착점: 다리 라인보다 바깥쪽. hip 링크에 붙일 때는 thigh joint 의
        # 측방 오프셋(0.08)을 더해야 같은 위치가 된다.
        lateral = SPLINT_LATERAL + (THIGH_JOINT_Y if attach == "hip" else 0.0)
        anchor = Gf.Vec3f(0.0, float(side * lateral), 0.0)

        # 부목 링크의 초기 포즈 = 부착점에서 d 방향으로 length_min 만큼 나간 위치.
        # 방향(회전)도 d 에 맞춰야 막대/끝단이 부목 축을 따라 놓인다.
        anchor_d = Gf.Vec3d(anchor[0], anchor[1], anchor[2])
        m_local = Gf.Matrix4d(1.0)
        m_local.SetRotateOnly(m_pitch.ExtractRotationQuat())
        m_local.SetTranslateOnly(anchor_d + d * float(length_min))
        splint_world = m_local * parent_world[f"{leg}_{attach}"] * root_world_inv

        splint_xform = UsdGeom.Xform.Define(stage, Sdf.Path(splint_path))
        splint_prim = splint_xform.GetPrim()
        splint_xform.AddTransformOp().Set(splint_world)

        UsdPhysics.RigidBodyAPI.Apply(splint_prim)
        UsdPhysics.MassAPI.Apply(splint_prim).CreateMassAttr(float(SPLINT_MASS))

        # 접촉 지오메트리: 끝단 구 하나뿐. 부목은 여기서만 지면을 딛는다.
        tip = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{splint_path}/collision"))
        tip.CreateRadiusAttr(float(SPLINT_TIP_RADIUS))
        tip.CreateDisplayColorAttr([Gf.Vec3f(*COLOR_BAR)])
        UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(tip.GetPrim())

        # 안쪽 막대 (시각 전용): 끝단(z=0)에서 부착점 쪽(+Z)으로.
        bar = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{splint_path}/bar_visual"))
        bar.CreateRadiusAttr(float(SPLINT_BAR_RADIUS))
        bar.CreateHeightAttr(float(SPLINT_BAR_LEN))
        bar.CreateAxisAttr(UsdGeom.Tokens.z)
        bar.CreateDisplayColorAttr([Gf.Vec3f(*COLOR_BAR)])
        UsdGeom.Xform(bar.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, SPLINT_BAR_LEN * 0.5)
        )

        # 바깥 슬리브 (시각 전용): 부착 링크에 고정. 막대가 드나드는 통.
        sleeve = UsdGeom.Cylinder.Define(
            stage, Sdf.Path(f"{parent_path}/{leg}_splint_sleeve_visual")
        )
        sleeve.CreateRadiusAttr(float(SPLINT_SLEEVE_RADIUS))
        sleeve.CreateHeightAttr(float(SPLINT_SLEEVE_LEN))
        sleeve.CreateAxisAttr(UsdGeom.Tokens.z)
        sleeve.CreateDisplayColorAttr([Gf.Vec3f(*COLOR_SLEEVE)])
        m_sleeve = Gf.Matrix4d(1.0)
        m_sleeve.SetRotateOnly(m_pitch.ExtractRotationQuat())
        m_sleeve.SetTranslateOnly(anchor_d + d * (SPLINT_SLEEVE_LEN * 0.5))
        UsdGeom.Xform(sleeve.GetPrim()).AddTransformOp().Set(m_sleeve)

        # 골반 브래킷 (시각 전용): hip 관절 축(y=0)에서 부목 축까지 옆으로 뻗어
        # 부목이 골반에 물려 있음을 보여준다. 부착 링크에 고정.
        cuff = UsdGeom.Cylinder.Define(
            stage, Sdf.Path(f"{parent_path}/{leg}_splint_cuff_visual")
        )
        cuff.CreateRadiusAttr(float(SPLINT_CUFF_RADIUS))
        cuff.CreateHeightAttr(float(lateral))
        cuff.CreateAxisAttr(UsdGeom.Tokens.y)
        cuff.CreateDisplayColorAttr([Gf.Vec3f(*COLOR_CUFF)])
        UsdGeom.Xform(cuff.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(0.0, float(side * lateral * 0.5), 0.0)
        )

        # prismatic 관절: 부착 링크 → 부목
        joint = UsdPhysics.PrismaticJoint.Define(
            stage, Sdf.Path(f"/{root_name}/joints/{leg}_splint_joint")
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(splint_path)])
        joint.CreateAxisAttr(UsdGeom.Tokens.z)
        joint.CreateLocalPos0Attr(anchor)
        joint.CreateLocalRot0Attr(rot_joint0)
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(rot_joint1)
        joint.CreateLowerLimitAttr(float(length_min))
        joint.CreateUpperLimitAttr(float(length_max))

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(1.0e5)
        drive.CreateDampingAttr(1.0e3)
        drive.CreateMaxForceAttr(1.0e6)
        drive.CreateTargetPositionAttr(float(length_min))

    stage.GetRootLayer().Save()
    return dst_usd
