"""부목 모델을 리포의 학습 구조에 얹을 수 있는지 검증한다 (구현 아님, 확인만).

확인 항목
  [1] 관절/액션 인덱스 매핑 — 리포 코드가 joint index 를 action index 로 쓰는 지점
  [2] shape 개수 일관성 — 콜라이더 비활성이 material shape 레이아웃을 깨는가
      (깨지면 events.py 의 발 마찰 적용이 통째로 무효)
  [3] per-reset 부목 길이 재샘플링 — 커리큘럼이 부목 길이를 굴릴 수 있는가
  [4] 정상 다리 동작 동일성 — --baseline 과 수치 비교
  [5] 부목 길이 식별 가능성 — 12 관절 관측만으로 L 을 복원할 수 있는가,
      그리고 그 복원이 nuisance(적재 질량)에 대해 견고한가
      ★ 기존 모델은 L_eff = fk(q_calf) 라는 '정확한 기구학 항등식'이었다 (r=0.9992).
        새 모델이 그 수준으로 새는지, 아니면 하중/접촉을 거쳐야만 알 수 있는지 판별.

    python -u test/spike_train_integration.py
    python -u test/spike_train_integration.py --baseline   # 순정 Go1 비교군
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", action="store_true", help="순정 Go1 (부목 없음)")
parser.add_argument("--attach", choices=["thigh", "hip"], default="thigh")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402

from go1_lab.splint import (  # noqa: E402
    LEGS,
    SPLINT_MIN,
    build_splint_usd,
    set_splint_presence,
)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(TEST_DIR, "assets")
os.makedirs(OUT_DIR, exist_ok=True)
SRC_USD = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd"
DST_USD = os.path.join(OUT_DIR, f"go1_splint_{args.attach}.usd")

BANNER = "=" * 78
N = int(args.num_envs)
FOLD_KNEE = -2.55
# L 과 nuisance(적재 질량)를 격자로 섞어 식별 가능성을 본다.
L_LO, L_HI = 0.33, 0.45
PAYLOAD_LO, PAYLOAD_HI = 0.0, 3.0
CALF_JOINTS = [f"{leg}_calf_joint" for leg in LEGS]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sec(title: str) -> None:
    log(f"\n{BANNER}\n{title}\n{BANNER}")


if not args.baseline:
    if os.path.exists(DST_USD):
        os.remove(DST_USD)
    build_splint_usd(SRC_USD, DST_USD, attach=args.attach)
usd_path = SRC_USD if args.baseline else DST_USD

actuators = {
    "base_legs": DCMotorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.7, saturation_effort=23.7, velocity_limit=30.0,
        stiffness=20.0, damping=0.5,
    ),
}
if not args.baseline:
    actuators["splints"] = ImplicitActuatorCfg(
        joint_names_expr=[f"{leg}_splint_joint" for leg in LEGS],
        effort_limit_sim=1.0e6, stiffness=1.0e5, damping=1.0e3,
    )

init_joints = {
    ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
    "F[L,R]_thigh_joint": 0.8, "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
}
if not args.baseline:
    init_joints[".*_splint_joint"] = SPLINT_MIN

robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=1.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45), joint_pos=init_joints, joint_vel={".*": 0.0}
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=actuators,
)


@configclass
class Cfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1
    )
    robot = robot_cfg
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )


sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device="cuda:0"))
scene = InteractiveScene(
    Cfg(num_envs=N, env_spacing=2.0, replicate_physics=False)
)

# 모든 env 에서 FL 을 부상 다리로 (baseline 은 부목 자체가 없음)
injured_leg = torch.zeros(N, dtype=torch.long)

sim.reset()
robot = scene["robot"]
contacts = scene["contact_forces"]
names = list(robot.data.joint_names)
if not args.baseline:
    stat = set_splint_presence(scene, robot, injured_leg, args.attach)
    log(f"[presence] {stat}")

# ─────────────────────────────────────────────────────────────────────────
sec("[1] 관절 인덱스 vs 액션 인덱스")
LEG_JOINT_SUFFIX = ("_hip_joint", "_thigh_joint", "_calf_joint")
leg_joint_names = [n for n in names if n.endswith(LEG_JOINT_SUFFIX)]
# JointPositionActionCfg(joint_names=[...]) 는 오름차순으로 resolve 되므로
# action 벡터의 k 번째 = leg_joint_names 의 k 번째.
joint_idx = {n: names.index(n) for n in leg_joint_names}
action_idx = {n: k for k, n in enumerate(leg_joint_names)}
log(f"  num_joints={robot.num_joints}  action_dim(12 관절만)={len(leg_joint_names)}")
log(f"  {'관절':18s} {'joint idx':>10s} {'action idx':>11s}   일치")
mismatch = []
for n in CALF_JOINTS:
    j, a = joint_idx[n], action_idx[n]
    ok = j == a
    if not ok:
        mismatch.append((n, j, a))
    log(f"  {n:18s} {j:>10d} {a:>11d}   {'O' if ok else 'X'}")
if mismatch:
    log("  ⚠️ 불일치 발생. 리포는 joint index 를 action 버퍼 인덱스로 그대로 쓴다:")
    log("     go1_lab_env.py:131-133, events.py:748  →  IndexError 또는 오마스킹")
    log(f"     최대 joint idx {max(j for _, j, _ in mismatch)} vs action dim "
        f"{len(leg_joint_names)}")
else:
    log("  전부 일치 — 인덱스 분리 불필요")

# ─────────────────────────────────────────────────────────────────────────
sec("[2] material shape 레이아웃 일관성 (발 마찰 적용의 전제)")
view = robot.root_physx_view
mats = view.get_material_properties()
log(f"  get_material_properties() shape = {tuple(mats.shape)}")
log(f"  root_physx_view.max_shapes = {view.max_shapes}")
per_body = []
for link_path in view.link_paths[0]:
    lv = robot._physics_sim_view.create_rigid_body_view(link_path)
    per_body.append(lv.max_shapes)
log(f"  body 별 shape 수 = {dict(zip(robot.body_names, per_body))}")
tot = sum(per_body)
log(f"  합계 {tot} vs max_shapes {view.max_shapes}  →  "
    f"{'일치 (events.py _foot_shape_spans 성립)' if tot == view.max_shapes else '불일치 (마찰 적용 깨짐)'}")
if not args.baseline:
    sp_shapes = [per_body[robot.body_names.index(f'{leg}_splint')] for leg in LEGS]
    log(f"  부목 링크 shape 수 = {dict(zip(LEGS, sp_shapes))}")
    log("  → 콜라이더를 꺼도 shape 자체는 남는가? "
        f"{'남는다 (레이아웃 안정)' if all(s > 0 for s in sp_shapes) else '사라진다 (레이아웃 불안정)'}")

# ─────────────────────────────────────────────────────────────────────────
if not args.baseline:
    sec("[3] per-reset 부목 길이 재샘플링")
    sp_ids = [names.index(f"{leg}_splint_joint") for leg in LEGS]
    fl_sp = sp_ids[0]
    for trial, (lo, hi) in enumerate([(0.33, 0.45), (0.36, 0.40)]):
        Ls = torch.linspace(lo, hi, N, device=robot.device)
        robot.write_joint_state_to_sim(
            position=Ls.unsqueeze(-1),
            velocity=torch.zeros((N, 1), device=robot.device),
            joint_ids=[fl_sp],
        )
        lim = torch.stack([Ls - 1e-4, Ls + 1e-4], dim=-1).unsqueeze(1)
        robot.write_joint_position_limit_to_sim(
            lim, joint_ids=[fl_sp], env_ids=torch.arange(N, device=robot.device),
            warn_limit_violation=False,
        )
        tgt = robot.data.default_joint_pos.clone()
        tgt[:, fl_sp] = Ls
        tgt[:, names.index("FL_calf_joint")] = FOLD_KNEE
        robot.set_joint_position_target(tgt)
        for _ in range(60):
            scene.write_data_to_sim(); sim.step(); scene.update(1 / 200.0)
        got = robot.data.joint_pos[:, fl_sp]
        err = (got - Ls).abs()
        log(f"  리샘플 #{trial + 1} 범위 [{lo}, {hi}] → 실현 "
            f"[{got.min():.4f}, {got.max():.4f}], 오차 max {err.max():.6f} m")

# ─────────────────────────────────────────────────────────────────────────
sec("[4]/[5] 정상 다리 동작 + 부목 길이 식별 가능성")
fl_calf_i = names.index("FL_calf_joint")
leg_ids = [names.index(n) for n in leg_joint_names]

# L 과 적재 질량(nuisance)을 서로 독립으로 배정한다.
half = N // 2
Ls = torch.linspace(L_LO, L_HI, N, device=robot.device)
payload = torch.zeros(N)
payload[half:] = torch.linspace(PAYLOAD_LO, PAYLOAD_HI, N - half)

masses = view.get_masses()
trunk_b = robot.body_names.index("trunk")
base_trunk = robot.data.default_mass[:, trunk_b].cpu()
masses[:, trunk_b] = base_trunk + payload
view.set_masses(masses, torch.arange(N, dtype=torch.long))

tgt = robot.data.default_joint_pos.clone()
if not args.baseline:
    tgt[:, names.index("FL_splint_joint")] = Ls
    tgt[:, fl_calf_i] = FOLD_KNEE
    robot.write_joint_state_to_sim(
        position=Ls.unsqueeze(-1),
        velocity=torch.zeros((N, 1), device=robot.device),
        joint_ids=[names.index("FL_splint_joint")],
    )
    robot.write_joint_state_to_sim(
        position=torch.full((N, 1), FOLD_KNEE, device=robot.device),
        velocity=torch.zeros((N, 1), device=robot.device),
        joint_ids=[fl_calf_i],
    )
robot.set_joint_position_target(tgt)
for _ in range(args.settle):
    robot.set_joint_position_target(tgt)
    scene.write_data_to_sim(); sim.step(); scene.update(1 / 200.0)

jp = robot.data.joint_pos[:, leg_ids]          # (N, 12) 정책이 보는 관절각
g = robot.data.projected_gravity_b             # (N, 3)
h = robot.data.root_pos_w[:, 2]                # 관측되지 않음 (height_scan off)

healthy_ids = [names.index(n) for n in leg_joint_names
               if not n.startswith("FL_")]
log(f"  정상 다리 9 관절 평균: "
    f"{[round(float(v), 4) for v in jp[:, [leg_joint_names.index(n) for n in leg_joint_names if not n.startswith('FL_')]].mean(0)]}")
log(f"  base height mean/std: {h.mean():.4f} / {h.std():.4f}")
log(f"  projected_gravity mean: {[round(float(v), 4) for v in g.mean(0)]}")

if not args.baseline:
    def r2(X: torch.Tensor, y: torch.Tensor, Xt=None, yt=None) -> float:
        Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1).double()
        w = torch.linalg.lstsq(Xb, y.double().unsqueeze(-1)).solution
        Xe, ye = (Xt, yt) if Xt is not None else (X, y)
        Xeb = torch.cat([Xe, torch.ones(Xe.shape[0], 1, device=Xe.device)], 1).double()
        pred = (Xeb @ w).squeeze(-1)
        ss_res = ((ye.double() - pred) ** 2).sum()
        ss_tot = ((ye.double() - ye.double().mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot)

    obs12 = jp
    no_pay = payload.to(robot.device) < 1e-6
    yes_pay = ~no_pay

    log("\n  L 을 12 관절각만으로 선형 복원:")
    log(f"    전체 fit R²                       = {r2(obs12, Ls):.4f}")
    log(f"    적재 0 만으로 fit (in-sample) R²   = "
        f"{r2(obs12[no_pay], Ls[no_pay]):.4f}")
    log(f"    ↑ 그 모델을 적재 있는 쪽에 적용 R²  = "
        f"{r2(obs12[no_pay], Ls[no_pay], obs12[yes_pay], Ls[yes_pay]):.4f}")
    log("\n  참고: 기존 모델은 L_eff = fk(q_calf) 라는 정확한 기구학 항등식이라")
    log("        단일 관절만으로 r=0.9992 / 평균오차 1.27 mm 였다.")
    per_j = [float(torch.corrcoef(torch.stack([obs12[:, k], Ls]))[0, 1])
             for k in range(obs12.shape[1])]
    best = max(range(len(per_j)), key=lambda k: abs(per_j[k]))
    log(f"    단일 관절 최대 |상관| = {abs(per_j[best]):.4f} "
        f"({leg_joint_names[best]})")

    sp_b = contacts.body_names.index("FL_splint")
    ft_b = contacts.body_names.index("FL_foot")
    f = contacts.data.net_forces_w
    log(f"\n  FL 부목 접촉력 mean = {float(f[:, sp_b].norm(dim=-1).mean()):.2f} N")
    log(f"  FL 발   접촉력 mean = {float(f[:, ft_b].norm(dim=-1).mean()):.2f} N")

log(f"\n{BANNER}\n완료\n{BANNER}")
sys.stdout.flush()
os._exit(0)
