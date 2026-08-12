"""리셋마다 부상 다리와 부목 길이가 재추첨되는지 Isaac Sim 에서 확인한다.

매 리셋에서:
  1. 부상 다리를 재추첨 (정상 / FL / FR / RL / RR)
  2. 부목 길이 L 을 [L_LO, L_HI] 에서 재추첨
  3. splint_presence 로 부목을 그 다리에만 존재하게 (렌더 + 질량 + 콜라이더)
  4. 부목 관절을 L 로 배치하고 per-env limit 으로 잠금
  5. 부상 다리 무릎을 접어 발을 들어올림

리셋마다 표를 찍어 지시값(다리, L)과 실측(부목 관절, 부목/발 접촉력)을 대조한다.
부목이 실제로 붙은 다리에서만 접지하고, 나머지 다리는 0 이어야 한다.

    python -u test/view_reset_random.py                  # GUI
    python -u test/view_reset_random.py --resets 6 --headless
    python -u test/view_reset_random.py --shot --headless # 리셋마다 스크린샷

⚠️ 확인 전용. 기존 env / cfg / 체크포인트는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--attach", choices=["thigh", "hip"], default="thigh")
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--resets", type=int, default=5)
parser.add_argument("--settle", type=int, default=250)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--shot", action="store_true", help="리셋마다 스크린샷 저장")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.shot:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
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
SHOT_DIR = os.path.join(TEST_DIR, "shots")
os.makedirs(OUT_DIR, exist_ok=True)
SRC = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd"
DST = os.path.join(OUT_DIR, f"go1_splint_{args.attach}.usd")

BANNER = "=" * 78
N = int(args.num_envs)
L_LO, L_HI = 0.33, 0.45
FOLD_KNEE = -2.55
NOMINAL_CALF = -1.5


def log(m: str = "") -> None:
    print(m, flush=True)


if os.path.exists(DST):
    os.remove(DST)
build_splint_usd(SRC, DST, attach=args.attach)

robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DST,
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
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8, "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": NOMINAL_CALF, ".*_splint_joint": SPLINT_MIN,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.7, saturation_effort=23.7, velocity_limit=30.0,
            stiffness=20.0, damping=0.5,
        ),
        "splints": ImplicitActuatorCfg(
            joint_names_expr=[f"{leg}_splint_joint" for leg in LEGS],
            effort_limit_sim=1.0e6, stiffness=1.0e5, damping=1.0e3,
        ),
    },
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
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2200.0)
    )


scene_cfg = Cfg(num_envs=N, env_spacing=1.1, replicate_physics=False)
if args.shot:
    scene_cfg.cam = CameraCfg(
        prim_path="/World/shot_cam", update_period=0.0, height=900, width=1600,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
    )

sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device="cuda:0"))
scene = InteractiveScene(scene_cfg)
sim.reset()

robot = scene["robot"]
contacts = scene["contact_forces"]
names = list(robot.data.joint_names)
dev = robot.device

splint_j = [names.index(f"{leg}_splint_joint") for leg in LEGS]
calf_j = [names.index(f"{leg}_calf_joint") for leg in LEGS]
splint_b = [contacts.body_names.index(f"{leg}_splint") for leg in LEGS]
foot_b = [contacts.body_names.index(f"{leg}_foot") for leg in LEGS]
all_envs = torch.arange(N, device=dev)

# 한 줄 배치 + 카메라
ROW_DX = 1.0
base_pose = robot.data.default_root_state.clone()
base_pose[:, 0] = -(
    torch.arange(N, device=dev, dtype=torch.float32) - (N - 1) / 2
) * ROW_DX
base_pose[:, 1] = 0.0
base_pose[:, 2] = 0.45
base_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
base_pose[:, 7:] = 0.0

eye = (0.0, 5.2, 1.05)
aim = (0.0, 0.0, 0.24)
sim.set_camera_view(eye=eye, target=aim)
if args.shot:
    os.makedirs(SHOT_DIR, exist_ok=True)
    scene["cam"].set_world_poses_from_view(
        torch.tensor([eye], device=sim.device), torch.tensor([aim], device=sim.device)
    )

gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
LEG_LABEL = {-1: "정상", 0: "FL", 1: "FR", 2: "RL", 3: "RR"}

log(f"\n{BANNER}")
log(f"리셋 랜덤화 확인 — {N} env, {args.resets} 회 리셋, attach={args.attach}")
log(f"부목 길이 범위 [{L_LO}, {L_HI}] m,  부상 다리 = 정상/FL/FR/RL/RR 균등 추첨")
log(f"{BANNER}")


def do_reset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """부상 다리 + 부목 길이를 재추첨하고 물리에 반영한다."""
    # -1(정상) 포함 5 가지 균등 추첨
    leg = torch.randint(-1, 4, (N,), generator=gen).to(torch.long)
    L = (L_LO + (L_HI - L_LO) * torch.rand(N, generator=gen)).to(dev)

    # (1) 부목 존재 여부 — 렌더 + 질량 + 콜라이더
    set_splint_presence(scene, robot, leg, args.attach)

    # (2) 부목 관절: 부상 다리만 L, 나머지는 주차
    sp_pos = torch.full((N, len(splint_j)), SPLINT_MIN, device=dev)
    has = leg >= 0
    if bool(has.any()):
        rows = torch.nonzero(has, as_tuple=False).squeeze(-1).to(dev)
        cols = leg[has].to(dev)
        sp_pos[rows, cols] = L[rows]
    robot.write_joint_state_to_sim(
        position=sp_pos, velocity=torch.zeros_like(sp_pos), joint_ids=splint_j
    )
    # per-env limit 으로 확실히 잠근다 (부상 다리 관절만 좁힌다)
    for k in range(len(LEGS)):
        lo = torch.full((N,), SPLINT_MIN, device=dev)
        hi = torch.full((N,), SPLINT_MIN, device=dev)
        m = leg.to(dev) == k
        lo = torch.where(m, L - 1e-4, lo - 1e-4)
        hi = torch.where(m, L + 1e-4, hi + 1e-4)
        robot.write_joint_position_limit_to_sim(
            torch.stack([lo, hi], dim=-1).unsqueeze(1),
            joint_ids=[splint_j[k]], env_ids=all_envs, warn_limit_violation=False,
        )

    # (3) 무릎: 부상 다리는 접어 발을 들고, 나머지는 nominal
    calf_pos = torch.full((N, len(calf_j)), NOMINAL_CALF, device=dev)
    if bool(has.any()):
        calf_pos[rows, cols] = FOLD_KNEE
    robot.write_joint_state_to_sim(
        position=calf_pos, velocity=torch.zeros_like(calf_pos), joint_ids=calf_j
    )

    # (4) 루트 포즈 + 나머지 관절 초기화
    robot.write_root_state_to_sim(base_pose)
    jp = robot.data.default_joint_pos.clone()
    jp[:, splint_j] = sp_pos
    jp[:, calf_j] = calf_pos
    robot.write_joint_state_to_sim(position=jp, velocity=torch.zeros_like(jp))

    tgt = robot.data.default_joint_pos.clone()
    tgt[:, splint_j] = sp_pos
    tgt[:, calf_j] = calf_pos
    robot.set_joint_position_target(tgt)
    return leg, L, tgt


for r in range(int(args.resets)):
    leg, L, tgt = do_reset()
    for _ in range(int(args.settle)):
        robot.set_joint_position_target(tgt)
        scene.write_data_to_sim()
        sim.step()
        scene.update(1.0 / 200.0)

    jp = robot.data.joint_pos
    f = contacts.data.net_forces_w
    log(f"\n── 리셋 #{r + 1} " + "─" * 62)
    log(f"  {'env':>3s} {'부상다리':>7s} {'지시 L':>8s} {'실측 관절':>9s} "
        f"{'부목|F|':>8s} {'발|F|':>8s} {'타다리 부목|F| 합':>16s}")
    for i in range(N):
        k = int(leg[i])
        if k < 0:
            got, sf, ff = float("nan"), 0.0, 0.0
            others = sum(float(f[i, splint_b[j]].norm()) for j in range(4))
            li = float("nan")
        else:
            got = float(jp[i, splint_j[k]])
            li = float(L[i])
            sf = float(f[i, splint_b[k]].norm())
            ff = float(f[i, foot_b[k]].norm())
            others = sum(
                float(f[i, splint_b[j]].norm()) for j in range(4) if j != k
            )
        log(f"  {i:>3d} {LEG_LABEL[k]:>7s} {li:>8.4f} {got:>9.4f} "
            f"{sf:>8.2f} {ff:>8.2f} {others:>16.3f}")

    # 자동 검증
    bad = []
    for i in range(N):
        k = int(leg[i])
        stray = sum(
            float(f[i, splint_b[j]].norm()) for j in range(4) if j != k
        )
        if stray > 1e-3:
            bad.append(f"env{i}: 다른 다리 부목 접촉 {stray:.3f} N")
        if k >= 0 and abs(float(jp[i, splint_j[k]]) - float(L[i])) > 2e-3:
            bad.append(
                f"env{i}: 길이 오차 "
                f"{abs(float(jp[i, splint_j[k]]) - float(L[i])):.4f} m"
            )
    log("  검증: " + ("OK — 지정한 다리에만 부목, 길이 일치"
                     if not bad else " / ".join(bad)))

    if args.shot:
        import numpy as np
        from PIL import Image

        rgb = scene["cam"].data.output["rgb"][0].detach().cpu().numpy()
        p = os.path.join(SHOT_DIR, f"reset_{r + 1}.png")
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).save(p)
        log(f"  스크린샷: {p}")

if not args.headless:
    log("\n창을 닫으면 종료됩니다. (마지막 리셋 상태 유지)")
    while sim_app.is_running():
        robot.set_joint_position_target(tgt)
        scene.write_data_to_sim()
        sim.step()
        scene.update(1.0 / 200.0)

log(f"\n{BANNER}\n완료\n{BANNER}")
sys.stdout.flush()
os._exit(0)
