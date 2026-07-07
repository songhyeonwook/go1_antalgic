# play_robot.py
import argparse
from isaaclab.app import AppLauncher

# 1. 시뮬레이션 앱 실행 설정
parser = argparse.ArgumentParser(description="View Go1 Robot")

# [수정됨] 수동으로 device 인자를 추가하던 줄을 삭제했습니다.
# AppLauncher가 자동으로 --device 인자를 처리해줍니다.
# parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu, cuda)") <--- 삭제됨

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Isaac Lab 모듈 임포트
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# asset.py에서 설정 가져오기
try:
    from asset import GO1_CUSTOM_CFG
except ImportError:
    print("[ERROR] asset.py를 찾을 수 없습니다.")
    exit()

@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    """Configuration for the custom scene."""
    
    # 1. 지형 설정
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
    )

    # 2. 로봇 설정
    robot = GO1_CUSTOM_CFG.replace(prim_path="/World/Robot")

    # 3. 조명 설정
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0, 
            color=(0.75, 0.75, 0.75)
        ),
    )

def main():
    # 3. SimulationContext 초기화
    # args_cli.device는 AppLauncher에 의해 자동으로 생성되어 있습니다.
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 카메라 시점 조정
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # 4. 씬 설정 및 생성
    scene_cfg = Go1SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 5. 시뮬레이션 리셋
    sim.reset()
    
    print("[INFO] Robot spawned. Starting simulation loop...")
    
    while simulation_app.is_running():
        # 1) 로봇 상태 기록
        scene.write_data_to_sim()
        
        # 2) 물리 시뮬레이션 스텝
        sim.step()
        
        # 3) 관측값 업데이트
        scene.update(dt=sim.get_physics_dt())

if __name__ == "__main__":
    main()
    simulation_app.close()