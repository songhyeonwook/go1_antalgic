# go1_lab/source/go1_lab/go1_lab/assets.py

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

# 현재 파일 위치 기준 상대 경로로 USD 파일 지정
# (또는 절대 경로 "/home/shw/..."를 직접 써도 됩니다)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join("/home/shw/go1/src/go1_lab/source/go1_lab/go1_lab/asset/go1/go1.usd")

##
# Configuration
##

GO1_CUSTOM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # 중요: USD 파일 내부의 Root Prim 경로가 "/base"인지 확인해야 합니다.
        # Unitree 로봇은 보통 "/base" 또는 "/trunk" 입니다.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4), # 초기 높이 (지형에 따라 조정 필요)
        joint_pos={
            "FL_hip_joint": 0.1,  "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,  "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        # USD의 Joint 이름과 정확히 일치해야 합니다.
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.7,
            velocity_limit=30.0,
            stiffness=25.0, # P-gain
            damping=0.5,    # D-gain
        ),
    },
)