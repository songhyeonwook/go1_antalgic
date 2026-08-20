# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go1 Lab 환경 설정 - 표준 Go1 Rough 환경을 상속받아 부목(splint) 부상 시나리오 랜덤화 추가.

부목 모델 v2 요약 (자세한 물리는 mdp/events.py 참고):
  - 로봇 USD 는 부목 링크 4개 + prismatic 관절 4개가 추가된 변형본
    (go1_lab.splint.build_cached_splint_usd 가 시작 시 생성)
  - num_joints=16, action_dim=12 — action/obs 는 12개 다리 관절로 명시 스코핑
    (부목 관절각 = L 이 그대로 노출되면 privileged 누설이고, 실기에 인코더도 없음)
  - policy(=student 배포) 관측 그룹에는 실기(Go1)에 존재하는 신호만 남긴다:
    base_lin_vel 은 privileged 그룹으로 이동
"""

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.envs import mdp as mdp_base
from isaaclab.managers import CurriculumTermCfg as CurTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import (
        UnitreeGo1RoughEnvCfg,
    )
except ImportError:
    try:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        _base_cfg_instance = load_cfg_from_registry(
            "Isaac-Velocity-Rough-Unitree-Go1-v0", "env_cfg_entry_point"
        )
        UnitreeGo1RoughEnvCfg = type(_base_cfg_instance)
    except Exception as e:
        raise ImportError(
            f"표준 Unitree Go1 Rough 환경 설정을 찾을 수 없습니다. "
            f"Isaac Lab이 올바르게 설치되어 있는지 확인하세요. 오류: {e}"
        )

from . import mdp
from .mdp.events import (
    initialize_splint_presence,
    randomize_peg_leg_actuation,
    peg_leg_curriculum,
    enforce_peg_leg_constraints,
)


##
# Environment configuration
##

# 정책 action / 관측 / 리셋 이벤트가 다루는 12개 다리 관절 (부목 관절 제외)
LEG_JOINT_PATTERNS = (".*_hip_joint", ".*_thigh_joint", ".*_calf_joint")


@configclass
class Go1LabPrivilegedObsCfg(ObsGroup):
    # Teacher/critic 에게만 제공되는 privileged observation (sim 전용 GT)
    peg_leg_one_index = ObsTerm(func=mdp.peg_leg_one_hot)  # 부상 다리 one-hot(5)
    peg_leg_splint_length = ObsTerm(func=mdp.peg_leg_splint_length)  # 부목 길이 L(1)
    peg_leg_foot_friction = ObsTerm(func=mdp.peg_leg_foot_friction)  # 부목 끝단 마찰(1)
    # 실기 Go1 에는 몸통 선속도 측정이 없으므로 policy 그룹에서 제거하고 여기로
    # 이동 — teacher/critic 은 obs_groups 매핑으로 계속 사용, student 는 못 봄
    base_lin_vel = ObsTerm(func=mdp_base.base_lin_vel)  # (3)

    def __post_init__(self):
        self.enable_corruption = False # priviliged observation noise
        self.concatenate_terms = True # observation term을 하나의 벡터로 합침

@configclass
class Go1LabEnvCfg(UnitreeGo1RoughEnvCfg):
    """Go1 Lab 환경 설정.
    
   - Phase 결정
   - 로봇 actuator 설정
   - observation 설정
   - domain randomization
   - reward 설정
   - termination 설정
   - 부상 다리 생성
   - curriculum 설정
   
    3-phase 학습 파이프라인:
      Phase 1 (GO1_PHASE=healthy): 정상 보행 pretrain
      Phase 2 (GO1_PHASE=teacher): peg-leg 환경 + privileged obs → Teacher PPO
      Phase 3 (GO1_PHASE=student): Teacher checkpoint 로드 → Student distill
    """
    use_peg_leg: bool = None
    use_peg_leg_action_mask: bool = None
    grace_steps: int = None
    # RLS live 갱신 파라미터 (None 이면 rls_estimate 채널이 prior 상수로 유지)
    rls_params: dict = None

    def __post_init__(self):
        super().__post_init__()

    def _apply_splint_asset_settings(self, peg_leg_cfg: dict) -> str:
        """부목 링크가 추가된 로봇 USD 를 생성해 적용합니다 (전 phase 공통).

        healthy phase 에서도 같은 asset 을 씁니다 — startup 이벤트가 presence 를
        '전부 없음'(렌더 off + 질량≈0 + 콜라이더 off)으로 두므로 동역학은 순정
        Go1 과 사실상 동일하고, 관측/액션 차원은 phase 간 완전히 일치합니다.

        """
        from go1_lab.splint import SPLINT_MIN, build_cached_splint_usd

        attach = str(peg_leg_cfg["attach"]).strip().lower()

        src_usd = self.scene.robot.spawn.usd_path
        self.scene.robot.spawn.usd_path = build_cached_splint_usd(src_usd, attach=attach)
        # 부목 관절은 SPLINT_MIN(주차: 끝단이 지면 위에 떠서 접지하지 않음)에서 시작
        self.scene.robot.init_state.joint_pos[".*_splint_joint"] = SPLINT_MIN

        # startup 에서 presence 를 _peg_leg_index(전부 -1)와 동기화.
        # 이후 reset 이벤트는 diff 로만 갱신 → env_fixed 학습에서는 토글 비용 0.
        self.events.init_splint_presence = EventTerm(
            func=initialize_splint_presence,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot"), "attach": attach},
        )
        return attach

    def _apply_joint_scope_settings(self) -> None:
        """action / 관측 / 관절 리셋 이벤트를 12개 다리 관절로 명시 제한합니다.

        부목 관절 4개가 추가되어 num_joints=16 이므로 기본값 ".*" 를 그대로 두면:
          - action 이 16차원이 되고 (부목은 기구이지 근육이 아님)
          - 부목 관절각 = L 이 policy 관측에 그대로 노출되며 (privileged 누설,
            실기에 해당 인코더도 없음)
          - reset_joints_by_scale 이 부목 관절 주차 위치를 흔들어 놓습니다.
        """
        self.actions.joint_pos.joint_names = list(LEG_JOINT_PATTERNS)
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(LEG_JOINT_PATTERNS)
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(LEG_JOINT_PATTERNS)
        )
        if getattr(self.events, "reset_robot_joints", None) is not None:
            self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(LEG_JOINT_PATTERNS)
            )
        # 관절 기반 리워드도 다리 관절로 제한. 부목 prismatic 드라이브의 유지력은
        # |τ| ~ 1e4 N 스케일이라 (실측), 전 관절 집계 시 dof_torques_l2 가
        # -1e7/step 로 폭발해 나머지 리워드를 전부 삼켜버린다.
        for term_name in ("dof_torques_l2", "dof_acc_l2", "dof_pos_limits"):
            term = getattr(self.rewards, term_name, None)
            if term is not None:
                term.params["asset_cfg"] = SceneEntityCfg(
                    "robot", joint_names=list(LEG_JOINT_PATTERNS)
                )

    def _apply_actuator_settings(self, cfg):
        actuator_type = str(cfg["type"]).strip().lower()
        
        if actuator_type != "pd":
            raise ValueError(
                f"Unsupported actuator type: {actuator_type}"
            )
        
        if not hasattr(self.scene, "robot"):
            raise AttributeError("env_cfg.scene.robot does not exist")
            
        
        pd_cfg = cfg['pd']
        effort_limit = float(pd_cfg["effort_limit"])
        
        # scene: 시뮬레이션 세계 안에 배치되는 물체와 센서를 묶어서 관리하는 설정 객체
        """
        scene
        ├── robot
        ├── terrain
        ├── contact_sensor
        ├── height_scanner
        ├── lights
        └── 기타 rigid object
        """
        self.scene.robot.actuators = {
            "base_legs": DCMotorCfg(
                joint_names_expr=list(LEG_JOINT_PATTERNS),
                effort_limit=effort_limit,
                saturation_effort=effort_limit,
                velocity_limit=float(pd_cfg["velocity_limit"]),
                stiffness=float(pd_cfg["kp"]),
                damping=float(pd_cfg["kd"]),
                friction=float(pd_cfg["friction"]),
            ),
            # 부목 prismatic 관절: 학습 대상이 아닌 '기구'. 고강성 드라이브가
            # per-env joint limit 과 함께 관절을 L 위치에 붙잡는다
            # (게인은 usd_builder 의 드라이브 설정과 동일, spike 로 오차 <1e-3 m 확인).
            "splints": ImplicitActuatorCfg(
                joint_names_expr=[".*_splint_joint"],
                effort_limit_sim=1.0e6,
                stiffness=1.0e5,
                damping=1.0e3,
            ),
        }
        
    def _apply_simulation_settings(self, cfg: dict) -> None:
        self.scene.replicate_physics = bool(cfg["replicate_physics"])
        self.scene.clone_in_fabric = bool(cfg["clone_in_fabric"])

        # 부목 presence(질량/콜라이더의 env 별 차이)는 프로토타입 복제와 양립 불가.
        # True 면 조용히 모든 env 에 부목이 남는 잘못된 물리가 되므로 즉시 실패시킨다.
        if self.scene.replicate_physics:
            raise ValueError(
                "splint presence 는 env 별 USD 차이가 필요하므로 "
                "simulation.replicate_physics=false 여야 합니다."
            )

        gpu_cfg = cfg["gpu"]

        self.sim.physx.gpu_total_aggregate_pairs_capacity = int(
            gpu_cfg["gpu_total_aggregate_pairs_capacity"]
        )
        self.sim.physx.gpu_found_lost_pairs_capacity = int(
            gpu_cfg["gpu_found_lost_pairs_capacity"]
        )
        
        terrain_cfg = cfg["terrain"]

        # 지형이 flat이면
        if bool(terrain_cfg["flat"]):
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            # curriculum은 학습이 진행될수록 환경의 난이도를 단계적으로 바꾸는 설정 모음
            # self.curriculum.terrain_levels: 로봇의 성능에 따라 지형 난이도 레벨을 올리거나 내리는 항목


        terrain_levels = terrain_cfg['curriculum']["terrain_levels"]
        if terrain_levels is None:
            self.curriculum.terrain_levels = None

        compliance_cfg = cfg["contact_compliance"]

        # 접촉의 물리적 성질을 설정(바닥의 단단함과 단단함에 따른 충격설정)
        if bool(compliance_cfg["enabled"]):
            stiffness = float(compliance_cfg["stiffness"])
            damping = float(compliance_cfg["damping"])
            # [접촉면이 얼마나 단단한지, 접촉할 때 충격이나 진동을 얼마나 감쇠]
            """
            stiffness가 큼: 단단한 바닥처럼 반응
            stiffness가 작음: 푹신하거나 눌리는 바닥처럼 반응
            
            damping이 큼: 충격과 튀어 오름이 많이 줄어듦
            damping이 작음: 충돌 후 진동하거나 튀는 현상이 커질 수 있음
            """
            materials = [self.sim.physics_material, self.scene.terrain.physics_material]
            
            # 시뮬레이션 기본 재질과 지형 재질을 찾아서 지원 가능한 재질에 접촉 스프링 강성인 stiffness와 접촉 감쇠인 damping을 적용
            for material in materials:
                # if material is not None and hasattr(material, "compliant_contact_stiffness"):
                material.compliant_contact_stiffness = stiffness
                material.compliant_contact_damping = damping    
        
    # 속도 명령 범위
    def _apply_command_settings(self, cfg: dict) -> None:
        ranges = self.commands.base_velocity.ranges

        linear_x_cfg = cfg["linear_velocity_x"]

        ranges.lin_vel_x = (
            float(linear_x_cfg["min"]),
            float(linear_x_cfg["max"]),
        )

        linear_y_abs = float(cfg["linear_velocity_y_abs"])
        ranges.lin_vel_y = (
            -linear_y_abs,
            linear_y_abs,
        )

        yaw_abs = float(cfg["angular_velocity_yaw_abs"])
        ranges.ang_vel_z = (
            -yaw_abs,
            yaw_abs,
        )

    def _apply_observation_settings(self, cfg: dict) -> None:
        self.observations.policy.history_length = int(
            cfg["history_length"]
        )

        # 지형 높이정보 제거
        if not bool(cfg["use_height_scan"]):
            if hasattr(self.observations.policy, "height_scan"):
                self.observations.policy.height_scan = None

        # 실기(Go1)에 몸통 선속도 측정이 없음 → policy(=student 배포) 그룹에서
        # 제거. teacher/critic 은 Go1LabPrivilegedObsCfg.base_lin_vel 로 계속 봄.
        self.observations.policy.base_lin_vel = None

        privileged_cfg = cfg["privileged"]

        if not bool(privileged_cfg["enabled"]):
            raise ValueError(
                "Phase 1 requires privileged observation for "
                "Phase 2 warm-start dimension compatibility."
            )

        # privileged observation group 설정 객체를 생성
        self.observations.privileged_obs = Go1LabPrivilegedObsCfg()
        # [FL, FR, RL, RR, injured_flag, L, friction, lin_vel(3)]

        # (μ 추정 경로 제거됨 — mu_noise_std 노이즈 주입도 함께 폐기.
        #  μ 채널은 차원 호환용으로 privileged 에 남되 깨끗한 GT 그대로 둔다.
        #  근거: μ 는 antalgic 보행에서 비식별 + 정책 민감도 ~1% 실측,
        #  주장은 'μ 강건성'으로 전환 — test/mu_robustness_report.py)

        # 부상 전 nominal 기준의 calf 관절각 4차원 추가
        if bool(cfg["use_calf_pos_nominal_rel"]):
            self.observations.policy.calf_pos_abs = ObsTerm(
                func=mdp.calf_pos_nominal_rel
            )
        else:
            self.observations.policy.calf_pos_abs = None

        # RLS 부목 길이 추정 채널 [L̂_norm, √P_norm] (2차원).
        # rls 블록이 있으면 live 갱신 (mdp/rls.py — 착지 등식 + 토크 게이트),
        # 없으면 prior 상수 (차원 예약만). healthy phase 는 부상 env 가 없어
        # live 여도 prior 에 머무르므로 두 경우가 동일하다.
        if bool(cfg["use_rls_estimate"]):
            self.observations.policy.rls_estimate = ObsTerm(
                func=mdp.rls_estimate
            )
            if "rls" in cfg:
                self.rls_params = {
                    "torque_gate_nm": float(cfg["rls"]["torque_gate_nm"]),
                    "foot_stance_n": float(cfg["rls"]["foot_stance_n"]),
                    "update_stride": int(cfg["rls"]["update_stride"]),
                    "meas_noise_std": float(cfg["rls"]["meas_noise_std"]),
                    "innovation_gate_m": float(cfg["rls"]["innovation_gate_m"]),
                    "min_axis_coef": float(cfg["rls"]["min_axis_coef"]),
                }
        
    def _apply_domain_randomization_settings(self, cfg) -> None:
        
        if not bool(cfg["enabled"]):
            return

        from isaaclab.utils.noise import GaussianNoiseCfg

        friction_cfg = cfg["ground_friction"]

        if (
            bool(friction_cfg["enabled"])
            and self.events.physics_material is not None
        ):
            self.events.physics_material.params[
                "static_friction_range"
            ] = tuple(float(v) for v in friction_cfg["static_range"])

            self.events.physics_material.params[
                "dynamic_friction_range"
            ] = tuple(float(v) for v in friction_cfg["dynamic_range"])

        mass_cfg = cfg["robot_mass"]

        # add_base_mass를 다시 생성
        if bool(mass_cfg["enabled"]):
            scale_range = tuple(
                float(v) for v in mass_cfg["scale_range"]
            )

            self.events.add_base_mass = EventTerm(
                func=mdp_base.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=str(mass_cfg["body_name"]),
                    ),
                    "mass_distribution_params": scale_range,
                    "operation": "scale",
                    "distribution": "uniform",
                    "recompute_inertia": True,
                },
            )

        push_cfg = cfg["random_push"]

        if bool(push_cfg["enabled"]):
            self.events.push_robot = EventTerm(
                func=mdp_base.push_by_setting_velocity,
                mode="interval",
                interval_range_s=tuple(
                    float(v) for v in push_cfg["interval_range_s"]
                ),
                params={
                    "velocity_range": {
                        "x": tuple(
                            float(v)
                            for v in push_cfg["velocity_x_range"]
                        ),
                        "y": tuple(
                            float(v)
                            for v in push_cfg["velocity_y_range"]
                        ),
                    }
                },
            )

        noise_cfg = cfg["observation_noise"]

        if bool(noise_cfg["enabled"]):
            noise_mapping = {
                "joint_pos": "joint_pos_std",
                "joint_vel": "joint_vel_std",
                "base_ang_vel": "base_ang_vel_std",
                "projected_gravity": "projected_gravity_std",
                "base_lin_vel": "base_lin_vel_std",
                # live RLS 채널은 sim 에서 오라클급이므로 실기 추정 오차만큼
                # 노이즈를 얹는다 (yaml 키가 없으면 노이즈 없이 유지)
                "rls_estimate": "rls_estimate_std",
            }

            for term_name, yaml_key in noise_mapping.items():
                term = getattr(
                    self.observations.policy,
                    term_name,
                    None,
                )

                if term is not None and yaml_key in noise_cfg:
                    term.noise = GaussianNoiseCfg(
                        mean=0.0,
                        std=float(noise_cfg[yaml_key]),
                    )

    def _apply_symmetric_balance_rewards(self, cfg: dict) -> None:
        reward_names = (
            "contact_force_asymmetry",
            "duty_factor_asymmetry",
            "diagonal_load_asymmetry",
            "front_rear_load_distribution",
            "trot_sync",
        )

        if not bool(cfg["enabled"]):
            for reward_name in reward_names:
                setattr(self.rewards, reward_name, None)
            return
        
        # phase1 balance reward를 사용한다면
        contact_cfg = cfg["contact_force_asymmetry"]

        self.rewards.contact_force_asymmetry = RewTerm(
            func=mdp.penalize_contact_force_asymmetry,
            weight=float(contact_cfg["weight"]),
            params={
                "ramp_duration_steps": int(
                    contact_cfg["ramp_duration_steps"]
                ),
            },
        )

        duty_cfg = cfg["duty_factor_asymmetry"]

        self.rewards.duty_factor_asymmetry = RewTerm(
            func=mdp.penalize_duty_factor_asymmetry,
            weight=float(duty_cfg["weight"]),
            params={
                "contact_threshold": float(
                    duty_cfg["contact_threshold"]
                ),
                "ramp_duration_steps": int(
                    duty_cfg["ramp_duration_steps"]
                ),
            },
        )

        diagonal_cfg = cfg["diagonal_load_asymmetry"]

        self.rewards.diagonal_load_asymmetry = RewTerm(
            func=mdp.penalize_diagonal_load_asymmetry,
            weight=float(diagonal_cfg["weight"]),
            params={
                "ramp_duration_steps": int(
                    diagonal_cfg["ramp_duration_steps"]
                ),
            },
        )

        front_rear_cfg = cfg[
            "front_rear_load_distribution"
        ]

        self.rewards.front_rear_load_distribution = RewTerm(
            func=mdp.penalize_front_rear_load_distribution,
            weight=float(front_rear_cfg["weight"]),
            params={
                "target_front_fraction": float(
                    front_rear_cfg["target_front_fraction"]
                ),
                "tolerance": float(
                    front_rear_cfg["tolerance"]
                ),
                "ramp_duration_steps": int(
                    front_rear_cfg["ramp_duration_steps"]
                ),
            },
        )

        trot_cfg = cfg["trot_synchronization"]

        self.rewards.trot_sync = RewTerm(
            func=mdp.reward_trot_synchronization,
            weight=float(trot_cfg["weight"]),
            params={
                "ramp_duration_steps": int(
                    trot_cfg["ramp_duration_steps"]
                ),
            },
        )

    def _apply_injury_reward_settings(self, cfg: dict, peg_leg_cfg) -> None:
        if not bool(cfg["enabled"]):
            return

        pain_cfg = cfg["penalty_pain"]

        if pain_cfg["enabled"]:
            self.rewards.penalty_pain = RewTerm(
                func=mdp.penalty_pain,
                weight=float(pain_cfg["weight"]),
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_name": "contact_forces",
                    "failure_force_threshold": float(pain_cfg["failure_force_threshold"]),
                    "pain_scale": float(pain_cfg["pain_scale"]),
                    
                    "max_exp_argument": float(pain_cfg["max_exp_argument"]),
                    "max_penalty": float(pain_cfg["max_penalty"]),
                    
                    "base_contact_cost": float(pain_cfg["base_contact_cost"]),
                    "contact_detect_threshold": float(pain_cfg["contact_detect_threshold"]),
                    "include_calf": bool(pain_cfg["include_calf"]),
                    # 부목을 거친 하중도 (감쇠된) 통증원 — antalgic 상한 형성
                    "include_splint": bool(pain_cfg["include_splint"]),
                    "splint_attenuation": float(pain_cfg["splint_attenuation"]),
                },
            )
        else:
            self.rewards.penalty_pain = None

        splint_range = peg_leg_cfg["splint_length_range"]
        splint_min, splint_max = min(splint_range), max(splint_range)
        # 부목 모델 v2 의 심각도: nominal leg reach(≈0.31 m)에서 멀수록 어렵다.
        # 긴 부목(=키다리)일수록 비대칭이 커지므로 severe=최대 길이, mild=최소 길이.
        severe_len, mild_len = splint_max, splint_min

        # 부상 다리를 사용하지 않는 것에 대한 패널티 (부목 끝단이 지면에서 받는 힘 측정)
        force_cfg = cfg["injured_limb_force_nonuse"]

        if force_cfg['enabled']:
            self.rewards.injured_limb_force_nonuse = RewTerm(
                func=mdp.penalize_injured_limb_force_nonuse, # 부상 다리의 평균 접촉력이 최소 목표보다 부족한지를 계산하는 함수
                weight=float(force_cfg["weight"]),
                params={
                    "sensor_name": "contact_forces",
                    "severe_splint_length": severe_len,
                    "mild_splint_length": mild_len,
                    "min_force_severe": float(force_cfg['min_force_severe']),
                    "min_force_mild": float(force_cfg['min_force_mild']),
                    "front_leg_multiplier": float(force_cfg["front_leg_multiplier"]),
                    "rear_leg_multiplier": float(force_cfg["rear_leg_multiplier"]),    
                    "ema_alpha": float(force_cfg["ema_alpha"]),
                    "ramp_start_steps": int(force_cfg["ramp_start_steps"]),
                    "ramp_duration_steps": int(force_cfg["ramp_duration_steps"]),
                    "include_calf": bool(force_cfg["include_calf"]),
                },
            )
        else:
            self.rewards.injured_limb_force_nonuse = None

        duty_nonuse_cfg = cfg[
            "injured_limb_load_duty_nonuse"
        ]
        
        # 부상 다리가 일정 시간 동안 하중을 거의 전혀 받지 않는 상태, 즉 부상 다리를 계속 들고 3족 보행하는 것을 막는 페널티
        if duty_nonuse_cfg["enabled"]:
            self.rewards.injured_limb_load_duty_nonuse = RewTerm(
                func=mdp.penalize_injured_limb_load_duty_nonuse,
                weight=float(duty_nonuse_cfg["weight"]),
                params={
                    "sensor_name": "contact_forces",
                    "load_contact_threshold": float(
                        duty_nonuse_cfg["load_contact_threshold"]
                    ),
                    
                    "severe_splint_length": severe_len,
                    "mild_splint_length": mild_len,

                    "min_duty_severe": float(
                        duty_nonuse_cfg["min_duty_severe"]
                    ),
                    "min_duty_mild": float(
                        duty_nonuse_cfg["min_duty_mild"]
                    ),
                    "front_leg_multiplier": float(
                        duty_nonuse_cfg["front_leg_multiplier"]
                    ),
                    "rear_leg_multiplier": float(
                        duty_nonuse_cfg["rear_leg_multiplier"]
                    ),
                    "ema_alpha": float(
                        duty_nonuse_cfg["ema_alpha"]
                    ),
                    "ramp_duration_steps": int(
                        duty_nonuse_cfg["ramp_duration_steps"]
                    ),
                },
            )
        else:
           self.rewards.injured_limb_load_duty_nonuse = None

        
    def _apply_reward_settings(self, cfg, peg_leg_cfg) -> None:
        
        task_cfg = cfg["task"]

        if bool(task_cfg["enabled"]):
            self.rewards.track_lin_vel_xy_exp.weight = float(
                task_cfg["track_linear_velocity_weight"]
            )
            self.rewards.track_ang_vel_z_exp.weight = float(
                task_cfg["track_angular_velocity_weight"]
            )
            self.rewards.lin_vel_z_l2.weight = float(
                task_cfg["linear_velocity_z_weight"]
            )

        gait_cfg = cfg["gait_tuning"]

        if bool(gait_cfg["enabled"]):
            self.rewards.feet_air_time.weight = float(
                gait_cfg["feet_air_time_weight"]
            )
            self.rewards.action_rate_l2.weight = float(
                gait_cfg["action_rate_weight"]
            )
            self.rewards.ang_vel_xy_l2.weight = float(
                gait_cfg["angular_velocity_xy_weight"]
            )

        energy_cfg = cfg["energy"]

        self.rewards.dof_torques_l2.weight *= float(
            energy_cfg["torque_penalty_scale"]
        )
        self.rewards.dof_acc_l2.weight *= float(
            energy_cfg["dof_acc_penalty_scale"]
        )
        self.rewards.action_rate_l2.weight *= float(
            energy_cfg["action_rate_penalty_scale"]
        )

        posture_cfg = cfg["posture"]

        self.rewards.flat_orientation_l2.weight = float(
            posture_cfg["flat_orientation_weight"]
        )

        self.rewards.base_height = RewTerm(
            func=mdp_base.base_height_l2,
            weight=float(posture_cfg["base_height_weight"]),
            params={
                "target_height": float(
                    posture_cfg["base_height_target"]
                ),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        
        calf_contact_cfg = cfg['calf_contact']
        
        if bool(calf_contact_cfg['enabled']):
            self.rewards.intact_calf_contact = RewTerm(
                func=mdp.penalize_knee_shin_contact,
                weight=float(calf_contact_cfg["weight"]),
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sensor_name": "contact_forces",
                    "force_threshold": float(calf_contact_cfg["force_threshold"]),
                    "max_overload": float(calf_contact_cfg["max_overload"]),
                    "use_z_only": bool(calf_contact_cfg["use_z_only"]),
                },
            )
        else:
            self.rewards.intact_calf_contact = None


        alive_cfg = cfg["survival_bonus"]
        
        if alive_cfg['enabled']:
            # alive bonus
            self.rewards.survival_bonus = RewTerm(
                func=mdp_base.is_alive,
                weight=float(alive_cfg["weight"]),
            )

        joint_mirror_cfg = cfg["joint_mirror_symmetry"]
        
        if joint_mirror_cfg['enabled']:
            self.rewards.joint_mirror_symmetry = RewTerm(
                func=mdp.penalize_joint_mirror_asymmetry,
                weight=float(joint_mirror_cfg['weight']),
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
        else:
            self.rewards.joint_mirror_symmetry = None
        
        
        # symmetric for Normal Walking 
        self._apply_symmetric_balance_rewards(cfg["symmetric"])
        
        # injury reward
        self._apply_injury_reward_settings(cfg["injury"], peg_leg_cfg)
                    
    def _apply_termination_settings(self, cfg) -> None:
        root_cfg = cfg["root_too_low"]
        if bool(root_cfg["enabled"]):
            self.terminations.root_too_low = DoneTerm(
                func=mdp_base.root_height_below_minimum,
                params={
                    "minimum_height": float(
                        root_cfg["minimum_height"]
                    ),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
        
        orientation_cfg = cfg["bad_orientation"]
        if bool(orientation_cfg["enabled"]):
            self.terminations.bad_orientation = DoneTerm(
                func=mdp_base.bad_orientation,
                params={
                    "limit_angle": float(
                        orientation_cfg["limit_angle"]
                    )
                },
            )
        else:
            self.terminations.bad_orientation = None
        
    # domain Random에 의해 덮어짐
    def _apply_payload_settings(self, cfg) -> None:
        # 상속받은 기존 질량제거 추후 domain randomization에서 생성해줌
        self.events.add_base_mass = None
        self.events.front_payload_mass = None
        self.events.front_payload_com = None         
        
        if not bool(cfg["enabled"]):
            return
        
        # Go1의 trunk에 앞쪽에 무게추가
        from isaaclab.envs.mdp.events import (
            randomize_rigid_body_com, # Center of Mass를 변경
            randomize_rigid_body_mass, # Rigid body의 질량을 변경
        )
        
        front_payload_kg = float(cfg["front_payload_kg"])
        front_com_x_m = float(cfg["front_com_x_m"])
        front_com_z_m = float(cfg["front_com_z_m"])

        trunk_cfg = SceneEntityCfg(
            "robot",
            body_names="trunk",
        )

        # trunk에 고정 추가 질량 적용
        self.events.front_payload_mass = EventTerm(
            func=randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": trunk_cfg,
                "mass_distribution_params": (
                    front_payload_kg,
                    front_payload_kg,
                ),
                "operation": "add",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

        # trunk CoM 이동
        self.events.front_payload_com = EventTerm(
            func=randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": trunk_cfg,
                "com_range": {
                    "x": (
                        front_com_x_m,
                        front_com_x_m,
                    ),
                    "y": (0.0, 0.0),
                    "z": (
                        front_com_z_m,
                        front_com_z_m,
                    ),
                },
            },
        )
    
    
    def _apply_curriculum_settings(self, cfg: dict,  steps_per_iteration: int, target_leg: str, healthy_slots: int):
        if not cfg['enabled']:
            return
        
        leg_cfg = cfg["leg_probability"]
        splint_cfg = cfg["splint_length"]

        # 부상 확률 curriculum
        prob_initial = float(leg_cfg["initial"])
        prob_final = float(leg_cfg["final"])
        prob_iterations = int(leg_cfg["iterations"])

        # 부목 길이 curriculum
        initial_min = float(splint_cfg["initial_min"])
        initial_max = float(splint_cfg["initial_max"])
        final_min = float(splint_cfg["final_min"])
        final_max = float(splint_cfg["final_max"])
        splint_iterations = int(splint_cfg["iterations"])
        
        # 설정값 검증
        if not 0.0 <= prob_initial <= 1.0:
            raise ValueError(
                f"leg_probability.initial must be in [0, 1], got {prob_initial}"
            )

        if not 0.0 <= prob_final <= 1.0:
            raise ValueError(
                f"leg_probability.final must be in [0, 1], got {prob_final}"
            )

        if prob_iterations <= 0:
            raise ValueError(
                f"leg_probability.iterations must be positive, got {prob_iterations}"
            )

        if splint_iterations <= 0:
            raise ValueError(
                f"splint_length.iterations must be positive, got {splint_iterations}"
            )

        if initial_min > initial_max:
            raise ValueError(
                "splint_length.initial_min must be less than or equal to initial_max"
            )

        if final_min > final_max:
            raise ValueError(
                "splint_length.final_min must be less than or equal to final_max"
            )
        
        self.curriculum.peg_leg_difficulty = CurTerm(
            func=peg_leg_curriculum,
            params={
                # 부상 확률: 0.1 -> 0.5
                "prob_start": prob_initial,
                "prob_end": prob_final,
                "prob_ramp_steps": prob_iterations,

                # 부목 길이 상한: 0.33 -> 0.30
                "splint_start": initial_max,
                "splint_end": final_max,

                # 부목 길이 하한: 0.28 -> 0.20
                "splint_lo_start": initial_min,
                "splint_lo_end": final_min,

                "splint_ramp_steps": splint_iterations,
                "steps_per_iteration": steps_per_iteration,
                "target_leg": target_leg,
                "healthy_slots": healthy_slots,
            },
        )
            
    
    def _set_target_and_peg_leg_prob_for_eval(self, eval_peg_leg: str | None, target_leg: str, prob_peg_leg: float):
        """평가 모드가 있으면 학습용 target/prob 값을 평가용으로 덮어쓴다."""

        # train.py에서 호출하면 평가 override가 없으므로 학습값 유지
        if eval_peg_leg is None:
            return target_leg, prob_peg_leg

        eval_mode = eval_peg_leg.strip().lower()

        if eval_mode == "normal":
            return "normal", 0.0

        if eval_mode in {"fl", "fr", "rl", "rr"}:
            return eval_mode, 1.0

        if eval_mode == "balanced":
            return "balanced_random", 0.8

        raise ValueError(
            f"Unsupported eval peg leg: {eval_mode!r}"
        )

         
    def _apply_peg_leg_event_settings(self, cfg: dict, steps_per_iteration: int, attach: str, eval_peg_leg:str = None) -> None:
        # 이벤트 자체를 등록하지 않는 모드
        self.use_peg_leg = bool(cfg["enabled"])

        if not self.use_peg_leg:
            self.events.randomize_peg_leg_actuation = None
            self.events.enforce_peg_leg = None
            self.curriculum.peg_leg_difficulty = None
            # 부상 env 가 없으므로 grace period 불필요 (None 이면 step()에서 비교 불가)
            self.grace_steps = 0

            if eval_peg_leg not in (None, "normal"):
                raise ValueError(
                    f"peg_leg.enabled=false인 환경에서는 "
                    f"eval.peg_leg={eval_peg_leg!r}를 사용할 수 없습니다."
                )

            return

        self.grace_steps = int(cfg['grace_steps'])

        if self.grace_steps < 0:
            raise ValueError("grace_steps must be greater than or equal to zero")



        # 기본값은 antalgic.yaml의 학습 설정
        target_leg = str(cfg["target_leg"]).strip().lower()
        prob_peg_leg = max(0.0, min(1.0, float(cfg["prob_peg_leg"])))


        # test.py에서 평가값을 넘겼다면 평가용으로 덮어쓰기
        target_leg, prob_peg_leg = (
            self._set_target_and_peg_leg_prob_for_eval(
                eval_peg_leg=eval_peg_leg,
                target_leg=target_leg,
                prob_peg_leg=prob_peg_leg,
            )
        )

        splint_range = tuple(float(value) for value in cfg["splint_length_range"])
        foot_friction_range = tuple(float(value) for value in cfg["foot_friction_range"])
        injured_splint_friction_only = bool(cfg["injured_splint_friction_only"])

        # 부목길이
        splint_range = (min(splint_range), max(splint_range))

        # 부목 끝단 마찰 계수
        foot_friction_range = (min(foot_friction_range), max(foot_friction_range))

        # 에피소드가 reset될 때 부상 상태와 관련 버퍼 초기화
        # env._peg_leg_index 가 생성됨
        hip_torque_cfg = cfg["hip_torque"]
        hip_torque_scale = float(hip_torque_cfg["scale"])
        weaken_joints = str(hip_torque_cfg["weaken_joints"]).strip().lower()


        splint_actuator_cfg = cfg["splint_actuator"]
        splint_calf_stiffness = float(splint_actuator_cfg["stiffness"])
        splint_calf_damping = float(splint_actuator_cfg["damping"])

        healthy_slots = int(cfg["env_fixed_healthy_slots"])

        self.events.randomize_peg_leg_actuation = EventTerm(
            func=randomize_peg_leg_actuation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "prob_peg_leg": prob_peg_leg,
                "target_leg": target_leg,
                "prob_joint_disabled":  float(cfg["prob_joint_disabled"]),
                "splint_length_range": splint_range,
                "foot_friction_range": foot_friction_range,

                "injured_splint_friction_only": injured_splint_friction_only,

                # 부상 무릎 접기 각도 + 부목 부착 링크
                "fold_knee_angle": float(cfg["fold_knee_angle"]),
                "attach": attach,

                "hip_torque_scale": hip_torque_scale,
                "weaken_joints": weaken_joints,

                # 잠긴 calf 전용 PD (compliant 부목 무릎)
                "splint_calf_stiffness": splint_calf_stiffness,
                "splint_calf_damping": splint_calf_damping,

                # 정상 보행을 할 환경 구성 갯수
                "healthy_slots": healthy_slots,
            },
        )

        # 매 스텝 부상 관절의 고정 상태 유지
        self.events.enforce_peg_leg = EventTerm(
            func=enforce_peg_leg_constraints,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        
        # # ----- (B) Peg-leg 커리큘럼 -----
        # 학습 초기: 10% 부상, 거의 정상 길이(0.30m)
        # 학습 후기: 50% 부상, 짧은 부목(0.20m)
        self._apply_curriculum_settings(cfg['curriculum'], steps_per_iteration, target_leg, healthy_slots)
        
    
    def apply_environment_settings(self, settings: dict, steps_per_iteration: int, eval_peg_leg:str =None):
        # phase = str(settings["name"]).strip().lower()
        steps_per_iteration = int(steps_per_iteration)

        self.use_peg_leg_action_mask = bool(settings["use_peg_leg_action_mask"])

        # 부목 asset 은 전 phase 공통 (healthy 는 presence 전부-없음 상태로 사용)
        attach = self._apply_splint_asset_settings(settings["peg_leg"])
        # 부목 관절이 action/관측/리셋에 새어들지 않게 12개 다리 관절로 스코핑
        self._apply_joint_scope_settings()

        self._apply_peg_leg_event_settings(settings['peg_leg'], steps_per_iteration, attach=attach, eval_peg_leg=eval_peg_leg)
        self._apply_actuator_settings(settings["actuator"])
        self._apply_simulation_settings(settings["simulation"])
        self._apply_command_settings(settings["command"])
        self._apply_observation_settings(settings["observation"])

        # payload가 기존 add_base_mass를 정리한 후,
        # domain randomization이 새 mass event를 등록하도록 이 순서 유지
        self._apply_payload_settings(settings["payload"])
        self._apply_domain_randomization_settings(settings["domain_randomization"])
        
        self._apply_reward_settings(settings["reward"], settings['peg_leg'])
        self._apply_termination_settings(settings["termination"])
    