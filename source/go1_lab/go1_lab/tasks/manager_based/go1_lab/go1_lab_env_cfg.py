# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go1 Lab 환경 설정 - 표준 Go1 Rough 환경을 상속받아 의족(Peg Leg) 시나리오 랜덤화 추가."""

import os

from isaaclab.actuators import DCMotorCfg
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
    randomize_peg_leg_actuation,
    peg_leg_curriculum,
    enforce_peg_leg_constraints,
)


##
# Environment configuration
##

@configclass
class Go1LabPrivilegedObsCfg(ObsGroup):
    # Teacher에게 제공할 privileged observation 정의
    #TODO: 이거 mdp에서 어떻게 특권정보 가져오는지 확인
    peg_leg_one_index = ObsTerm(func=mdp.peg_leg_one_hot) # 부상 다리
    peg_leg_splint_length = ObsTerm(func=mdp.peg_leg_splint_length) # 부목 길이 
    peg_leg_foot_friction = ObsTerm(func=mdp.peg_leg_foot_friction) # 발 마찰 계수

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

    def __post_init__(self):
        super().__post_init__()

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
                joint_names_expr=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
                effort_limit=effort_limit,
                saturation_effort=effort_limit,
                velocity_limit=float(pd_cfg["velocity_limit"]),
                stiffness=float(pd_cfg["kp"]),
                damping=float(pd_cfg["kd"]),
                friction=float(pd_cfg["friction"]),
            )
        }
        
    def _apply_simulation_settings(self, cfg: dict) -> None:
        self.scene.replicate_physics = bool(cfg["replicate_physics"])
        self.scene.clone_in_fabric = bool(cfg["clone_in_fabric"])

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

        privileged_cfg = cfg["privileged"]

        if not bool(privileged_cfg["enabled"]):
            raise ValueError(
                "Phase 1 requires privileged observation for "
                "Phase 2 warm-start dimension compatibility."
            )

        # privileged observation group 설정 객체를 생성
        self.observations.privileged_obs = Go1LabPrivilegedObsCfg()
        # [FL, FR, RL, RR, injured_flag]
        
        # 부상 전 nominal 기준의 calf 관절각 4차원 추가
        if bool(cfg["use_calf_pos_nominal_rel"]):
            self.observations.policy.calf_pos_abs = ObsTerm(
                func=mdp.calf_pos_nominal_rel
            )
        else:
            self.observations.policy.calf_pos_abs = None
        
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
            }

            for term_name, yaml_key in noise_mapping.items():
                term = getattr(
                    self.observations.policy,
                    term_name,
                    None,
                )

                if term is not None:
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
                },
            )
        else:
            self.rewards.penalty_pain = None
        
    
        
        splint_min, splint_max = peg_leg_cfg['splint_length_range'][0], peg_leg_cfg['splint_length_range'][1]
        
        # 부상 다리를 사용하지 않는 것에 대한 패널티 (그걸 지면으로 부터 받는 힘을 측정)
        force_cfg = cfg["injured_limb_force_nonuse"]
        
        if force_cfg['enabled']:
            self.rewards.injured_limb_force_nonuse = RewTerm(
                func=mdp.penalize_injured_limb_force_nonuse, # 부상 다리의 평균 접촉력이 최소 목표보다 부족한지를 계산하는 함수
                weight=float(force_cfg["weight"]),
                params={
                    "sensor_name": "contact_forces",
                    "severe_splint_length": splint_min,
                    "mild_splint_length": splint_max,
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
            self.rewards.injured_limb_force_nonuse = True
                    
            
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
                    
                    "severe_splint_length": splint_min,
                    "mild_splint_length": splint_max,
                    
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
                    "use_z_only": bool(calf_contact_cfg.get["use_z_only"]),
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
            self.terminations.base_contact = None
        
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

         
    def _apply_peg_leg_event_settings(self, cfg: dict, steps_per_iteration: int, eval_peg_leg:str = None) -> None:
        # 이벤트 자체를 등록하지 않는 모드
        self.use_peg_leg = bool(cfg["enabled"])
        
        if not self.use_peg_leg:
            self.events.randomize_peg_leg_actuation = None
            self.events.enforce_peg_leg = None
            self.curriculum.peg_leg_difficulty = None
            
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
        injured_foot_friction_only = bool(cfg["injured_foot_friction_only"])
        
        # 부목길이
        splint_range = (min(splint_range), max(splint_range))
        
        # 발 마찰 계수
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
                
                "injured_foot_friction_only": injured_foot_friction_only,
                
                "hip_torque_scale": hip_torque_scale,
                "weaken_joints": weaken_joints,
                
                # 부상 calf 전용 PD
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
        
        self._apply_peg_leg_event_settings(settings['peg_leg'], steps_per_iteration, eval_peg_leg=eval_peg_leg)
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
    