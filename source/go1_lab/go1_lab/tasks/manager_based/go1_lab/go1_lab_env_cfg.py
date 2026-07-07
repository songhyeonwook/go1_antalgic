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
    """Teacher 전용 privileged observation (2차원).

    Student LSTM 출력 target:
    [0] 부상 상태 index: 0=정상, 1=FL, 2=FR, 3=RL, 4=RR
    [1] 부목 등가 길이 (m): Go1 역기구학 기반
    [2] 발 마찰 계수
    """

    peg_leg_index = ObsTerm(func=mdp.peg_leg_index)
    peg_leg_splint_length = ObsTerm(func=mdp.peg_leg_splint_length)
    peg_leg_foot_friction = ObsTerm(func=mdp.peg_leg_foot_friction)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go1LabEnvCfg(UnitreeGo1RoughEnvCfg):
    """Go1 Lab 환경 설정.

    3-phase 학습 파이프라인:
      Phase 1 (GO1_PHASE=healthy): 정상 보행 pretrain
      Phase 2 (GO1_PHASE=teacher): peg-leg 환경 + privileged obs → Teacher PPO
      Phase 3 (GO1_PHASE=student): Teacher checkpoint 로드 → Student distill
    """

    def __post_init__(self):
        super().__post_init__()

        # =================================================================
        # 1. Phase & Eval Mode
        # =================================================================
        # GO1_PHASE    : "healthy" | "teacher" | "student"
        # GO1_EVAL_MODE: "random"(학습) | "normal" | "fl_peg" | "fr_peg" | "rl_peg" | "rr_peg"
        phase = os.getenv("GO1_PHASE", "healthy").strip().lower()
        eval_mode = os.getenv("GO1_EVAL_MODE", "random").strip().lower()
        enable_peg_leg = phase in {"teacher", "student"}
        self.use_peg_leg_action_mask = enable_peg_leg

        # =================================================================
        # 1b. Actuator: PD position control (paper §4.3 + Go1 sim-to-real std)
        # =================================================================
        # The inherited Go1 uses ActuatorNetMLP (a learned torque model). Every
        # canonical Go1 sim-to-real pipeline (legged_gym / walk-these-ways / RMA /
        # actuator-fault studies) instead uses PD position control (Kp~20, Kd~0.5,
        # 50 Hz) — which also matches how the real Go1 is deployed (position
        # targets with settable Kp/Kd) and the paper's §4.3. The learned actuator
        # net is softer/laggier than a Kp20 deployment, so it cannot HOLD a loaded
        # injured stance -> the trunk collapses (root_too_low). Switch to PD so the
        # built-in proportional stiffness stabilises the antalgic stance.
        # Enable with GO1_PD_ACTUATOR=1.
        if os.getenv("GO1_PD_ACTUATOR", "0").strip().lower() in {"1", "true", "yes", "on"} and hasattr(self.scene, "robot"):
            _kp = float(os.getenv("GO1_PD_KP", "20.0"))
            _kd = float(os.getenv("GO1_PD_KD", "0.5"))
            _eff = float(os.getenv("GO1_PD_EFFORT_LIMIT", "23.7"))
            self.scene.robot.actuators = {
                "base_legs": DCMotorCfg(
                    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                    effort_limit=_eff,
                    saturation_effort=_eff,
                    velocity_limit=30.0,
                    stiffness=_kp,
                    damping=_kd,
                    friction=0.0,
                )
            }

        # PhysX GPU broad-phase capacity: the default gpu_total_aggregate_pairs_
        # capacity (2**21) overflows with 4096 envs under the stiffer PD actuator
        # ("needs to increase ... otherwise the simulation will miss interactions"
        # -> missed contacts -> instability -> NaN). Raise the pair buffers (cheap
        # on a 96GB GPU) to keep the contact solver consistent.
        if hasattr(self, "sim") and hasattr(self.sim, "physx"):
            self.sim.physx.gpu_total_aggregate_pairs_capacity = max(
                int(getattr(self.sim.physx, "gpu_total_aggregate_pairs_capacity", 0)), 2**23
            )
            self.sim.physx.gpu_found_lost_pairs_capacity = max(
                int(getattr(self.sim.physx, "gpu_found_lost_pairs_capacity", 0)), 2**23
            )

        # Command-velocity range override (faithful viability via the TASK, not a
        # use-floor): commanding FAST forward makes 3-leg non-use unable to track
        # the velocity / unstable, so the agent must bear load on the peg to keep
        # up -> the antalgic partial-loading can emerge from pure pain+energy+task.
        # GO1_CMD_VX_MIN/MAX set the forward range; GO1_CMD_VY_ABS / GO1_CMD_YAW_ABS
        # shrink lateral/turn so the task is dominated by forward speed.
        if hasattr(self, "commands") and hasattr(self.commands, "base_velocity"):
            _r = self.commands.base_velocity.ranges
            _vxmin, _vxmax = os.getenv("GO1_CMD_VX_MIN"), os.getenv("GO1_CMD_VX_MAX")
            if _vxmin and _vxmax:
                _r.lin_vel_x = (float(_vxmin), float(_vxmax))
            _vy = os.getenv("GO1_CMD_VY_ABS")
            if _vy:
                _r.lin_vel_y = (-float(_vy), float(_vy))
            _yaw = os.getenv("GO1_CMD_YAW_ABS")
            if _yaw:
                _r.ang_vel_z = (-float(_yaw), float(_yaw))

        # =================================================================
        # 2. Scene & Observation
        # =================================================================
        if hasattr(self, "scene"):
            # ⚠️ 주의: Healthy/Peg-leg 여부와 상관없이 무조건 False로 두어야 합니다.
            # replicate_physics=True 로 설정할 경우, Isaac Sim PhysX 엔진에서 발바닥과 몸통의 
            # 접촉(Contact) 센서(ContactSensor)를 초기화하지 못하고 튕기는 치명적 에러가 발생합니다.
            if hasattr(self.scene, "replicate_physics"):
                self.scene.replicate_physics = False
            if hasattr(self.scene, "clone_in_fabric"):
                self.scene.clone_in_fabric = False

        # ⚠️ history_length 은 모든 Phase에서 1로 유지합니다.
        # LSTM policy 가 시계열 메모리를 담당하므로 입력 frame stacking 은 불필요합니다.
        # 특히 Phase 3 Distillation 에서는 Teacher 가 H=1 로 학습된 체크포인트를
        # 로드하는데, H>1 로 설정하면 Teacher LSTM 입력 차원이 달라져 로드 실패합니다.
        self.observations.policy.history_length = 1

        # Proprioception-only policy observation (GO1_PROPRIO_ONLY=1). The paper
        # is explicit: proprioception-only R^48, "no exteroceptive sensing"
        # (abstract, contribution iii, §4.3). The 187-dim height_scan both
        # violates that and bloats the obs (235→48), which made a clean
        # left/right-symmetric gait hard to converge (open-loop mirror
        # equivariance was tight but the closed-loop limit cycle still leaned).
        # Removing it restores the paper's blind locomotion structure and makes
        # the mirror transform trivial (no height grid). The privileged_obs group
        # (teacher only) is unaffected.
        if os.getenv("GO1_PROPRIO_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}:
            if hasattr(self.observations.policy, "height_scan"):
                self.observations.policy.height_scan = None

        # Optional flat terrain for TRAINING (GO1_FLAT_TERRAIN=1). The paper is
        # flat / proprioception-only; training on ROUGH terrain + height_scan
        # introduces a systematic left-right bias (probe on flat is symmetric,
        # but rough-terrain training consistently commits to FR). Flattening
        # makes the height_scan constant → removes that bias source.
        if os.getenv("GO1_FLAT_TERRAIN", "0").strip().lower() in {"1", "true", "yes", "on"}:
            if hasattr(self.scene, "terrain"):
                _terr = self.scene.terrain
                if hasattr(_terr, "terrain_type"):
                    _terr.terrain_type = "plane"
                if hasattr(_terr, "terrain_generator"):
                    _terr.terrain_generator = None
            # terrain_levels curriculum needs a terrain_generator; disable on flat.
            if hasattr(self, "curriculum") and hasattr(self.curriculum, "terrain_levels"):
                self.curriculum.terrain_levels = None

        # Contact compliance: soften the foot-ground contact so the RIGID peg's
        # sharp contact impulse is ABSORBED. The rigid splinted leg (calf τ=0) can
        # then bear partial load without the sharp impact destabilising the trunk
        # (the loading→sink failure mode). Models a rubber peg tip / compliant
        # ground — keeps the splint rigid (intentional). GO1_CONTACT_COMPLIANCE_
        # STIFFNESS>0 enables (lower = softer); default 0 = off (exact prior physics).
        _cc = float(os.getenv("GO1_CONTACT_COMPLIANCE_STIFFNESS", "0.0"))
        if _cc > 0.0:
            _ccd = float(os.getenv("GO1_CONTACT_COMPLIANCE_DAMPING", "2000.0"))
            _mats = [getattr(self.sim, "physics_material", None)]
            if hasattr(self.scene, "terrain"):
                _mats.append(getattr(self.scene.terrain, "physics_material", None))
            for _mat in _mats:
                if _mat is not None and hasattr(_mat, "compliant_contact_stiffness"):
                    _mat.compliant_contact_stiffness = _cc
                    _mat.compliant_contact_damping = _ccd

        # 모든 Phase에서 privileged obs 등록 (Phase 1 → Phase 2 warm-start 호환성 보장)
        # Phase 1(healthy)에서는 값이 [0, 0, 1.0] (정상, 부목 없음, 기본 마찰) 로 고정되어
        # Teacher(Phase 2) 와 동일한 observation dim 을 유지합니다.
        if hasattr(self, "observations"):
            self.observations.privileged_obs = Go1LabPrivilegedObsCfg()
            # ONE-HOT injury encoding (GO1_INJURY_ONEHOT=1). A scalar index (0..4)
            # is a weak conditioning signal: the teacher actor cannot cleanly learn
            # 5 distinct per-leg antalgic gaits from one continuous value and
            # collapses to its dominant mode (loads FL, abandons FR/RL/RR). A
            # one-hot [FL,FR,RL,RR,injured_flag] makes per-leg conditioning LINEAR
            # (RMA-standard for discrete privileged factors), so the teacher can
            # produce a distinct antalgic response for every injury location.
            # Changes privileged dim 3->7, so Phase-1 must be retrained with it.
            if os.getenv("GO1_INJURY_ONEHOT", "0").strip().lower() in {"1", "true", "yes", "on"}:
                self.observations.privileged_obs.peg_leg_index = ObsTerm(func=mdp.peg_leg_one_hot)

        # =================================================================
        # [NEW] 생물학적 무게 중심(CoM) 이동을 위한 Payload 추가
        # =================================================================
        # 실제 개(Dog)의 전방 치우친 무게 배분을 모사하기 위해 trunk에 payload를 추가하고 CoM을 전방 이동.
        # 이 설정을 통해 부상 시 뒷다리를 끄는(Dragging) RL의 꼼수를 물리적으로 원천 차단합니다.
        if hasattr(self, "events"):
            from isaaclab.envs.mdp.events import (
                randomize_rigid_body_mass,
                randomize_rigid_body_com,
            )

            # 기존 UnitreeGo1RoughEnvCfg에서 상속된 add_base_mass 무효화 (클래스 기반 이벤트인데 mode="startup"으로 설정되어 있어 에러 발생)
            if hasattr(self.events, "add_base_mass"):
                self.events.add_base_mass = None

            # Go1 trunk 앞쪽에 얹은 payload를 등가적으로 모델링합니다.
            # 기본값은 nominal Go1 기준 실험을 위해 비활성화합니다.
            # 필요하면 환경변수로 실험별 조정:
            #   GO1_FRONT_PAYLOAD_KG=0.0  -> payload 비활성화
            #   GO1_FRONT_PAYLOAD_KG=2.0 GO1_FRONT_COM_X_M=0.05
            front_payload_kg = float(os.getenv("GO1_FRONT_PAYLOAD_KG", "0.0"))
            front_com_x_m = float(os.getenv("GO1_FRONT_COM_X_M", "0.0"))
            front_com_z_m = float(os.getenv("GO1_FRONT_COM_Z_M", "0.0"))

            trunk_cfg = SceneEntityCfg("robot", body_names="trunk")
            if front_payload_kg > 0.0:
                self.events.front_payload_mass = EventTerm(
                    func=randomize_rigid_body_mass,
                    mode="startup",
                    params={
                        "asset_cfg": trunk_cfg,
                        "mass_distribution_params": (front_payload_kg, front_payload_kg),
                        "operation": "add",
                        "distribution": "uniform",
                        "recompute_inertia": True,
                    },
                )

            if abs(front_com_x_m) > 0.0 or abs(front_com_z_m) > 0.0:
                self.events.front_payload_com = EventTerm(
                    func=randomize_rigid_body_com,
                    mode="startup",
                    params={
                        "asset_cfg": trunk_cfg,
                        "com_range": {
                            "x": (front_com_x_m, front_com_x_m),
                            "y": (0.0, 0.0),
                            "z": (front_com_z_m, front_com_z_m),
                        },
                    },
                )

        # =================================================================
        # Domain randomization for sim-to-real (paper §4.8). Gated by
        # GO1_DOMAIN_RAND=1. Implements ground friction, robot mass, random
        # pushes, and Gaussian observation noise. Motor-strength and
        # action-latency randomization are added separately (custom — the
        # ActuatorNetMLP has no built-in gain randomization).
        # =================================================================
        if os.getenv("GO1_DOMAIN_RAND", "0").strip().lower() in {"1", "true", "yes", "on"} and hasattr(self, "events"):
            from isaaclab.utils.noise import GaussianNoiseCfg

            # (a) ground friction U(0.5, 1.5)
            if getattr(self.events, "physics_material", None) is not None:
                self.events.physics_material.params["static_friction_range"] = (0.5, 1.5)
                self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)

            # (b) robot mass U(0.85, 1.15) × nominal (scale the trunk mass;
            #     the inherited add_base_mass targets a "base" body absent on Go1)
            self.events.add_base_mass = EventTerm(
                func=mdp_base.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                    "mass_distribution_params": (0.85, 1.15),
                    "operation": "scale",
                    "distribution": "uniform",
                    "recompute_inertia": True,
                },
            )

            # (c) random base pushes for perturbation robustness
            self.events.push_robot = EventTerm(
                func=mdp_base.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(8.0, 12.0),
                params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
            )

            # (d) Gaussian observation noise: joint states N(0,0.02), IMU N(0,0.05)
            _pol = self.observations.policy
            for _term, _std in (
                ("joint_pos", 0.02),
                ("joint_vel", 0.02),
                ("base_ang_vel", 0.05),
                ("projected_gravity", 0.05),
                ("base_lin_vel", 0.05),
            ):
                if getattr(_pol, _term, None) is not None:
                    getattr(_pol, _term).noise = GaussianNoiseCfg(mean=0.0, std=_std)

        # Phase 1 paper baseline:
        # 이후 Phase 2/3의 "Normal" 기준이 되는 보행이므로 좌우 force/duty 대칭을
        # Phase 1부터 직접 맞춥니다. 완전히 순수한 Isaac Lab baseline이 필요하면
        # GO1_PHASE1_BALANCE_REWARDS=0 으로 끌 수 있습니다.
        use_phase1_balance_rewards = os.getenv("GO1_PHASE1_BALANCE_REWARDS", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if (not enable_peg_leg) and use_phase1_balance_rewards:
            self.rewards.contact_force_asymmetry = RewTerm(
                func=mdp.penalize_contact_force_asymmetry,
                weight=float(os.getenv("GO1_CONTACT_FORCE_ASYM_WEIGHT", "-0.003")),
                params={
                    "ramp_start_steps": int(os.getenv("GO1_CONTACT_FORCE_ASYM_RAMP_START_STEPS", "2000")),
                    "ramp_duration_steps": int(os.getenv("GO1_CONTACT_FORCE_ASYM_RAMP_STEPS", "6000")),
                },
            )
            self.rewards.duty_factor_asymmetry = RewTerm(
                func=mdp.penalize_duty_factor_asymmetry,
                weight=float(os.getenv("GO1_DUTY_FACTOR_ASYM_WEIGHT", "-0.015")),
                params={
                    "contact_threshold": float(os.getenv("GO1_PAIN_CONTACT_THRESHOLD_N", "1.0")),
                    "ramp_start_steps": int(os.getenv("GO1_DUTY_FACTOR_ASYM_RAMP_START_STEPS", "2000")),
                    "ramp_duration_steps": int(os.getenv("GO1_DUTY_FACTOR_ASYM_RAMP_STEPS", "6000")),
                },
            )
            self.rewards.diagonal_load_asymmetry = RewTerm(
                func=mdp.penalize_diagonal_load_asymmetry,
                weight=float(os.getenv("GO1_DIAGONAL_LOAD_ASYM_WEIGHT", "-0.0015")),
                params={
                    "ramp_start_steps": int(os.getenv("GO1_DIAGONAL_LOAD_ASYM_RAMP_START_STEPS", "2000")),
                    "ramp_duration_steps": int(os.getenv("GO1_DIAGONAL_LOAD_ASYM_RAMP_STEPS", "6000")),
                },
            )
            self.rewards.front_rear_load_distribution = RewTerm(
                func=mdp.penalize_front_rear_load_distribution,
                weight=float(os.getenv("GO1_FRONT_REAR_LOAD_DIST_WEIGHT", "-0.0005")),
                params={
                    "target_front_fraction": float(os.getenv("GO1_FRONT_LOAD_TARGET_FRACTION", "0.60")),
                    "tolerance": float(os.getenv("GO1_FRONT_LOAD_TARGET_TOLERANCE", "0.10")),
                    "ramp_start_steps": int(os.getenv("GO1_FRONT_REAR_LOAD_DIST_RAMP_START_STEPS", "2000")),
                    "ramp_duration_steps": int(os.getenv("GO1_FRONT_REAR_LOAD_DIST_RAMP_STEPS", "6000")),
                },
            )
            self.rewards.trot_sync = RewTerm(
                func=mdp.reward_trot_synchronization,
                weight=float(os.getenv("GO1_TROT_SYNC_WEIGHT", "0.03")),
                params={
                    "ramp_start_steps": int(os.getenv("GO1_TROT_SYNC_RAMP_START_STEPS", "2000")),
                    "ramp_duration_steps": int(os.getenv("GO1_TROT_SYNC_RAMP_STEPS", "6000")),
                },
            )
            self.rewards.duty_factor_deviation = RewTerm(
                func=mdp.penalize_duty_factor_deviation,
                weight=float(os.getenv("GO1_DUTY_FACTOR_DEVIATION_WEIGHT", "0.0")),
                params={
                    "contact_threshold": float(os.getenv("GO1_PAIN_CONTACT_THRESHOLD_N", "1.0")),
                    "target_contact_count": float(os.getenv("GO1_TARGET_CONTACT_COUNT", "2.0")),
                },
            )
            _target_duty = tuple(
                float(v)
                for v in os.getenv("GO1_LEG_DUTY_TARGETS", "0.55,0.55,0.50,0.50").split(",")
            )
            if len(_target_duty) != 4:
                raise ValueError(
                    "GO1_LEG_DUTY_TARGETS must contain four comma-separated values "
                    "for FL,FR,RL,RR."
                )
            self.rewards.leg_duty_factor_targets = RewTerm(
                func=mdp.penalize_leg_duty_factor_targets,
                weight=float(os.getenv("GO1_LEG_DUTY_TARGET_WEIGHT", "0.0")),
                params={
                    "contact_threshold": float(os.getenv("GO1_PAIN_CONTACT_THRESHOLD_N", "1.0")),
                    "target_duty": _target_duty,
                    "tolerance": float(os.getenv("GO1_LEG_DUTY_TARGET_TOLERANCE", "0.03")),
                    "ramp_start_steps": int(os.getenv("GO1_LEG_DUTY_TARGET_RAMP_START_STEPS", "2000")),
                    "ramp_duration_steps": int(os.getenv("GO1_LEG_DUTY_TARGET_RAMP_STEPS", "6000")),
                },
            )
            if hasattr(self.rewards, "feet_air_time"):
                self.rewards.feet_air_time.weight = float(
                    os.getenv("GO1_PHASE1_FEET_AIR_TIME_WEIGHT", str(self.rewards.feet_air_time.weight))
                )

        # Phase 1은 peg-leg curriculum/통증 보상/termination 수정 없이 종료합니다.
        if not enable_peg_leg:
            return

        # =================================================================
        # 3. 보상 설정 방침 (Phase 2/3 Trot 유지 튜닝)
        # =================================================================
        # 기본 UnitreeGo1RoughEnvCfg는 feet_air_time이 0.01로 매우 낮아,
        # 로봇이 발을 거의 떼지 않고 질질 끄는 Tapping(토핑) 보행에 빠지기 쉽습니다.
        # 인위적인 Trot Symmetry 보상 없이 자연스러운 대각선 보행(Trot)을 창발시키기 위해,
        # 체공 시간 보상과 흔들림 페널티의 밸런스를 바로잡습니다.

        use_phase2_gait_tuning = os.getenv("GO1_PHASE2_GAIT_TUNING", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if use_phase2_gait_tuning and hasattr(self.rewards, "feet_air_time"):
            # 발을 공중에 충분히 띄우도록 유도 (기존 0.01 -> 0.25)
            self.rewards.feet_air_time.weight = float(os.getenv("GO1_FEET_AIR_TIME_WEIGHT", "0.25"))

        if use_phase2_gait_tuning and hasattr(self.rewards, "action_rate_l2"):
            # 다리를 부들부들 떨며 짧게 터치하는 현상 억제 (기존 -0.01 -> -0.05)
            self.rewards.action_rate_l2.weight = float(os.getenv("GO1_ACTION_RATE_L2_WEIGHT", "-0.05"))

        if use_phase2_gait_tuning and hasattr(self.rewards, "ang_vel_xy_l2"):
            # 좌우 기우뚱거림(Rolling/Pitching) 강하게 억제
            # -> Pacing이나 토끼뜀을 방지하고 Trot으로 수렴하게 만듦 (기존 -0.05 -> -0.1)
            self.rewards.ang_vel_xy_l2.weight = float(os.getenv("GO1_ANG_VEL_XY_L2_WEIGHT", "-0.1"))

        # [Trot 대칭성 보상 추가]
        # 자연스러운 발현만으로는 좌우 대칭(SI 0%)에 도달하기까지 매우 오랜 학습이 필요하므로,
        # 정상(Phase 1) 보행에 한하여 좌우/대각 대칭성을 강제하는 보상을 활성화합니다.
        # (이 보상들은 mdp/rewards.py에 이미 "정상 다리에만 적용되도록" 구현되어 있습니다.)
        self.rewards.contact_force_asymmetry = RewTerm(
            func=mdp.penalize_contact_force_asymmetry,
            weight=float(os.getenv("GO1_CONTACT_FORCE_ASYM_WEIGHT", "-0.012")),
            params={
                "ramp_duration_steps": int(os.getenv("GO1_CONTACT_FORCE_ASYM_RAMP_STEPS", "6000")),
            },
        )
        self.rewards.duty_factor_asymmetry = RewTerm(
            func=mdp.penalize_duty_factor_asymmetry,
            weight=float(os.getenv("GO1_DUTY_FACTOR_ASYM_WEIGHT", "-0.04")),
            params={
                "contact_threshold": float(os.getenv("GO1_PAIN_CONTACT_THRESHOLD_N", "1.0")),
                "ramp_duration_steps": int(os.getenv("GO1_DUTY_FACTOR_ASYM_RAMP_STEPS", "6000")),
            },
        )
        self.rewards.diagonal_load_asymmetry = RewTerm(
            func=mdp.penalize_diagonal_load_asymmetry,
            weight=float(os.getenv("GO1_DIAGONAL_LOAD_ASYM_WEIGHT", "-0.006")),
            params={
                "ramp_duration_steps": int(os.getenv("GO1_DIAGONAL_LOAD_ASYM_RAMP_STEPS", "6000")),
            },
        )
        self.rewards.front_rear_load_distribution = RewTerm(
            func=mdp.penalize_front_rear_load_distribution,
            weight=float(os.getenv("GO1_FRONT_REAR_LOAD_DIST_WEIGHT", "-0.0025")),
            params={
                "target_front_fraction": float(os.getenv("GO1_FRONT_LOAD_TARGET_FRACTION", "0.55")),
                "tolerance": float(os.getenv("GO1_FRONT_LOAD_TARGET_TOLERANCE", "0.06")),
                "ramp_duration_steps": int(os.getenv("GO1_FRONT_REAR_LOAD_DIST_RAMP_STEPS", "6000")),
            },
        )
        self.rewards.trot_sync = RewTerm(
            func=mdp.reward_trot_synchronization,
            weight=float(os.getenv("GO1_TROT_SYNC_WEIGHT", "0.12")),
            params={
                "ramp_duration_steps": int(os.getenv("GO1_TROT_SYNC_RAMP_STEPS", "6000")),
            },
        )

        # =================================================================
        # 4. Phase 2/3 안정화 — 너무 이른 몸통 접촉 종료 방지
        # =================================================================
        # Phase 1에서는 Isaac Lab 기본 base_contact termination을 그대로 둡니다.
        # Phase 2/3에서는 peg-leg 적응 중 trunk가 살짝 닿는 상황이 너무 강한 종료 신호가 될 수 있어,
        # 대신 "몸통이 너무 낮아진 경우"만 종료하고, 자세는 base_height penalty로 유도합니다.
        self.rewards.base_height = RewTerm(
            func=mdp_base.base_height_l2,
            weight=float(os.getenv("GO1_BASE_HEIGHT_WEIGHT", "-0.3")),
            params={
                "target_height": float(os.getenv("GO1_BASE_HEIGHT_TARGET", "0.30")),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Flat-orientation (body-tilt ANGLE) penalty — keeps the trunk level so the
        # policy loads a SHORTER peg leg by squatting/leaning slightly rather than
        # ROLLING toward the injured corner until it tips over (bad_orientation).
        # Default 0 = unchanged; set GO1_FLAT_ORIENTATION_WEIGHT (e.g. -1.5) to enable.
        _flat_ori_w = float(os.getenv("GO1_FLAT_ORIENTATION_WEIGHT", "0.0"))
        if _flat_ori_w != 0.0:
            self.rewards.flat_orientation_l2 = RewTerm(
                func=mdp_base.flat_orientation_l2,
                weight=_flat_ori_w,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )

        # One-sided anti-collapse floor (stops the policy loading a SHORT peg by
        # sinking the trunk to the ground / root_too_low). Steep penalty BELOW the
        # floor only -> keeps the body up; loading then decreases with severity.
        # Default 0 = off; set GO1_BASE_HEIGHT_FLOOR_WEIGHT (e.g. -8.0) to enable.
        _floor_w = float(os.getenv("GO1_BASE_HEIGHT_FLOOR_WEIGHT", "0.0"))
        if _floor_w != 0.0:
            self.rewards.base_height_floor = RewTerm(
                func=mdp.penalize_base_height_floor,
                weight=_floor_w,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "height_floor": float(os.getenv("GO1_BASE_HEIGHT_FLOOR", "0.32")),
                },
            )

        self.terminations.root_too_low = DoneTerm(
            func=mdp_base.root_height_below_minimum,
            params={
                "minimum_height": float(os.getenv("GO1_ROOT_TOO_LOW_MIN_HEIGHT", "0.15")),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Paper §4.3 terminations (opt-in via GO1_STRICT_TERMINATIONS=1):
        #   terminate on base contact + base roll/pitch beyond ±0.8 rad.
        # Rationale ("task forces leg use"): without these, the policy can hold
        # the SHORTER peg leg off the ground (non-use) via tilted/degenerate
        # postures and still survive. Restoring them forces an upright, stable
        # 4-leg stance, so the peg must contact and bear load; pain then caps
        # that load near Fth(=10N) → ~69% GRF reduction (normal per-leg force
        # ≈ 32N, so 10N ≈ 69% reduction — matches the paper).
        strict_terminations = os.getenv("GO1_STRICT_TERMINATIONS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if strict_terminations:
            self.terminations.bad_orientation = DoneTerm(
                func=mdp_base.bad_orientation,
                params={"limit_angle": float(os.getenv("GO1_BAD_ORIENTATION_LIMIT", "0.8"))},
            )
            # keep the inherited base_contact termination (do NOT disable it)
        else:
            if hasattr(self.terminations, "base_contact"):
                self.terminations.base_contact = None

        # =================================================================
        # 5. Peg-leg 시나리오 (Phase 2/3 전용)
        # =================================================================
        # ----- 부상 확률/부목 길이 결정 -----
        prob_peg_leg = max(0.0, min(1.0, float(os.getenv("GO1_PROB_PEG_LEG", "0.5"))))
        splint_min = float(os.getenv("GO1_SPLINT_LENGTH_MIN", "0.20"))
        splint_max = float(os.getenv("GO1_SPLINT_LENGTH_MAX", "0.30"))
        splint_min, splint_max = min(splint_min, splint_max), max(splint_min, splint_max)
        foot_friction_min = float(os.getenv("GO1_FOOT_FRICTION_MIN", "0.4"))
        foot_friction_max = float(os.getenv("GO1_FOOT_FRICTION_MAX", "1.0"))
        foot_friction_min, foot_friction_max = (
            min(foot_friction_min, foot_friction_max),
            max(foot_friction_min, foot_friction_max),
        )
        target_leg = "random"

        # eval_mode 오버라이드 (평가 모드가 커리큘럼 설정보다 우선)
        if eval_mode == "normal":
            target_leg = "normal"
            prob_peg_leg = 0.0
        elif eval_mode in {"fl_peg", "fr_peg", "rl_peg", "rr_peg"}:
            target_leg = eval_mode.replace("_peg", "")
            prob_peg_leg = 1.0
        elif eval_mode == "balanced":
            # Balanced validation keeps Normal/FL/FR/RL/RR at 1:1:1:1:1,
            # but uses a random permutation so left/right injury conditions do
            # not stay tied to fixed env ids or terrain shards.
            target_leg = os.getenv("GO1_BALANCED_TARGET_MODE", "balanced_random").strip().lower()
            prob_peg_leg = 0.8

        # ----- (A) Peg-leg 리셋 이벤트 (에피소드 시작 시 1회) -----
        self.events.randomize_peg_leg_actuation = EventTerm(
            func=randomize_peg_leg_actuation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "prob_peg_leg": prob_peg_leg,
                "target_leg": target_leg,
                "prob_joint_disabled": 1.0,
                "actuation_mode": "locked",
                "splint_length_range": (splint_min, splint_max),
                "foot_friction_range": (foot_friction_min, foot_friction_max),
            },
        )

        # ----- (A-2) Peg-leg 강제 고정 이벤트 (매 스텝) -----
        # 관절이 굽혀지지 않도록 매 스텝마다 위치/속도를 강제 고정합니다.
        self.events.enforce_peg_leg = EventTerm(
            func=enforce_peg_leg_constraints,
            mode="interval",
            interval_range_s=(0.0, 0.0), 
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ----- (B) Peg-leg 커리큘럼 -----
        # 학습 초기: 10% 부상, 거의 정상 길이(0.30m)
        # 학습 후기: 50% 부상, 짧은 부목(0.20m)
        if os.getenv("GO1_USE_PEG_LEG_CURRICULUM", "1").strip().lower() in {"1", "true", "yes", "on"}:
            self.curriculum.peg_leg_difficulty = CurTerm(
                func=peg_leg_curriculum,
                params={
                    "prob_start": float(os.getenv("GO1_CURRICULUM_PROB_START", "0.1")),
                    "prob_end": prob_peg_leg,
                    "prob_ramp_steps": int(os.getenv("GO1_CURRICULUM_PROB_RAMP_STEPS", "5000")),
                    "splint_start": float(os.getenv("GO1_CURRICULUM_SPLINT_HI_START", "0.33")),
                    "splint_end": splint_max,
                    "splint_lo_start": float(os.getenv("GO1_CURRICULUM_SPLINT_LO_START", "0.28")),
                    "splint_lo_end": splint_min,
                    "splint_ramp_steps": int(os.getenv("GO1_CURRICULUM_SPLINT_RAMP_STEPS", "8000")),
                },
            )

        # =================================================================
        # 5. Phase 2/3 부상 시나리오 — Emergent Limping
        # =================================================================
        #
        # [연구 논지]
        #   실제 동물이 다리를 다쳤을 때 '절뚝이는' 보행은 누군가가 가르쳐서
        #   나오는 것이 아니라, `통증(pain)` 과 `에너지 소비(energy)` 라는 두 가지
        #   생리학적 비용을 최소화하려는 과정에서 자연선택이 찾은 최적해입니다.
        #
        #   이 프로젝트는 강화학습 에이전트에게 그 두 가지 비용 신호만 주고
        #   (보행 방식을 '처방' 하지 않고) 절뚝임이 emergent behavior 로
        #   나타남을 보이는 것이 목표입니다.
        #
        # [설계 원칙]
        #   정상 env 와 부상 env 모두 동일한 Phase 1 baseline 보상을 공유합니다
        #   (track_lin_vel, track_ang_vel, feet_air_time, dof_torques_l2,
        #   dof_acc_l2, action_rate_l2 등). Phase 2 에서는 부상 env 에
        #   pain/load-duration 비용과 건측 과부하/완전 비사용 방지용 약한
        #   생체역학 regularizer만 추가합니다.
        #
        # [pain 모델 (include_calf=True)]
        #   부상 다리의 발(foot) + 정강이(calf) 접촉력을 합산 → 팔다리 전체가
        #   '아픈 부위'. 무릎 보행 exploit 도 통증으로 자연 차단됩니다.
        #
        #     leg_force = |F_z(foot)| + |F_z(calf)|
        #     pain      = base_cost · 1[leg_force > 1N]                  # 접촉 자체의 통증
        #               + load_cost · 1[leg_force > load_th]             # 누적 하중 통증
        #               + clamp(exp(scale · max(leg_force - th, 0)) - 1) # 과부하 통증
        #
        # [에너지 패널티]
        #   Phase 1 baseline 에서 이미 `dof_torques_l2(-0.0002)` 와
        #   `dof_acc_l2(-2.5e-7)` 이 활성화되어 있어, 모든 관절 사용이
        #   비용으로 반영됩니다. 별도로 부상 다리만 억제하는 항은 두지 않습니다
        #   ('에너지 최소화' 는 전신에 공평하게 적용).
        #
        # =================================================================
        #
        # [v7 핵심 변경점]
        #   v5/v6: threshold=5N, base_cost=0.15 → 너무 공격적 → 넘어짐 40%
        #   v7초안: threshold=20N, base_cost=0.05 → 너무 관대 → 드래깅 위험
        #
        #   최종 균형점: threshold=12N, base_cost=0.1
        #   - 12N 이하 접촉(균형 잡기용 톡톡): base_cost(0.1)만 부과
        #     → 짧게 짚으면 누적 비용 작음, 길게 끌면 누적 비용 큼
        #   - 12N 초과(체중 실기): 지수적 통증 급등 (pain_scale=0.15)
        #     → 세게 밟는 것은 강력히 억제
        #
        #   드래깅 비용 분석:
        #     끌기(연속 접촉) = 0.1 × 0.08 = 매 스텝 -0.008 누적
        #     20스텝 끌기 = -0.16 (상당한 페널티)
        #     vs. 짧게 2~3스텝 짚기 = -0.016~0.024 (감당 가능)
        #   → "짧게 짚어서 균형 잡기 OK, 질질 끌기 NO"
        self.rewards.penalty_pain = RewTerm(
            func=mdp.penalty_pain,
            weight=float(os.getenv("GO1_PAIN_WEIGHT", "-0.08")),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_name": "contact_forces",
                "failure_force_threshold": float(os.getenv("GO1_PAIN_THRESHOLD_N", "12.0")),
                "pain_scale": float(os.getenv("GO1_PAIN_SCALE", "0.15")),
                "overload_tolerance": float(os.getenv("GO1_PAIN_OVERLOAD_TOLERANCE", "0.0")),
                "max_exp_argument": float(os.getenv("GO1_PAIN_MAX_EXP_ARGUMENT", "6.0")),
                "max_penalty": float(os.getenv("GO1_PAIN_MAX_PENALTY", "20.0")),
                "base_contact_cost": float(os.getenv("GO1_PAIN_BASE_CONTACT_COST", "2.0")),
                "contact_detect_threshold": float(os.getenv("GO1_PAIN_CONTACT_THRESHOLD_N", "1.0")),
                "load_contact_cost": float(os.getenv("GO1_PAIN_LOAD_CONTACT_COST", "0.0")),
                "load_contact_threshold": float(os.getenv("GO1_LOAD_CONTACT_THRESHOLD_N", "10.0")),
                "load_contact_cost_severe_multiplier": float(
                    os.getenv("GO1_PAIN_LOAD_COST_SEVERE_MULT", "1.20")
                ),
                "load_contact_cost_mild_multiplier": float(
                    os.getenv("GO1_PAIN_LOAD_COST_MILD_MULT", "0.80")
                ),
                "base_contact_cost_severe_multiplier": float(
                    os.getenv("GO1_PAIN_BASE_CONTACT_SEVERE_MULT", "1.0")
                ),
                "base_contact_cost_mild_multiplier": float(
                    os.getenv("GO1_PAIN_BASE_CONTACT_MILD_MULT", "1.0")
                ),
                "include_calf": os.getenv("GO1_PAIN_INCLUDE_CALF", "1").strip().lower()
                in {"1", "true", "yes", "on"},
                "severity_scaled": os.getenv("GO1_PAIN_SEVERITY_SCALING", "0").strip().lower()
                in {"1", "true", "yes", "on"},
                "severe_splint_length": splint_min,
                "mild_splint_length": splint_max,
                "threshold_severe_multiplier": float(os.getenv("GO1_PAIN_THRESHOLD_SEVERE_MULT", "0.80")),
                "threshold_mild_multiplier": float(os.getenv("GO1_PAIN_THRESHOLD_MILD_MULT", "1.15")),
                "scale_severe_multiplier": float(os.getenv("GO1_PAIN_SCALE_SEVERE_MULT", "1.25")),
                "scale_mild_multiplier": float(os.getenv("GO1_PAIN_SCALE_MILD_MULT", "0.85")),
            },
        )

        self.rewards.intact_limb_overload = RewTerm(
            func=mdp.penalize_intact_limb_overload,
            weight=float(os.getenv("GO1_INTACT_OVERLOAD_WEIGHT", "0.0")),
            params={
                "sensor_name": "contact_forces",
                "overload_threshold": float(os.getenv("GO1_INTACT_OVERLOAD_THRESHOLD_N", "65.0")),
                "overload_scale": float(os.getenv("GO1_INTACT_OVERLOAD_SCALE", "1.0")),
                "max_penalty": float(os.getenv("GO1_INTACT_OVERLOAD_MAX_PENALTY", "120.0")),
                "use_z_only": True,
            },
        )

        self.rewards.injured_limb_force_nonuse = RewTerm(
            func=mdp.penalize_injured_limb_force_nonuse,
            weight=float(os.getenv("GO1_INJURED_FORCE_NONUSE_WEIGHT", "0.0")),
            params={
                "sensor_name": "contact_forces",
                "severe_splint_length": splint_min,
                "mild_splint_length": splint_max,
                "min_force_severe": float(os.getenv("GO1_INJURED_MIN_FORCE_SEVERE_N", "2.0")),
                "min_force_mild": float(os.getenv("GO1_INJURED_MIN_FORCE_MILD_N", "11.0")),
                "front_leg_multiplier": float(os.getenv("GO1_INJURED_MIN_FORCE_FRONT_MULT", "1.15")),
                "rear_leg_multiplier": float(os.getenv("GO1_INJURED_MIN_FORCE_REAR_MULT", "1.0")),
                "ema_alpha": float(os.getenv("GO1_INJURED_NONUSE_EMA_ALPHA", "0.995")),
                "ramp_duration_steps": int(os.getenv("GO1_INJURED_NONUSE_RAMP_STEPS", "8000")),
                "include_calf": os.getenv("GO1_INJURED_NONUSE_INCLUDE_CALF", "1").strip().lower()
                in {"1", "true", "yes", "on"},
            },
        )

        # §4.7 symmetry-encouraging BASELINE penalty: -λs||q - M(q)||^2 (joint
        # mirror). Default 0 (off). Set GO1_SYMMETRY_PENALTY_WEIGHT<0 to train the
        # symmetry-encouraging comparison paradigm (forces L-R symmetric gait even
        # under injury → expected to fail the antalgic biomechanical match).
        self.rewards.joint_mirror_symmetry = RewTerm(
            func=mdp.penalize_joint_mirror_asymmetry,
            weight=float(os.getenv("GO1_SYMMETRY_PENALTY_WEIGHT", "0.0")),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.injured_limb_load_duty_nonuse = RewTerm(
            func=mdp.penalize_injured_limb_load_duty_nonuse,
            weight=float(os.getenv("GO1_INJURED_DUTY_NONUSE_WEIGHT", "0.0")),
            params={
                "sensor_name": "contact_forces",
                "load_contact_threshold": float(os.getenv("GO1_LOAD_CONTACT_THRESHOLD_N", "10.0")),
                "severe_splint_length": splint_min,
                "mild_splint_length": splint_max,
                "min_duty_severe": float(os.getenv("GO1_INJURED_MIN_DUTY_SEVERE", "0.05")),
                "min_duty_mild": float(os.getenv("GO1_INJURED_MIN_DUTY_MILD", "0.28")),
                "front_leg_multiplier": float(os.getenv("GO1_INJURED_MIN_DUTY_FRONT_MULT", "1.10")),
                "rear_leg_multiplier": float(os.getenv("GO1_INJURED_MIN_DUTY_REAR_MULT", "1.0")),
                "ema_alpha": float(os.getenv("GO1_INJURED_NONUSE_EMA_ALPHA", "0.995")),
                "ramp_duration_steps": int(os.getenv("GO1_INJURED_NONUSE_RAMP_STEPS", "8000")),
            },
        )

        self.rewards.injured_limb_load_duty_overuse = RewTerm(
            func=mdp.penalize_injured_limb_load_duty_overuse,
            weight=float(os.getenv("GO1_INJURED_DUTY_OVERUSE_WEIGHT", "0.0")),
            params={
                "sensor_name": "contact_forces",
                "load_contact_threshold": float(os.getenv("GO1_LOAD_CONTACT_THRESHOLD_N", "10.0")),
                "severe_splint_length": splint_min,
                "mild_splint_length": splint_max,
                "max_duty_severe": float(os.getenv("GO1_INJURED_MAX_DUTY_SEVERE", "0.30")),
                "max_duty_mild": float(os.getenv("GO1_INJURED_MAX_DUTY_MILD", "0.50")),
                "front_leg_multiplier": float(os.getenv("GO1_INJURED_MAX_DUTY_FRONT_MULT", "0.95")),
                "rear_leg_multiplier": float(os.getenv("GO1_INJURED_MAX_DUTY_REAR_MULT", "1.0")),
                "ramp_duration_steps": int(os.getenv("GO1_INJURED_OVERUSE_RAMP_STEPS", "8000")),
            },
        )

        self.rewards.injured_limb_light_drag = RewTerm(
            func=mdp.penalize_injured_limb_light_drag,
            weight=float(os.getenv("GO1_INJURED_LIGHT_DRAG_WEIGHT", "0.0")),
            params={
                "sensor_name": "contact_forces",
                "contact_threshold": float(os.getenv("GO1_PAIN_CONTACT_THRESHOLD_N", "1.0")),
                "load_contact_threshold": float(os.getenv("GO1_LOAD_CONTACT_THRESHOLD_N", "10.0")),
                "ramp_duration_steps": int(os.getenv("GO1_INJURED_NONUSE_RAMP_STEPS", "8000")),
            },
        )

        # 생존 보상 — 오래 생존할수록 더 많은 보상 → 넘어짐 회피 강화
        self.rewards.survival_bonus = RewTerm(
            func=mdp_base.is_alive,
            weight=float(os.getenv("GO1_SURVIVAL_BONUS_WEIGHT", "1.5")),
        )

        # =================================================================
        # 7. Phase 2/3 전용: 소폭 보상 조정 (warmstart 호환성 유지)
        # =================================================================
        # 에너지 페널티를 50%만 완화하여 절뚝임을 위한 에너지 예산을 소폭 확보
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = float(os.getenv("GO1_TRACK_LIN_VEL_WEIGHT", "2.5"))
        if hasattr(self.rewards, "dof_torques_l2"):
            _dt_abs = os.getenv("GO1_DOF_TORQUES_WEIGHT")
            if _dt_abs:
                # absolute override (unambiguous; bypasses the *=scale chain)
                self.rewards.dof_torques_l2.weight = float(_dt_abs)
            else:
                self.rewards.dof_torques_l2.weight *= float(os.getenv("GO1_TORQUE_PENALTY_SCALE", "0.5"))
        if hasattr(self.rewards, "dof_acc_l2"):
            self.rewards.dof_acc_l2.weight *= float(os.getenv("GO1_DOF_ACC_PENALTY_SCALE", "0.5"))
        if hasattr(self.rewards, "action_rate_l2"):
            self.rewards.action_rate_l2.weight *= float(os.getenv("GO1_ACTION_RATE_PENALTY_SCALE", "0.5"))
