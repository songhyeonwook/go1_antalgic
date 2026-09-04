# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

"""
RslRlOnPolicyRunnerCfg              RslRlDistillationRunnerCfg
        │                                      │
   BaseRunnerCfg                        Phase3DistillationRunnerCfg
    ├── Phase1HealthyRunnerCfg              (phase 3)
    └── Phase2InjuryRunnerCfg


"""
@configclass
class BaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    # Phase 1과 Phase 2에서 공통으로 사용하는 Runner 설정
    num_steps_per_env = 24
    save_interval = 50
    check_for_nan = True

    obs_groups = {
        "policy": ["policy", "privileged_obs"],
        "critic": ["policy", "privileged_obs"],
    }

    # Actor 그조
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        #noise_std_type="scalar",
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005, # 0.01 -> 0.05로 수정됨
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Phase1HealthyRunnerCfg(BaseRunnerCfg):
    # Phase 1: 정상 상태에서 healthy locomotion을 학습.
    max_iterations = 6000
    experiment_name = "unitree_go1_phase1"
    run_name = "phase1"
    
@configclass
class Phase2InjuryRunnerCfg(BaseRunnerCfg):
    # phase 2: privileged information을 사용하는 부상 보행 Teacher
    max_iterations = 12000
    experiment_name = "unitree_go1_phase2"
    run_name = "phase2"


"""
agent_cfg.class_name
        ↓
resolve_runner_class(...)
        ↓
어떤 Runner 클래스를 만들지 결정
        ↓
Phase3DistillationRunner 생성
        ↓
Phase3DistillationRunner._construct_algorithm()
        ↓
Phase3StudentTeacher + Phase3Distillation 생성
"""
# 기본 StudentTeacherRecurrent 말고, Phase3StudentTeacher 클래스를 사용
@configclass
class Phase3StudentTeacherCfg(RslRlDistillationStudentTeacherRecurrentCfg):
    class_name: str = "Phase3StudentTeacher"
    # 출력 정규화 (yaml train.mse_norm). train.py 의 update_agent_cfg 가 덮어쓴다.
    # mse_norm_enable=False 면 아래 값은 무시되고 정책 buffer 는 항등 (mean 0, std 1) 이 된다.
    mse_norm_enable: bool = False
    action_mean: float | list[float] = 0.0   # 12 관절 또는 스칼라
    action_pstd: float = 1.0
    splint_mean: float = 0.0
    splint_std: float = 1.0
    vel_mean: float | list[float] = 0.0      # 3 축 또는 스칼라
    vel_pstd: float = 1.0

@configclass
class Phase3DistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "Phase3Distillation"
    splint_loss_coef: float = 0.5
    vel_loss_coef: float = 1.0
    # train.py 가 env yaml 의 peg_leg.splint_length_range 로 덮어쓴다. 여기 값은 기본값일 뿐.
    splint_length_range: tuple[float, float] = (0.33, 0.45)

@configclass
class Phase3DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    # BaseRunnerCfg 를 상속하지 말 것. obs_groups 에 critic 이 섞이고
    # policy 에 privileged_obs 가 들어가 student 가 53차원을 보게 된다.
    class_name = "Phase3DistillationRunner"
    num_steps_per_env = 50
    save_interval = 50
    max_iterations = 4000            # yaml train.max_iterations 가 덮어씀
    experiment_name = "unitree_go1_phase3"
    run_name = "phase3"
    check_for_nan = True

    obs_groups = {
        "policy": ["policy"],                       # student 49
        "teacher": ["policy", "privileged_obs"],    # teacher 53
    }

    policy = Phase3StudentTeacherCfg(
        init_noise_std=0.05,
        noise_std_type="log",        # Isaac Lab 기본은 "scalar". yaml 의 set: log 와 반드시 일치
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],   # 256 -> 512 -> 256 -> 128 -> 12
        teacher_hidden_dims=[512, 256, 128],   # Phase-2 actor 와 동일해야 로드됨
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )

    algorithm = Phase3DistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=50, # 50-step          
        max_grad_norm=1.0,
        loss_type="mse",
    )