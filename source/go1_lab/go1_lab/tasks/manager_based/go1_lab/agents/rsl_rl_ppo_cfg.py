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

from go1_lab.tasks.manager_based.go1_lab.mdp import symmetric_ppo  # noqa: F401  registers SymmetricPPO
from go1_lab.tasks.manager_based.go1_lab.mdp import aux_distillation  # noqa: F401  registers StudentTeacherRecurrentAux / DistillationAux
from go1_lab.tasks.manager_based.go1_lab.mdp.rls import RLS_L_PRIOR, RLS_L_SCALE


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

    """
    example..
    policy observation
        - 관절 위치
        - 관절 속도
        - 몸체 각속도
        - 중력 방향
        - 이전 action

    privileged observation
    [
        FL 부상 여부,
        FR 부상 여부,
        RL 부상 여부,
        RR 부상 여부,
        전체 부상 플래그,
        부목 길이,
        발 마찰계수
    ]
    """
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


# =====================================================================
# Phase 3: Student Distillation (Teacher latent 모사)
# =====================================================================

@configclass
class StudentTeacherRecurrentAuxCfg(RslRlDistillationStudentTeacherRecurrentCfg):
    """StudentTeacherRecurrent + latent 보조 예측 헤드 [L̂].

    μ 헤드는 제거됨 — μ 는 antalgic 보행에서 비식별임이 실측됐고(근접 슬립 0%,
    teacher 민감도 ~1%), 연구 주장을 '추정'에서 'μ 강건성'으로 전환
    (test/mu_robustness_report.py: μ∈[0.3, 2.0] 전 구간 성능 평탄).
    ⚠️ 이 변경 전에 학습된 P3 체크포인트(aux 2출력)는 구 설정으로만 로드 가능.
    """

    class_name: str = "StudentTeacherRecurrentAux"
    aux_num_targets: int = 1


@configclass
class DistillationAuxCfg(RslRlDistillationAlgorithmCfg):
    """Distillation + 보조 지도 손실 (부상 env 마스킹).

    aux_targets 의 shift/scale 은 관측 정규화와 동일 규약:
      L: (L − RLS_L_PRIOR) / RLS_L_SCALE  (rls_estimate 채널과 일치)
    """

    class_name: str = "DistillationAux"
    aux_loss_coef: float = 0.5
    # privileged_obs = [FL, FR, RL, RR, injured_flag, L, μ, lin_vel(3)]
    # (μ 채널은 차원 호환을 위해 관측에 남지만 추정 대상이 아님)
    aux_mask: dict = {"group": "privileged_obs", "index": 4}
    aux_targets: list = [
        {"name": "splint_length", "group": "privileged_obs", "index": 5,
         "shift": RLS_L_PRIOR, "scale": RLS_L_SCALE},
    ]


@configclass
class DistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Phase 3: Student distillation.

    Teacher(Phase 2)를 동결하고 Student LSTM이
    proprioceptive history만으로 Teacher의 latent z_t를 추정합니다.

    loss: ||z_t - z_hat_t||² (MSE)
    """

    num_steps_per_env = 32
    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_go1_rough_student"
    check_for_nan = True

    obs_groups = {
        "policy": ["policy"],
        "teacher": ["policy", "privileged_obs"],
    }
    
    """
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["policy", "privileged_obs"],
    }
    
    """

    policy = StudentTeacherRecurrentAuxCfg(
        init_noise_std=0.05,
        noise_std_type="log",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )

    algorithm = DistillationAuxCfg(
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=32,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
