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
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from go1_lab.tasks.manager_based.go1_lab.mdp import mirror
from go1_lab.tasks.manager_based.go1_lab.mdp import symmetric_ppo  # noqa: F401  registers SymmetricPPO


# =====================================================================
# Phase 1: Healthy pretrain (정상 보행 선학습)
# =====================================================================

@configclass
class HealthyPPOLstmRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1: 정상 로봇 보행 pretrain (PPO+LSTM).

    ⭐ 설계 원칙:
      1. **Teacher(Phase 2) 와 완전히 동일한 policy 아키텍처 + obs dim** 을 사용하여
         Phase 2 가 Phase 1 체크포인트를 `--resume` 으로 warm-start 할 수 있도록 합니다.
         → `obs_groups` 에 `privileged_obs` 를 포함 (Phase 1 에서는 [0, 0, 1.0] 로 고정).
      2. **Isaac Lab 표준 Go1 rough PPO 하이퍼파라미터** 를 최대한 반영
         (`learning_rate=1e-3`, `num_steps_per_env=24`, entropy/clip/gamma 동일).
      3. 5 frame observation stacking 은 LSTM 과 중복 메모리이므로 비사용
         (env_cfg 에서 history_length 오버라이드 제거함).
    """

    num_steps_per_env = 24
    # LSTM 은 MLP 대비 수렴이 2-3× 느림. 표준 Go1 rough(MLP 1500) 와 "동등 수준"
    # baseline 을 확보하려면 약 5-6k iter 가 필요. 3k 는 trot 기본 구조는 나오지만
    # duty factor 편차/CoM sway 가 완전히 수렴하지 않는 경우가 있어 6000 로 상향.
    max_iterations = 6000
    save_interval = 100
    experiment_name = "unitree_go1_rough_healthy"
    check_for_nan = True

    # Teacher 와 동일한 obs_groups — Phase 2 에서 resume 시 dim mismatch 없음.
    obs_groups = {
        "policy": ["policy", "privileged_obs"],
        "critic": ["policy", "privileged_obs"],
    }

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Isaac Lab 표준 Go1 rough 와 동일 (LSTM 도 adaptive schedule 하에서
        # 1e-3 시작 → 자동으로 줄어들며 안정). 3e-4 는 지나치게 보수적이었음.
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfg(HealthyPPOLstmRunnerCfg):
    """기본 엔트리포인트: Phase 1 정상 보행 PPO+LSTM."""

    pass


@configclass
class HealthyPPOLstmSymmetryRunnerCfg(HealthyPPOLstmRunnerCfg):
    """Phase 1 paper baseline with Go1 left/right symmetry augmentation."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Do not set symmetry_cfg for recurrent policies. In this RSL-RL
        # version, symmetry logging/loss calls act_inference on recurrent
        # mini-batches and sends a 4-D tensor into the LSTM. LSTM symmetry is
        # enforced through environment rewards and the external paper selectors.
    )


@configclass
class HealthyPPOLstmMirrorRunnerCfg(HealthyPPOLstmRunnerCfg):
    """Phase 1 healthy baseline with LSTM-compatible left/right mirror augmentation.

    Identical architecture / obs_groups / hyperparameters to the plain LSTM
    baseline; only the PPO algorithm class is swapped to ``SymmetricPPO``, which
    mirror-doubles the rollout storage before each update (works with recurrent
    policies, unlike RSL-RL's built-in symmetry_cfg). The reward is untouched —
    this enforces the paper's normal-gait mirror-symmetry constraint structurally.
    """

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SymmetricPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HealthyPPOLstmDutyRefineRunnerCfg(HealthyPPOLstmRunnerCfg):
    """Short LSTM refine that preserves force symmetry and fixes duty/trot."""

    max_iterations = 1500
    save_interval = 100
    run_name = "phase1_default_duty_trot_refine"

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.7,
        # The warm-start checkpoint already has good force symmetry. This stage
        # corrects duty factor and trot timing. Do not set symmetry_cfg here:
        # recurrent symmetry logging in this RSL-RL version feeds a 4-D tensor
        # to the LSTM. Symmetry is enforced by rewards and paper selectors.
    )


@configclass
class HealthyPPOLstmTrotBoostRunnerCfg(HealthyPPOLstmRunnerCfg):
    """Phase 1 trot-boost refine: push trot_score from 0.90 to 0.92+
    while holding FL/FR force SI ≤ 5%.

    Root cause of previous instability:
      force_asym weight (-0.0008) was 100x weaker than duty_target (-0.080).
      Duty targets pushed the policy toward patterns that broke force balance.

    Fix:
      • LR 3x lower (3e-5) — prevents large policy jumps that break SI
      • KL limit 2x tighter (0.003) — conservative per-step updates
      • force_asym 15x stronger (via GO1_CONTACT_FORCE_ASYM_WEIGHT env)
      • duty_target 4x weaker — SI stabilisation is the priority
      • trot_sync slightly stronger — nudges trot from 0.90→0.92

    Start from the paper-grade Phase 1 checkpoint (model_6100) which
    already satisfies all gates except trot_score.
    """

    max_iterations = 600
    save_interval = 50
    run_name = "phase1_trot_boost_from_6100"

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.08,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.5,
    )


@configclass
class OfficialGo1SymmetryRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1 symmetry fine-tune compatible with Isaac Lab's official Go1 MLP.

    This runner intentionally keeps the official Go1 policy surface:
    policy-only observations, MLP actor/critic, 24 rollout steps, and the
    standard PPO hyperparameters. That makes Isaac Lab's published
    `Isaac-Velocity-Rough-Unitree-Go1-v0` checkpoint load cleanly, then the
    environment-level symmetry rewards and mirror data augmentation can correct
    left/right force and duty bias for the paper baseline.
    """

    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "unitree_go1_rough_healthy"
    run_name = "phase1_official_symmetric"
    check_for_nan = True

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mirror.compute_symmetric_states,
        ),
    )


@configclass
class OfficialGo1SymmetryRefineRunnerCfg(OfficialGo1SymmetryRunnerCfg):
    """Short low-LR refinement from a near-symmetric official Go1 checkpoint."""

    max_iterations = 1200
    save_interval = 100
    run_name = "phase1_official_symmetric_front_refine"

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.7,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mirror.compute_symmetric_states,
        ),
    )


# =====================================================================
# RMA-canonical feedforward (MLP) teacher pipeline + exact mirror symmetry
# =====================================================================
# The recurrent (LSTM) teacher cannot be made left/right symmetric: the antalgic
# commitment lives in the LSTM hidden state, which has no canonical mirror, so
# data augmentation cannot enforce equivariance. In canonical RMA the teacher is
# FEEDFORWARD (it observes the privileged state directly, so needs no memory);
# recurrence belongs to the student adaptation module. A feedforward teacher
# makes mirror data-augmentation EXACT → reliably mirror-symmetric injury
# responses. The deployed student (Phase 3) stays recurrent (LSTM).

@configclass
class HealthyMlpSymmetryRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1 healthy pretrain — feedforward MLP + exact left/right mirror aug.

    Matches the MLP teacher's architecture / obs_groups (policy + privileged) so
    Phase 2 can warm-start from it. ``privileged_obs`` is constant in Phase 1
    ([0,0,1.0]); the mirror callback maps the (normal) injury index to itself.
    """

    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "unitree_go1_rough_healthy"
    run_name = "phase1_mlp_symmetric"
    check_for_nan = True

    obs_groups = {
        "policy": ["policy", "privileged_obs"],
        "critic": ["policy", "privileged_obs"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Clean left/right mirror data augmentation (mirror minibatch only). On a
        # proprioception-only (R^48) policy + flat symmetric terrain this is
        # enough for a symmetric gait — the confounds that previously broke the
        # closed-loop limit cycle (187-dim height_scan, asymmetric rough terrain)
        # are removed. NOTE: an explicit mirror_loss term destabilised the gait
        # (CoM sway 0.02→0.10) and stacking balance rewards over-constrained it,
        # so neither is used. The reward stays the standard locomotion reward.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mirror.compute_symmetric_states,
        ),
    )


@configclass
class HealthyMlpSymmetryRefineRunnerCfg(HealthyMlpSymmetryRunnerCfg):
    """Phase 1 low-LR symmetry REFINE from a good-trot MLP checkpoint.

    Single-shot from-scratch symmetric-trot training is high variance (a good
    trot emerges but left/right force balance — esp. the rear — is unreliable).
    The proven path (cf. the LSTM model_6100 baseline) is multi-stage: train a
    good trot first, then low-LR refine with the closed-loop balance rewards to
    pin force/duty symmetry WITHOUT disturbing the gait. Warm-start from the
    good-trot checkpoint; keep mirror data augmentation; balance rewards are
    supplied via GO1_PHASE1_BALANCE_REWARDS=1.
    """

    max_iterations = 2500
    save_interval = 100
    run_name = "phase1_mlp_symmetric_refine"

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.7,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=mirror.compute_symmetric_states,
        ),
    )


@configclass
class TeacherMlpSymmetryRunnerCfg(HealthyMlpSymmetryRunnerCfg):
    """Phase 2 antalgic teacher — feedforward MLP + exact left/right mirror aug.

    Identical architecture / obs_groups to the Phase 1 MLP baseline (warm-start
    compatible). The peg-leg environment + privileged injury index are supplied
    via ``GO1_PHASE=teacher``. The antalgic reward (eq.3) is untouched; mirror
    augmentation is exact for this feedforward policy, so left- and right-injury
    cases become mirror images.
    """

    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go1_rough_teacher"
    run_name = "phase2_mlp_teacher_symmetric"


@configclass
class TeacherMlpRunnerCfg(HealthyMlpSymmetryRunnerCfg):
    """Phase 2 antalgic teacher — feedforward MLP, NO mirror augmentation.

    Same architecture / obs_groups as the symmetric MLP teacher (warm-start
    compatible from the MLP Phase-1) but the symmetry data augmentation is
    removed: under active load-bearing, mirror augmentation drives the policy to
    the symmetric non-use optimum instead of the antalgic partial loading. The
    loaded antalgic policy is therefore trained WITHOUT mirror aug; left/right
    consistency for deployment is restored by canonicalization (mirror the
    obs/action for right-side injuries at inference), and for the paper by the
    n-seed aggregate.
    """

    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go1_rough_teacher"
    run_name = "phase2_mlp_teacher_plain"

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =====================================================================
# Phase 2: Teacher (peg-leg 환경 + privileged obs)
# =====================================================================

@configclass
class TeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2: Teacher PPO+LSTM (privileged obs 포함).

    Teacher 는 privileged_obs(부상 다리, 부목 길이, 마찰) 를 직접 관측해 최적 보행을 학습합니다.

    ⭐ **Phase 1 warm-start 구조**:
      - `HealthyPPOLstmRunnerCfg` 와 동일한 아키텍처/obs dim 이므로 Phase 1 체크포인트를
        `--resume --load_run healthy_vX --checkpoint model_N.pt` 로 이어서 학습 가능.
      - Phase 1: 정상 env 만 (peg-leg 없음) → 기본 보행 학습.
      - Phase 2: peg-leg 시나리오 50% 포함 → 기본 보행 유지 + 부상 적응 추가 학습.
    """

    num_steps_per_env = 24
    # Phase 1 에서 이미 보행이 학습되었으므로 Teacher 는 peg-leg 적응에만 집중.
    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go1_rough_teacher"
    check_for_nan = True

    obs_groups = {
        "policy": ["policy", "privileged_obs"],
        "critic": ["policy", "privileged_obs"],
    }

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
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
class TeacherMirrorRunnerCfg(TeacherRunnerCfg):
    """Phase 2 Teacher (peg-leg + privileged obs) with LSTM-compatible left/right
    mirror augmentation.

    Same architecture / obs_groups / hyperparameters as ``TeacherRunnerCfg``; only
    the PPO algorithm class is swapped to ``SymmetricPPO`` (storage-level mirror
    doubling, recurrent-safe). The antalgic reward (eq.3: task − energy − pain) is
    NOT modified — mirror augmentation is a structural constraint that makes
    left-injury and right-injury cases mirror images while the antalgic response
    still emerges from the pain/energy objective.
    """

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SymmetricPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# =====================================================================
# Phase 3: Student Distillation (Teacher latent 모사)
# =====================================================================

@configclass
class DistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Phase 3: Student distillation.

    Teacher(Phase 2)를 동결하고 Student LSTM이
    proprioceptive history만으로 Teacher의 latent z_t를 추정합니다.

    loss: ||z_t - z_hat_t||² (MSE)
    """

    num_steps_per_env = 32
    # Phase 2 teacher 가 15000 iter 로 학습되었을 때, student 가 교사의 latent 를
    # 5가지 시나리오(정상 + FL/FR/RL/RR 부상) 전체에 걸쳐 모사하기 위해 권장 12000 iter.
    # 근거:
    #   - sample-equivalent: Phase 2(15k × 24 × 4096) = 1.47B → Phase 3 에서 동일 경험 확보에
    #     약 11250 iter (32 × 4096) 필요. 12000 은 그에 소폭 여유 추가.
    #   - 다중 시나리오 LSTM distillation 은 60-80% of teacher time 이 경험적 sweet spot.
    #   - Loss plateau 에 도달하면 save_interval 로 저장된 중간 체크포인트에서 조기 중단 가능.
    max_iterations = 12000
    save_interval = 100
    experiment_name = "unitree_go1_rough_student"
    check_for_nan = True

    obs_groups = {
        "student": ["policy"],
        "teacher": ["policy", "privileged_obs"],
    }

    policy = RslRlDistillationStudentTeacherRecurrentCfg(
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
        # The Phase-2 teacher is a FEEDFORWARD MLP (TeacherMlp pivot), so the
        # distillation must load it as non-recurrent (recurrence lives only in the
        # student LSTM). teacher_recurrent=True would expect an RNN the MLP teacher
        # checkpoint does not have.
        teacher_recurrent=False,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        # 1e-3는 LSTM distillation에서 불안정/편향(한쪽 다리만 잘 배우는 현상)을
        # 유발할 수 있음. 5e-4로 낮춰 수렴을 안정화.
        learning_rate=5.0e-4,
        # BPTT 윈도우: 20→32로 늘려 보행 주기(약 20~30 step) + 부상 적응 long-horizon 패턴을
        # Student가 모사할 수 있도록 함. num_steps_per_env=32와 정렬.
        gradient_length=32,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
