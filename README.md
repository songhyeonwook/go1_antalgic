#splint-v2 branch

# Antalgic Reinforcement Learning for the Peg-Leg Problem (Unitree Go1)

Code accompanying *"Antalgic reinforcement learning solves the peg-leg problem and
matches injured-animal biomechanics."* A quadruped whose distal leg segment is
immobilised by a functional splint learns a compensatory (antalgic) gait from a
nociceptor-inspired reward, and reproduces injured-animal load-redistribution
biomechanics — using only proprioception at deployment.

Built on **NVIDIA Isaac Lab 5.1** + **RSL-RL**, robot **Unitree Go1**.

## Method overview

- **Antalgic objective** (`mdp/rewards.py`): `r = W_task·r_track − W_energy·‖τ‖² − W_pain·C_pain`,
  with a nociceptor-inspired penalty on the affected-limb normal contact force
  `C_pain(F) = P_base·1_contact + max(0, exp(α(F − F_th)) − 1)` (P_base=0.05, F_th=10 N, α=2.0),
  under a minimal load-bearing (viability) constraint.
- **Injury model — functional splint** (`mdp/events.py`): the affected knee is
  immobilised at a functional, ground-reaching angle by a stiff joint-level spring
  (not a shortened peg) and action-masked; effective length `L_peg` is randomisable.
- **RMA teacher–student** (`agents/rsl_rl_ppo_cfg.py`): a feedforward MLP teacher with
  privileged injury state; an LSTM student infers injury from a proprioceptive history
  and is distilled from the teacher (deploys on proprioception R⁴⁸ only).
- **Control**: joint-position PD (Kp=20, Kd=0.5) at 200 Hz, 50 Hz policy.
- **Domain randomisation** (`mdp/events.py`, `go1_lab_env_cfg.py`): friction, mass,
  motor strength, observation noise, action latency.

## Repository layout

```
source/go1_lab/                          Isaac Lab extension (the environment + algorithm)
  go1_lab/tasks/manager_based/go1_lab/
    go1_lab_env_cfg.py                   env config: obs/reward/termination/DR/curriculum, GO1_* switches
    go1_lab_env.py                       env class (peg-leg action masking, joint locking)
    mdp/rewards.py                       reward terms incl. nociceptor pain (eq.4) + viability floors
    mdp/events.py                        injury injection, functional splint, L_peg / DR randomisation, curriculum
    mdp/observations.py                  proprioception + privileged injury observations
    mdp/mirror.py, mdp/symmetric_ppo.py  left/right mirror augmentation (symmetry)
    agents/rsl_rl_ppo_cfg.py             teacher (MLP) / student (LSTM) / distillation runner configs
  go1_lab/asset/                         Go1 and Go1-pegleg USD assets
scripts/rsl_rl/
  train.py, cli_args.py                  training entry point
  train_phase2_stable.sh (→ _v13 → .sh) Phase-2 antalgic teacher
  train_phase3_student_v11.sh           Phase-3 student distillation
  train_uni_curriculum.sh               wide-speed warmstart curriculum
  analyze_phase2_balanced.sh            balanced evaluation wrapper
  analyze_student.py                     eval + LSTM injury-ID probe + t-SNE
  biomech_analyze.py                     biomechanics (GRF, duty, impulse, CoM, direction-of-change)
  extract_paper_metrics.py              SI (eq.7), impulse/stance %, CoM
  play.py, play_result.py, export_policy_onnx.py   playback / deployment export
train_phase1.sh                          Phase-1 healthy pre-training
```

## Setup

1. Install NVIDIA Isaac Lab 5.1 (see the Isaac Lab documentation).
2. Install this extension:
   ```bash
   python -m pip install -e source/go1_lab
   ```
   The task registers as `Template-Go1-Lab-v0`.

## Reproduce

Behaviour is switched via `GO1_*` environment variables (see `go1_lab_env_cfg.py`).

```bash
# Phase 1 — healthy symmetric pre-training
./train_phase1.sh

# Phase 2 — antalgic teacher (functional splint, eq.4 pain, viability floor, PD, DR)
#   unified across the four leg locations; wide-speed via the warmstart curriculum
cd scripts/rsl_rl && bash ./train_uni_curriculum.sh

# Phase 3 — student distillation (proprioception-only LSTM)
TEACHER_CKPT=<phase2_ckpt> bash ./train_phase3_student_v11.sh

# Evaluation / metrics
GO1_PHASE=student CHECKPOINT=<student_ckpt> AGENT=rsl_rl_distill_cfg_entry_point \
  bash ./analyze_phase2_balanced.sh            # + LSTM injury-ID probe, t-SNE
python3 extract_paper_metrics.py <biomech_dump.npz>   # SI, impulse, CoM, direction-of-change
```

Key switches: `GO1_INJURY_ONEHOT`, `GO1_PD_ACTUATOR`/`GO1_PD_KP`/`GO1_PD_KD`,
`GO1_SPLINT_CALF_ANGLE`, `GO1_SPLINT_LENGTH_MIN/MAX`, `GO1_PAIN_WEIGHT`,
`GO1_DOMAIN_RAND`, `GO1_STRICT_TERMINATIONS`, `GO1_CMD_VX_MIN/MAX`.

## Citation

```
@article{song_antalgic_rl,
  title  = {Antalgic reinforcement learning solves the peg-leg problem and matches injured-animal biomechanics},
  author = {Song, Hyeonwook and others},
  year   = {2026}
}
```
