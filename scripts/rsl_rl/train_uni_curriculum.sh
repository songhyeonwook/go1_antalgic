#!/bin/bash
# Unified teacher WIDE-speed via WARMSTART CURRICULUM.
#
# WHY: from-scratch wide-speed (uni3, vx 0.1-1.0) fixed high-speed falling and let
# Normal+RL run fast, but FROZE FL/FR/RR (they survive but don't move) and broke
# L-R symmetry — because starting from random, "stand still" is the survival-optimal
# solution for the hard injured+fast conditions.
#
# FIX: warmstart from uni2 (the low-speed all-4-leg-walking, L-R-symmetric teacher)
# and RAMP the command speed max gently in stages, each warmstarting the previous.
# Every step is a small perturbation from an all-legs-walking solution, so the legs
# keep walking as the speed rises instead of collapsing to freeze.
#
# uni2 config is reproduced EXACTLY (functional splint + eq.3 + viability floor + PD +
# one-hot + strict + DR); ONLY GO1_CMD_VX_MAX changes across stages.
set -uo pipefail
cd /home/shw/go1_peg/scripts/rsl_rl
source /home/shw/miniconda3/etc/profile.d/conda.sh; conda activate isaac

UNI2="/home/shw/go1_peg/scripts/rsl_rl/logs/rsl_rl/unitree_go1_rough_teacher/2026-07-02_21-50-57_phase2_uni2_s42/model_17999.pt"

latest_ckpt () {  # $1 = run-name glob → prints newest model_N.pt path
  local run
  run=$(ls -dt logs/rsl_rl/unitree_go1_rough_teacher/*"$1" 2>/dev/null | head -1)
  [ -z "$run" ] && return 1
  ls "$run"/model_*.pt 2>/dev/null | awk -F'[_/.]' '{print $(NF-1)"\t"$0}' | sort -n | tail -1 | cut -f2-
}

run_stage () {  # $1=run_name  $2=warmstart_ckpt  $3=cmd_vx_max  $4=max_iter
  echo "===== STAGE $1 : warmstart=$(basename "$2") vx_max=$3 iter=$4 ====="
  PHASE1_CKPT="$2" GO1_NO_WARMSTART=0 \
  GO1_INJURY_ONEHOT=1 \
  GO1_PD_ACTUATOR=1 GO1_PD_KP=20.0 GO1_PD_KD=0.5 \
  GO1_STRICT_TERMINATIONS=1 GO1_BAD_ORIENTATION_LIMIT=0.8 GO1_DOMAIN_RAND=1 \
  GO1_CMD_VX_MIN=0.10 GO1_CMD_VX_MAX="$3" GO1_CMD_VY_ABS=0.0 GO1_CMD_YAW_ABS=0.15 \
  GO1_FLAT_ORIENTATION_WEIGHT=-2.0 GO1_SURVIVAL_BONUS_WEIGHT=1.0 \
  GO1_SPLINT_CALF_ANGLE=-1.5 GO1_PEG_HIP_TORQUE_SCALE=1.0 GO1_SPLINT_CALF_STIFFNESS=12 GO1_SPLINT_CALF_DAMPING=1.0 \
  GO1_INJURED_FORCE_NONUSE_WEIGHT=-0.5 GO1_INJURED_MIN_FORCE_SEVERE_N=4.0 GO1_INJURED_MIN_FORCE_MILD_N=4.0 \
  GO1_INJURED_DUTY_NONUSE_WEIGHT=-1.0 GO1_INJURED_MIN_DUTY_SEVERE=0.30 GO1_INJURED_MIN_DUTY_MILD=0.30 \
  GO1_INJURED_NONUSE_EMA_ALPHA=0.97 GO1_INJURED_NONUSE_RAMP_STEPS=40000 \
  GO1_PAIN_WEIGHT=-0.05 GO1_BASE_HEIGHT_FLOOR_WEIGHT=0.0 GO1_PEG_GRACE_STEPS=30 \
  PHASE2_RUN_NAME="$1" PHASE2_MAX_ITER="$4" NUM_ENVS=2048 SEED=42 \
  bash ./train_phase2_stable.sh
}

# Stage 1: 0.3 → 0.5  (warmstart uni2)
run_stage phase2_uc1_s42 "$UNI2" 0.50 5000
CK1=$(latest_ckpt phase2_uc1_s42) || { echo "STAGE1 ckpt not found"; exit 1; }

# Stage 2: 0.5 → 0.75 (warmstart stage 1)
run_stage phase2_uc2_s42 "$CK1" 0.75 5000
CK2=$(latest_ckpt phase2_uc2_s42) || { echo "STAGE2 ckpt not found"; exit 1; }

# Stage 3: 0.75 → 1.0 (warmstart stage 2)
run_stage phase2_uc3_s42 "$CK2" 1.00 7000
CK3=$(latest_ckpt phase2_uc3_s42) || { echo "STAGE3 ckpt not found"; exit 1; }

echo "===== CURRICULUM DONE ====="
echo "FINAL wide-speed teacher: $CK3"
