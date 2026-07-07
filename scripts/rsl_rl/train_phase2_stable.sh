#!/bin/bash
# Phase 2 STABLE antalgic — fixes the antalgic_v17 TIP-OVER failure.
#
# DIAGNOSIS (2026-06-21, from biomech npz): the v17 "loaded 12.5N" was an
# ARTIFACT. At eval the injured robot ROLLED toward the injured (shorter-peg)
# corner — roll grew 0->~40-50 deg over the 10-step grace window — and
# terminated via bad_orientation in ~11 steps. The "force" was the average of
# fall-spike contacts, not stable loaded locomotion. ROOT CAUSE: the reward had
# NO trunk-tilt-ANGLE penalty (flat_orientation_l2 weight 0), and base_height
# was pinned at 0.30, so the only way to satisfy the use-floor (load the short
# peg) at height 0.30 was to DIP that corner (roll) -> it rolled until it fell.
#
# THREE additions vs v17 (everything else identical):
#   1. flat_orientation_l2 = -1.5   (penalise trunk tilt ANGLE -> stay level)
#   2. base_height target 0.27 / weight -0.15  (relaxed: allow a slight squat so
#      a short peg reaches the ground at a LEVEL posture, not by rolling)
#   3. use-floor moderated to -1.0 / min 6N    (load the peg without over-driving
#      the roll; flat_orientation now has room to keep the body level)
# The L_peg curriculum (0.28->0.20) and DR are inherited from v13.
#
# Usage:
#   cd ~/go1_peg/scripts/rsl_rl
#   PHASE2_RUN_NAME=phase2_stable_s42 ./train_phase2_stable.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PHASE1_CKPT="${PHASE1_CKPT:-$SCRIPT_DIR/logs/rsl_rl/unitree_go1_rough_healthy/2026-06-16_12-51-52_faithful_p1_s1/model_3999.pt}"

# proprio R48 + flat terrain + domain randomisation (paper §4.3/4.8)
export GO1_PROPRIO_ONLY=1
export GO1_FLAT_TERRAIN=1
export GO1_DOMAIN_RAND=1
export GO1_PEG_WEAKEN_JOINTS=hip
export AGENT="${AGENT:-rsl_rl_teacher_mlp_cfg_entry_point}"

# pain eq.4 (fixed) + tight cap
export GO1_PAIN_WEIGHT="${GO1_PAIN_WEIGHT:--0.10}"
export GO1_PAIN_MAX_PENALTY=200
export GO1_PAIN_MAX_EXP_ARGUMENT=10
# energy (dof_torques_l2 = -0.006, matches v17)
export GO1_TORQUE_PENALTY_SCALE="${GO1_TORQUE_PENALTY_SCALE:-30}"

# use-floor (moderated vs v17's -2.0/8N)
export GO1_INJURED_FORCE_NONUSE_WEIGHT="${GO1_INJURED_FORCE_NONUSE_WEIGHT:--1.0}"
export GO1_INJURED_MIN_FORCE_SEVERE_N="${GO1_INJURED_MIN_FORCE_SEVERE_N:-6.0}"
export GO1_INJURED_MIN_FORCE_MILD_N="${GO1_INJURED_MIN_FORCE_MILD_N:-6.0}"
export GO1_INJURED_NONUSE_EMA_ALPHA=0.90
export GO1_INJURED_NONUSE_RAMP_STEPS=2000

# *** THE FIX: balance shaping ***
export GO1_FLAT_ORIENTATION_WEIGHT="${GO1_FLAT_ORIENTATION_WEIGHT:--1.5}"
export GO1_BASE_HEIGHT_TARGET="${GO1_BASE_HEIGHT_TARGET:-0.27}"
export GO1_BASE_HEIGHT_WEIGHT="${GO1_BASE_HEIGHT_WEIGHT:--0.15}"

PHASE2_RUN_NAME="${PHASE2_RUN_NAME:-phase2_stable_antalgic}" \
PHASE2_MAX_ITER="${PHASE2_MAX_ITER:-6000}" \
NUM_ENVS="${NUM_ENVS:-4096}" \
SEED="${SEED:-42}" \
exec "$SCRIPT_DIR/train_phase2_paper_v13.sh" "$@"
