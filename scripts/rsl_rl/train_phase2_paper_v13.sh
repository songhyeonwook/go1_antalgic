#!/bin/bash
# Phase 2 teacher v13 — PAPER-FAITHFUL MINIMAL REWARD (paper eq.3 + eq.4 only).
#
# ============================================================================
# WHY v13 EXISTS (root-cause of v10-v12 failure)
# ============================================================================
# The paper's central scientific claim (Section 2.2, contribution iv) is:
#   "none of these quantities was prescribed: the policy was not given a
#    target gait, a target duty factor, or a target asymmetry. The reward
#    specified ONLY velocity tracking, energy cost, and the nociceptor
#    penalty. The biomechanical pattern is therefore an EMERGENT consequence
#    of optimisation under the antalgic objective."
#
# Paper reward, eq.(3):
#   r_t = W_task r_task(v,v*) - W_energy ||tau||^2 - W_pain C_pain(Fz)
# Paper pain, eq.(4) with FIXED parameters:
#   C_pain(Fz) = Pbase 1[contact] + max(0, exp(alpha (Fz - Fth)) - 1)
#   Pbase = 0.05,  Fth = 10.0 N,  alpha = 2.0
#
# v10-v12 enabled ~15 auxiliary reward terms (contact_force_asymmetry,
# duty_factor_asymmetry, diagonal_load_asymmetry, front_rear_load_dist,
# trot_sync, leg_duty_target, injured_limb_force/duty_nonuse, duty_overuse,
# light_drag, intact_limb_overload, severity-scaled pain). These:
#   (1) DIRECTLY CONTRADICT the paper's "emergent / not prescribed" claim
#       -> a top-tier reviewer rejects the work.
#   (2) FIGHT EACH OTHER (anti-nonuse pushes force up, pain pushes it down,
#       asymmetry terms push toward symmetry while antalgia is asymmetric)
#       -> no clean equilibrium -> mirror gap randomly 6..74 pp per checkpoint.
#
# v13 strips everything back to eq.(3): the SAME standard Phase-1 locomotion
# reward (task + energy + standard regularizers) in healthy AND injured
# episodes, plus the ONE injury-specific term: pain (eq.4). Antalgic gait,
# mirror symmetry, and the ~69% GRF reduction must EMERGE from this alone.
#
# Why ~69% emerges naturally:
#   alpha=2.0, Fth=10 N makes any Fz>~11 N exponentially forbidden, so the
#   policy keeps injured-leg force just under threshold. Healthy per-leg
#   vertical force ~30-40 N -> reducing to <10 N ~= 70% reduction = paper 69%.
#   It does NOT collapse to full non-use because velocity tracking + alive +
#   lower energy on the other legs reward keeping light support contact, and
#   Pbase mildly penalises persistent dragging (-> reduced stance fraction).
#
# Why mirror symmetry is FREE: pain uses identical params for whichever leg
#   is injured; robot + Phase-1 baseline are L/R symmetric -> FL-injury and
#   FR-injury responses are mirror images by construction.
#
# Why normal-gait symmetry is FREE: healthy episodes (prob 1-p) have no
#   affected limb -> C_pain=0 -> pure task+energy -> Phase-1 symmetric trot.
#
# Severity (L_peg) is handled by MORPHOLOGY (eq.1-2 Jacobian), NOT by scaling
#   the pain function. Pain params are FIXED, exactly as in eq.(4).
#
# The ONLY tuning knob is W_pain (GO1_PAIN_WEIGHT). Use sweep_phase2_v13.sh.
#
# include_calf=1 is kept and is MORE correct for the peg condition: the rigid
# stump contacts ground through the calf/peg, so Fz at the affected limb is
# the calf/peg contact force (the removed foot no longer bears load).
#
# Usage (single run):
#   cd ~/go1_peg/scripts/rsl_rl
#   GO1_PAIN_WEIGHT=-0.10 PHASE2_RUN_NAME=phase2_paper_v13_wp010 \
#     ./train_phase2_paper_v13.sh
# Or use the sweep wrapper:
#   ./sweep_phase2_v13.sh
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Resolve Phase 1 paper-grade warmstart checkpoint -----------------------
EXP_HEALTHY="${PHASE1_EXP_NAME:-unitree_go1_rough_healthy}"
P1_FILE="$SCRIPT_DIR/logs/rsl_rl/$EXP_HEALTHY/PAPER_GRADE_PHASE1_CHECKPOINT.txt"
if [ -z "${PHASE1_CKPT:-}" ] && [ -f "$P1_FILE" ]; then
    CAND="$(head -n 1 "$P1_FILE" | tr -d '[:space:]')"
    if [ "$CAND" != "NO_PAPER_GRADE_CANDIDATE" ] && [ -n "$CAND" ] && [ -f "$CAND" ]; then
        export PHASE1_CKPT="$CAND"
    fi
fi

PHASE2_RUN_NAME="${PHASE2_RUN_NAME:-phase2_paper_v13}" \
PHASE2_MAX_ITER="${PHASE2_MAX_ITER:-15000}" \
NUM_ENVS="${NUM_ENVS:-8192}" \
SEED="${SEED:-42}" \
GO1_PROB_PEG_LEG="${GO1_PROB_PEG_LEG:-0.5}" \
GO1_SPLINT_LENGTH_MIN="${GO1_SPLINT_LENGTH_MIN:-0.20}" \
GO1_SPLINT_LENGTH_MAX="${GO1_SPLINT_LENGTH_MAX:-0.30}" \
GO1_FOOT_FRICTION_MIN="${GO1_FOOT_FRICTION_MIN:-0.5}" \
GO1_FOOT_FRICTION_MAX="${GO1_FOOT_FRICTION_MAX:-1.5}" \
GO1_USE_PEG_LEG_CURRICULUM="${GO1_USE_PEG_LEG_CURRICULUM:-1}" \
GO1_CURRICULUM_PROB_START="${GO1_CURRICULUM_PROB_START:-0.1}" \
GO1_CURRICULUM_PROB_RAMP_STEPS="${GO1_CURRICULUM_PROB_RAMP_STEPS:-5000}" \
GO1_CURRICULUM_SPLINT_LO_START="${GO1_CURRICULUM_SPLINT_LO_START:-0.28}" \
GO1_CURRICULUM_SPLINT_HI_START="${GO1_CURRICULUM_SPLINT_HI_START:-0.33}" \
GO1_CURRICULUM_SPLINT_RAMP_STEPS="${GO1_CURRICULUM_SPLINT_RAMP_STEPS:-8000}" \
GO1_PAIN_WEIGHT="${GO1_PAIN_WEIGHT:--0.10}" \
GO1_PAIN_THRESHOLD_N="${GO1_PAIN_THRESHOLD_N:-10.0}" \
GO1_PAIN_SCALE="${GO1_PAIN_SCALE:-2.0}" \
GO1_PAIN_BASE_CONTACT_COST="${GO1_PAIN_BASE_CONTACT_COST:-0.05}" \
GO1_PAIN_OVERLOAD_TOLERANCE="${GO1_PAIN_OVERLOAD_TOLERANCE:-0.0}" \
GO1_PAIN_MAX_EXP_ARGUMENT="${GO1_PAIN_MAX_EXP_ARGUMENT:-8.0}" \
GO1_PAIN_MAX_PENALTY="${GO1_PAIN_MAX_PENALTY:-100.0}" \
GO1_PAIN_LOAD_CONTACT_COST="0.0" \
GO1_PAIN_SEVERITY_SCALING="0" \
GO1_PAIN_INCLUDE_CALF="${GO1_PAIN_INCLUDE_CALF:-1}" \
GO1_PHASE1_BALANCE_REWARDS="0" \
GO1_PHASE2_GAIT_TUNING="0" \
GO1_CONTACT_FORCE_ASYM_WEIGHT="0.0" \
GO1_DUTY_FACTOR_ASYM_WEIGHT="0.0" \
GO1_DIAGONAL_LOAD_ASYM_WEIGHT="0.0" \
GO1_FRONT_REAR_LOAD_DIST_WEIGHT="0.0" \
GO1_TROT_SYNC_WEIGHT="0.0" \
GO1_LEG_DUTY_TARGET_WEIGHT="0.0" \
GO1_DUTY_FACTOR_DEVIATION_WEIGHT="0.0" \
GO1_INTACT_OVERLOAD_WEIGHT="0.0" \
GO1_INJURED_FORCE_NONUSE_WEIGHT="${GO1_INJURED_FORCE_NONUSE_WEIGHT:-0.0}" \
GO1_INJURED_DUTY_NONUSE_WEIGHT="${GO1_INJURED_DUTY_NONUSE_WEIGHT:-0.0}" \
GO1_INJURED_DUTY_OVERUSE_WEIGHT="0.0" \
GO1_INJURED_LIGHT_DRAG_WEIGHT="0.0" \
GO1_TRACK_LIN_VEL_WEIGHT="${GO1_TRACK_LIN_VEL_WEIGHT:-2.5}" \
GO1_SURVIVAL_BONUS_WEIGHT="${GO1_SURVIVAL_BONUS_WEIGHT:-0.5}" \
GO1_BASE_HEIGHT_WEIGHT="${GO1_BASE_HEIGHT_WEIGHT:--0.3}" \
GO1_ROOT_TOO_LOW_MIN_HEIGHT="${GO1_ROOT_TOO_LOW_MIN_HEIGHT:-0.15}" \
exec "$SCRIPT_DIR/train_phase2.sh" "$@"
