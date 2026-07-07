#!/bin/bash
# Balanced Phase 2 teacher validation: Normal/FL/FR/RL/RR = 1:1:1:1:1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXP_NAME="${EXP_NAME:-unitree_go1_rough_teacher}"
RUN_NAME="${PHASE2_RUN_NAME:-phase2_teacher_antalgic_limp_v7}"
CHECKPOINT="${CHECKPOINT:-}"
TASK="${TASK:-Template-Go1-Lab-v0}"
AGENT="${AGENT:-}"
USE_PRETRAINED_CHECKPOINT="${USE_PRETRAINED_CHECKPOINT:-0}"
PRETRAINED_TASK="${PRETRAINED_TASK:-}"
NUM_ENVS="${NUM_ENVS:-1000}"
STEPS="${STEPS:-2000}"
TARGET_VX="${TARGET_VX:-1.0}"
LOAD_CONTACT_THRESHOLD="${LOAD_CONTACT_THRESHOLD:-10.0}"
METRICS_JSON="${METRICS_JSON:-}"

if [ -z "$CHECKPOINT" ] && [ "$USE_PRETRAINED_CHECKPOINT" != "1" ]; then
    LOG_ROOT="$SCRIPT_DIR/logs/rsl_rl/$EXP_NAME"
    RUN_DIR="$(find "$LOG_ROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 || true)"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: run not found under $LOG_ROOT with suffix $RUN_NAME"
        echo "       override with CHECKPOINT=/path/to/model_N.pt"
        exit 1
    fi
    CHECKPOINT="$(
        find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' \
            | awk -F'[_/.]' '{ print $(NF-1) "\t" $0 }' \
            | sort -n \
            | tail -n 1 \
            | cut -f2-
    )"
fi

if [ "$USE_PRETRAINED_CHECKPOINT" != "1" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "-----------------------------------------------"
echo "  Phase 2 balanced validation"
echo "  task=$TASK"
if [ "$USE_PRETRAINED_CHECKPOINT" = "1" ]; then
    echo "  checkpoint=Isaac Lab published pretrained ($PRETRAINED_TASK)"
else
    echo "  checkpoint=$CHECKPOINT"
fi
echo "  num_envs=$NUM_ENVS steps=$STEPS vx=$TARGET_VX"
echo "  load_contact_threshold=$LOAD_CONTACT_THRESHOLD N"
echo "  splint_range=[${GO1_SPLINT_LENGTH_MIN:-0.20}, ${GO1_SPLINT_LENGTH_MAX:-0.30}] m"
echo "-----------------------------------------------"

METRICS_ARG=()
if [ -n "$METRICS_JSON" ]; then
    METRICS_ARG=(--metrics_json "$METRICS_JSON")
fi
AGENT_ARG=()
if [ -n "$AGENT" ]; then
    AGENT_ARG=(--agent "$AGENT")
fi
PRETRAINED_ARG=()
if [ "$USE_PRETRAINED_CHECKPOINT" = "1" ]; then
    PRETRAINED_ARG=(--use_pretrained_checkpoint)
    if [ -n "$PRETRAINED_TASK" ]; then
        PRETRAINED_ARG+=(--pretrained_task "$PRETRAINED_TASK")
    fi
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-go1}" \
GO1_PHASE="${GO1_PHASE:-teacher}" \
GO1_SPLINT_LENGTH_MIN="${GO1_SPLINT_LENGTH_MIN:-0.20}" \
GO1_SPLINT_LENGTH_MAX="${GO1_SPLINT_LENGTH_MAX:-0.30}" \
python3 -u analyze_student.py \
    --checkpoint "$CHECKPOINT" \
    --task "$TASK" \
    "${AGENT_ARG[@]}" \
    "${PRETRAINED_ARG[@]}" \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --flat \
    --contact_use_z_only \
    --load_contact_threshold "$LOAD_CONTACT_THRESHOLD" \
    --target_vx "$TARGET_VX" \
    --headless \
    --seed "${SEED:-42}" \
    --balance \
    "${METRICS_ARG[@]}"
