#!/bin/bash
# Phase 3 student distillation for v11 teacher.
#
# Reads PAPER_GRADE_PHASE2_CHECKPOINT.txt automatically.
# After training runs balanced evaluation + select_phase3_student_candidate.py.
#
# Usage:
#   cd ~/go1_peg/scripts/rsl_rl
#   ./train_phase3_student_v11.sh
#
# Optional overrides:
#   TEACHER_CKPT=/path/to/model.pt PHASE3_MAX_ITER=15000 ./train_phase3_student_v11.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="${TASK:-Template-Go1-Lab-v0}"
EXP_NAME="${EXP_NAME:-unitree_go1_rough_student}"
RUN_NAME="${PHASE3_RUN_NAME:-phase3_student_v11}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${PHASE3_MAX_ITER:-12000}"
SEED="${SEED:-42}"
TEACHER_CKPT="${TEACHER_CKPT:-}"

# Auto-read teacher from Phase 2 selector output
if [ -z "$TEACHER_CKPT" ]; then
    P2_FILE="$SCRIPT_DIR/logs/rsl_rl/unitree_go1_rough_teacher/PAPER_GRADE_PHASE2_CHECKPOINT.txt"
    if [ -f "$P2_FILE" ]; then
        CANDIDATE="$(head -n 1 "$P2_FILE" | tr -d '[:space:]')"
        if [ "$CANDIDATE" != "NO_PAPER_GRADE_CANDIDATE" ] && [ -n "$CANDIDATE" ]; then
            TEACHER_CKPT="$CANDIDATE"
        fi
    fi
fi

if [ -z "$TEACHER_CKPT" ] || [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: no paper-grade Phase 2 teacher checkpoint found."
    echo "       Run train_phase2_paper_v11.sh → analyze_phase2_balanced.sh"
    echo "       → select_phase2_paper_candidate.py first."
    echo "       Or set TEACHER_CKPT=/path/to/model_N.pt explicitly."
    exit 1
fi

echo "============================================================"
echo "  Phase 3: Student distillation (v11 teacher)"
echo "  teacher=$TEACHER_CKPT"
echo "  run_name=$RUN_NAME"
echo "  num_envs=$NUM_ENVS  max_iter=$MAX_ITER  seed=$SEED"
echo "============================================================"

# --- distillation training --------------------------------------------------
GO1_PHASE=student \
GO1_EVAL_MODE=random \
python3 train.py \
    --task "$TASK" \
    --agent rsl_rl_distill_cfg_entry_point \
    --num_envs "$NUM_ENVS" \
    --headless \
    --experiment_name "$EXP_NAME" \
    --run_name "$RUN_NAME" \
    --teacher_ckpt_path "$TEACHER_CKPT" \
    --max_iterations "$MAX_ITER" \
    --seed "$SEED"

echo ""
echo "Training complete. Running balanced evaluation on final checkpoint ..."

# --- evaluate best checkpoint -----------------------------------------------
STUDENT_RUN="$(
    find "$SCRIPT_DIR/logs/rsl_rl/$EXP_NAME" -maxdepth 1 -type d \
        -name "*_${RUN_NAME}" | sort | tail -n 1
)"

if [ -z "$STUDENT_RUN" ]; then
    echo "WARNING: could not find student run directory under logs/rsl_rl/$EXP_NAME"
    exit 0
fi

STUDENT_CKPT="$(
    find "$STUDENT_RUN" -maxdepth 1 -type f -name 'model_*.pt' \
        | awk -F'[_/.]' '{ print $(NF-1) "\t" $0 }' \
        | sort -n | tail -n 1 | cut -f2-
)"

if [ -z "$STUDENT_CKPT" ] || [ ! -f "$STUDENT_CKPT" ]; then
    echo "WARNING: no student checkpoint found in $STUDENT_RUN"
    exit 0
fi

echo "Evaluating student: $STUDENT_CKPT"

EXP_NAME=unitree_go1_rough_student \
CHECKPOINT="$STUDENT_CKPT" \
METRICS_JSON="$STUDENT_RUN/student_analysis_metrics.json" \
PHASE2_RUN_NAME="$RUN_NAME" \
GO1_PHASE=student \
GO1_EVAL_MODE=balanced \
./analyze_phase2_balanced.sh

echo ""
echo "Running Phase 3 paper candidate selector ..."
python3 select_phase3_student_candidate.py

P3_FILE="$SCRIPT_DIR/logs/rsl_rl/unitree_go1_rough_student/PAPER_GRADE_PHASE3_CHECKPOINT.txt"
if [ -f "$P3_FILE" ]; then
    CANDIDATE="$(cat "$P3_FILE")"
    if [ "$CANDIDATE" = "NO_PAPER_GRADE_CANDIDATE" ]; then
        echo ""
        echo "No paper-grade Phase 3 candidate yet."
        echo "Try training more iterations or evaluating intermediate checkpoints."
    else
        echo ""
        echo "Paper-grade Phase 3 student checkpoint ready:"
        echo "  $CANDIDATE"
        echo ""
        echo "Export with:"
        echo "  GO1_PHASE=student python3 export_policy_onnx.py \\"
        echo "    --checkpoint '$CANDIDATE' \\"
        echo "    --agent rsl_rl_distill_cfg_entry_point \\"
        echo "    --phase student --headless"
    fi
fi
