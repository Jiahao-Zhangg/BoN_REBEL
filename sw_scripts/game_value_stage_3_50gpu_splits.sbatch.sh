#!/bin/bash

#SBATCH --job-name=stage3_game_value_splits
#SBATCH --output=logs/stage3_splits_%A_%a.out
#SBATCH --error=logs/stage3_splits_%A_%a.err
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus-per-task=1
#SBATCH --partition=ml.p5.48xlarge,ml.p4d.24xlarge

set -euo pipefail

# Ensure logs dir exists for Slurm outputs
mkdir -p logs

# Change to repo root (two levels up from this script)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
echo "Switched to $(pwd)"

# Activate conda environment (try FSX path first, fallback to $HOME)
if [[ -f "/fsx/gstevenw/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /fsx/gstevenw/miniconda3/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate new_vllm
echo "Activated conda env: $(conda info --envs | awk '/\*/ {print $1}')"
echo "Python path: $(which python)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# -----------------------------
# Parameters (mirrors game_value_stage_3_sbatch.sh)
# -----------------------------
DATASET_REPO="zjhhhh/stage3_preprocessed"
JUDGE_MODEL="Qwen/Qwen3-14B"
OUTPUT_REPO_PREFIX="zjhhhh/game_stage3"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
N_RESPONSE=8
N_JUDGE_SAMPLES=5
SAMPLE_SIZE=500

# Checkpoints (same as game_value_stage_3_sbatch.sh)
CHECKPOINT_REPO_1="zjhhhh/qwen2.5_3B_Instruct_min_stage3_seed_555134_eta_1e4_step_1"
CHECKPOINT_REPO_2="zjhhhh/qwen2.5_3B_Instruct_min_stage3_seed_555134_eta_1e4_step_101"
CHECKPOINT_REPO_3="zjhhhh/qwen2.5_3B_Instruct_min_stage3_seed_555134_eta_1e4_step_201"
CHECKPOINT_REPO_4="zjhhhh/qwen2.5_3B_Instruct_min_stage3_seed_555134_eta_1e4_step_301"
CHECKPOINT_REPO_5="zjhhhh/qwen2.5_3B_Instruct_min_stage3_seed_555134_eta_1e4_step_382_final"
CHECKPOINT_REPO_6="zjhhhh/qwen2.5_3B_Instruct_multi_stage3_seed_555134_eta_1e4_step_1"
CHECKPOINT_REPO_7="zjhhhh/qwen2.5_3B_Instruct_multi_stage3_seed_555134_eta_1e4_step_101"
CHECKPOINT_REPO_8="zjhhhh/qwen2.5_3B_Instruct_multi_stage3_seed_555134_eta_1e4_step_201"
CHECKPOINT_REPO_9="zjhhhh/qwen2.5_3B_Instruct_multi_stage3_seed_555134_eta_1e4_step_301"
CHECKPOINT_REPO_10="zjhhhh/qwen2.5_3B_Instruct_multi_stage3_seed_555134_eta_1e4_step_382_final"

# Use 5 splits of 100 prompts each => 10 checkpoints * 5 splits = 50 tasks
SPLIT_SIZE=100
NUM_SPLITS=$(( SAMPLE_SIZE / SPLIT_SIZE ))

# Derive a base repo for stage 1 (same convention)
BASE_OUTPUT_REPO="${OUTPUT_REPO_PREFIX}_base"

# Assemble checkpoints array
CHECKPOINTS=(
  "$CHECKPOINT_REPO_1"
  "$CHECKPOINT_REPO_2"
  "$CHECKPOINT_REPO_3"
  "$CHECKPOINT_REPO_4"
  "$CHECKPOINT_REPO_5"
  "$CHECKPOINT_REPO_6"
  "$CHECKPOINT_REPO_7"
  "$CHECKPOINT_REPO_8"
  "$CHECKPOINT_REPO_9"
  "$CHECKPOINT_REPO_10"
)

# -----------------------------
# Map array task -> (checkpoint, split)
# -----------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CP_IDX=$(( TASK_ID / NUM_SPLITS ))
SPLIT_IDX=$(( TASK_ID % NUM_SPLITS ))

if (( CP_IDX < 0 || CP_IDX >= ${#CHECKPOINTS[@]} )); then
  echo "Invalid checkpoint index $CP_IDX for TASK_ID=$TASK_ID" >&2
  exit 2
fi

CHECKPOINT_ID="${CHECKPOINTS[$CP_IDX]}"
BASE_NAME="${CHECKPOINT_ID%%:*}"
SANITIZED_CP="${BASE_NAME//\//__}"

START_IDX=$(( SPLIT_IDX * SPLIT_SIZE ))
END_IDX=$(( START_IDX + SPLIT_SIZE ))

echo "=== Stage 2 split job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $TASK_ID"
echo "Checkpoint Index: $CP_IDX -> $CHECKPOINT_ID"
echo "Split Index: $SPLIT_IDX (rows $START_IDX..$((END_IDX-1)))"
echo "Node: ${SLURMD_NODENAME:-unknown}"

# Unique summary path per task
SUMMARY_JSON="logs/${SANITIZED_CP}_split${SPLIT_IDX}_scores_${SLURM_JOB_ID}_${TASK_ID}.json"

echo "Launching eval_and_judge for ${CHECKPOINT_ID} [split ${SPLIT_IDX}]"

python src/ultrafeedback_judge/game_value_eval_and_judge.py \
  --base_repo "$BASE_OUTPUT_REPO" \
  --output_repo_prefix "${OUTPUT_REPO_PREFIX}_split${SPLIT_IDX}" \
  --check_points "$CHECKPOINT_ID" \
  --judge_model "$JUDGE_MODEL" \
  --world_size 1 \
  --n_response "$N_RESPONSE" \
  --n_judge_samples "$N_JUDGE_SAMPLES" \
  --model_temperature 0.1 \
  --switch_position \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --score_json_path "$SUMMARY_JSON"

echo "Completed ${CHECKPOINT_ID} [split ${SPLIT_IDX}]"


