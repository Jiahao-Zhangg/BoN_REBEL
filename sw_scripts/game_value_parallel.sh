#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   game_value_three_stage.sh \
#     --dataset_repo <hf/dataset> \
#     --base_output_repo <org/base_repo> \
#     --base_model <model> \
#     --check_points "m1 m2 m3" \
#     --output_repo_prefix <org/prefix> \
#     --judge_model <judge> \
#     [--gpu_groups "0 1 2,3"] \
#     [--extra_args_base "--n_response 2 --world_size 4 --base_gpus 0,1"] \
#     [--extra_args_eval "--n_response 2 --top_p 0.9"] \
#     [--extra_args_score "--beta 1.0"]

# Parse args
DATASET_REPO=""
BASE_OUTPUT_REPO=""
BASE_MODEL=""
CHECK_POINTS=""
OUTPUT_REPO_PREFIX=""
JUDGE_MODEL=""
EXTRA_ARGS_BASE=""
EXTRA_ARGS_EVAL=""
EXTRA_ARGS_SCORE=""
GPU_GROUPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_repo) DATASET_REPO="$2"; shift 2;;
    --base_output_repo) BASE_OUTPUT_REPO="$2"; shift 2;;
    --base_model) BASE_MODEL="$2"; shift 2;;
    --check_points) CHECK_POINTS="$2"; shift 2;;
    --output_repo_prefix) OUTPUT_REPO_PREFIX="$2"; shift 2;;
    --judge_model) JUDGE_MODEL="$2"; shift 2;;
    --gpu_groups) GPU_GROUPS="$2"; shift 2;;
    --extra_args_base) EXTRA_ARGS_BASE="$2"; shift 2;;
    --extra_args_eval) EXTRA_ARGS_EVAL="$2"; shift 2;;
    --extra_args_score) EXTRA_ARGS_SCORE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$DATASET_REPO" || -z "$BASE_OUTPUT_REPO" || -z "$BASE_MODEL" || -z "$CHECK_POINTS" || -z "$OUTPUT_REPO_PREFIX" || -z "$JUDGE_MODEL" ]]; then
  echo "Missing required arguments" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PYTHON:-python}

# Stage 1: generate base responses and push
$PY "$ROOT_DIR/src/ultrafeedback_judge/game_value_generate_base.py" \
  --dataset_repo "$DATASET_REPO" \
  --base_output_repo "$BASE_OUTPUT_REPO" \
  --base_model "$BASE_MODEL" \
  ${EXTRA_ARGS_BASE}

# Stage 2: evaluate checkpoints using the base repo and push per-checkpoint merged datasets
if [[ -z "$GPU_GROUPS" ]]; then
  # Single-process fallback
  $PY "$ROOT_DIR/src/ultrafeedback_judge/game_value_eval_and_judge.py" \
    --base_repo "$BASE_OUTPUT_REPO" \
    --output_repo_prefix "$OUTPUT_REPO_PREFIX" \
    --check_points ${CHECK_POINTS} \
    --judge_model "$JUDGE_MODEL" \
    ${EXTRA_ARGS_EVAL}
else
  # Parallel per-checkpoint with explicit GPU assignment
  read -r -a GROUPS_ARR <<< "$GPU_GROUPS"
  pids=()
  idx=0
  for cp in ${CHECK_POINTS}; do
    group_idx=$(( idx % ${#GROUPS_ARR[@]} ))
    gpu_group="${GROUPS_ARR[$group_idx]}"
    idx=$(( idx + 1 ))

    # Sanitize model name for a temporary summary path
    base_name="${cp%%:*}"
    sanitized="${base_name//\//__}"
    sanitized="${sanitized// /_}"
    summary_tmp="$ROOT_DIR/${sanitized}_eval_tmp.json"

    echo "[Stage2] Launching $cp on GPUs [$gpu_group]"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_group" TOKENIZERS_PARALLELISM=false \
    $PY "$ROOT_DIR/src/ultrafeedback_judge/game_value_eval_and_judge.py" \
      --base_repo "$BASE_OUTPUT_REPO" \
      --output_repo_prefix "$OUTPUT_REPO_PREFIX" \
      --check_points ${cp//,/ } \
      --judge_model "$JUDGE_MODEL" \
      --score_json_path "$summary_tmp" \
      ${EXTRA_ARGS_EVAL} &
    pids+=("$!")
  done

  # Wait for all to complete
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "One or more Stage 2 jobs failed" >&2
    exit 1
  fi
fi

# Stage 3: compute scores locally from the per-checkpoint repos
$PY "$ROOT_DIR/src/ultrafeedback_judge/game_value_compute_scores.py" \
  --output_repo_prefix "$OUTPUT_REPO_PREFIX" \
  --check_points ${CHECK_POINTS//,/ } \
  ${EXTRA_ARGS_SCORE}
