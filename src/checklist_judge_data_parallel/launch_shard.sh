#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for single-node (one shard) and multi-node (many shards) runs.
# - Single-node: provide --idx N, or rely on SLURM_ARRAY_TASK_ID if present.
# - Multi-node fan-out within a single Slurm job: provide --start-idx S and --num-shards K.
#   The script will srun one task per node (round-robin) with world_size=8 per task.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${DIR}/run_inference_on_shard.py"

# Defaults (override via flags below)
IDX=""
USE_ARRAY=0
START_IDX=""
NUM_SHARDS=""

# Default directories inside repo folder
SHARD_DIR="${DIR}/local_shards"
OUT_DIR="${DIR}/outputs"
JUDGE_MODEL="Qwen/Qwen2.5-72B-Instruct"
JUDGE_TYPE="preference_5score"
WORLD_SIZE=8
SELECTION_PAIRS=3
BASE_PAIRS=2
CURRENT_PAIRS=2
N_SAMPLES=5
MAX_TOKENS=256
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
SWITCH_POSITION=0

usage() {
  cat >&2 <<EOF
Usage:
  Single shard (local or within Slurm array):
    $0 --idx N [common options]
    # or rely on SLURM_ARRAY_TASK_ID without --idx

  Multi-node fan-out (within a Slurm allocation):
    $0 --start-idx S --num-shards K [common options]

  Auto-assign per-node (within a Slurm allocation, one task per node):
    # No --idx needed; uses SLURM_NODEID as shard offset
    $0 [--start-idx S] [common options]

Common options:
  --shard-dir PATH         Default: ${SHARD_DIR}
  --out-dir PATH           Default: ${OUT_DIR}
  --model MODEL            Default: ${JUDGE_MODEL}
  --type TYPE              Default: ${JUDGE_TYPE}
  --world-size N           Default: ${WORLD_SIZE} (tensor_parallel_size)
  --selection-pairs N      Default: ${SELECTION_PAIRS}
  --base-pairs N           Default: ${BASE_PAIRS}
  --current-pairs N        Default: ${CURRENT_PAIRS}
  --n-samples N            Default: ${N_SAMPLES}
  --max-tokens N           Default: ${MAX_TOKENS}
  --temperature F          Default: ${TEMPERATURE}
  --top-p F                Default: ${TOP_P}
  --top-k N                Default: ${TOP_K}
  --switch-position        Enable bidirectional judging
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --idx) IDX="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --shard-dir) SHARD_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --model) JUDGE_MODEL="$2"; shift 2 ;;
    --type) JUDGE_TYPE="$2"; shift 2 ;;
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    --selection-pairs) SELECTION_PAIRS="$2"; shift 2 ;;
    --base-pairs) BASE_PAIRS="$2"; shift 2 ;;
    --current-pairs) CURRENT_PAIRS="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p) TOP_P="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --switch-position) SWITCH_POSITION=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}"

run_single() {
  local idx="$1"
  local cmd=(
    python "${PY}"
    --idx "${idx}"
    --shard_dir "${SHARD_DIR}"
    --output_dir "${OUT_DIR}"
    --judge_model "${JUDGE_MODEL}"
    --judge_type "${JUDGE_TYPE}"
    --selection_pairs "${SELECTION_PAIRS}"
    --base_pairs "${BASE_PAIRS}"
    --current_pairs "${CURRENT_PAIRS}"
    --world_size "${WORLD_SIZE}"
    --n_samples "${N_SAMPLES}"
    --max_tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --top_k "${TOP_K}"
  )
  if [[ "${SWITCH_POSITION}" -eq 1 ]]; then
    cmd+=( --switch_position )
  fi
  echo "Running single shard idx=${idx}: ${cmd[*]}"
  exec "${cmd[@]}"
}

count_shards() {
  # Count directories named shard_*
  if [[ -d "${SHARD_DIR}" ]]; then
    find "${SHARD_DIR}" -maxdepth 1 -type d -name 'shard_*' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

node_rank() {
  # Prefer SLURM_NODEID, else compute from nodelist + hostname
  if [[ -n "${SLURM_NODEID:-}" ]]; then
    echo "${SLURM_NODEID}"
    return 0
  fi
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    local host
    host=$(hostname)
    mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    local i=0
    for n in "${nodes[@]}"; do
      if [[ "$n" == "$host" ]]; then
        echo "$i"; return 0
      fi
      i=$(( i + 1 ))
    done
  fi
  echo 0
}

fan_out_multinode() {
  if ! command -v scontrol >/dev/null 2>&1 || ! command -v srun >/dev/null 2>&1; then
    echo "Error: fan-out mode requires Slurm (scontrol and srun)" >&2
    exit 2
  fi
  if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Error: fan-out mode requires a Slurm allocation (no SLURM_JOB_NODELIST)" >&2
    exit 2
  fi
  if [[ -z "${START_IDX}" || -z "${NUM_SHARDS}" ]]; then
    echo "Error: --start-idx and --num-shards are required for fan-out mode" >&2
    usage
    exit 2
  fi

  mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  local num_nodes=${#nodes[@]}
  if (( num_nodes == 0 )); then
    echo "Error: no nodes in SLURM_JOB_NODELIST" >&2
    exit 2
  fi

  echo "Fan-out across ${num_nodes} nodes: start_idx=${START_IDX}, num_shards=${NUM_SHARDS}"
  local i=0
  while (( i < NUM_SHARDS )); do
    local idx=$(( START_IDX + i ))
    local node=${nodes[$(( i % num_nodes ))]}
    echo "Launching shard idx=${idx} on node=${node}"
    srun --nodes=1 --ntasks=1 -w "${node}" --gres=gpu:${WORLD_SIZE} --exclusive \
      bash -lc "python '${PY}' \
        --idx '${idx}' \
        --shard_dir '${SHARD_DIR}' \
        --output_dir '${OUT_DIR}' \
        --judge_model '${JUDGE_MODEL}' \
        --judge_type '${JUDGE_TYPE}' \
        --selection_pairs '${SELECTION_PAIRS}' \
        --base_pairs '${BASE_PAIRS}' \
        --current_pairs '${CURRENT_PAIRS}' \
        --world_size '${WORLD_SIZE}' \
        --n_samples '${N_SAMPLES}' \
        --max_tokens '${MAX_TOKENS}' \
        --temperature '${TEMPERATURE}' \
        --top_p '${TOP_P}' \
        --top_k '${TOP_K}' \
        $([[ "${SWITCH_POSITION}" -eq 1 ]] && echo "--switch_position")" &
    i=$(( i + 1 ))
  done
  wait
}

# Decide mode
if [[ -n "${START_IDX}" || -n "${NUM_SHARDS}" ]]; then
  fan_out_multinode
elif [[ -n "${IDX}" ]]; then
  run_single "${IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_single "${SLURM_ARRAY_TASK_ID}"
elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  # Auto-assign one shard per node using node rank
  rank=$(node_rank)
  base=${START_IDX:-0}
  idx=$(( base + rank ))
  total=$(count_shards)
  if (( total > 0 )) && (( idx >= total )); then
    echo "Node rank ${rank} maps to shard idx=${idx}, but only ${total} shards present under ${SHARD_DIR}. Exiting." >&2
    exit 0
  fi
  run_single "${idx}"
else
  echo "Error: provide --idx, or set SLURM_ARRAY_TASK_ID, or use --start-idx/--num-shards for fan-out mode" >&2
  usage
  exit 1
fi
