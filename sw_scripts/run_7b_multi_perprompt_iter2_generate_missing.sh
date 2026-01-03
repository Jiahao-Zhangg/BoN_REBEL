#!/bin/bash

#SBATCH --job-name=7b_multi_perprompt_iter2_generate_missing
#SBATCH --output=logs/7b_multi_perprompt_iter2_generate_missing_%A_%a.out
#SBATCH --error=logs/7b_multi_perprompt_iter2_generate_missing_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --partition=ml.p4d.24xlarge
#SBATCH --exclusive
#SBATCH --array=0-3
# Optional: restrict scheduling to the original four nodes (any of them)
#SBATCH --nodelist=ip-10-1-173-179,ip-10-1-184-205,ip-10-1-196-96,ip-10-1-226-48

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# All shards that should exist for iter1
ALL_STARTS=(0 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000)

# Shards that are already present on the Hub (from your screenshot).
# We will regenerate only the ones NOT in this list.
EXISTING_STARTS=(12000 16000 24000 26000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000)

# Derive the missing shards (bash-only implementation)
is_in_existing() {
  local value="$1"
  for x in "${EXISTING_STARTS[@]}"; do
    if [[ "$x" -eq "$value" ]]; then
      return 0
    fi
  done
  return 1
}

MISSING_STARTS=()
for v in "${ALL_STARTS[@]}"; do
  if ! is_in_existing "$v"; then
    MISSING_STARTS+=("$v")
  fi
done

if [[ "${#MISSING_STARTS[@]}" -eq 0 ]]; then
  echo "No missing shards computed; nothing to do."
  exit 0
fi

echo "All shards:      ${ALL_STARTS[*]}"
echo "Existing shards: ${EXISTING_STARTS[*]}"
echo "Missing shards:  ${MISSING_STARTS[*]}"

# Available nodes for assignment (one task per node)
AVAILABLE_NODES=("ip-10-1-173-179" "ip-10-1-184-205" "ip-10-1-196-96" "ip-10-1-226-48")
NUM_TASKS=${#AVAILABLE_NODES[@]}
NODE_INDEX="$TASK_ID"

if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID; must be 0..$(( NUM_TASKS - 1 ))" >&2
  exit 1
fi

ASSIGNED_NODE="${AVAILABLE_NODES[$NODE_INDEX]}"

# Common configuration (can be tweaked if needed)
TOTAL_GPUS="8"
GPUS_PER_JOB="1"
CHUNK_SIZE="2000"
MODEL="zjhhhh/7b_perprompt_step_332_final"
OUTPUT_PREFIX="zjhhhh/Qwen7b_multi_perprompt_iter1_"
GPU_MEM_UTIL="0.8"
EXTRA_ARGS=""
# Comma-separated logical GPU IDs on the node
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

echo "Activating conda environment: new_vllm"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Array Task ID: $TASK_ID"
echo "Assigned Node: $ASSIGNED_NODE"
echo "Actual Node: ${SLURMD_NODENAME:-unknown}"
if [[ -n "${SLURMD_NODENAME:-}" && "$SLURMD_NODENAME" != "$ASSIGNED_NODE" ]]; then
  echo "WARNING: Running on $SLURMD_NODENAME but expected $ASSIGNED_NODE"
fi

ALL_GPUS=()
if [[ -n "$GPU_LIST" ]]; then
  IFS=',' read -r -a ALL_GPUS <<< "$GPU_LIST"
  TOTAL_GPUS="${#ALL_GPUS[@]}"
  echo "Using GPU list from GPU_LIST: ${ALL_GPUS[*]}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a ALL_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  TOTAL_GPUS="${#ALL_GPUS[@]}"
  echo "Using GPU list from CUDA_VISIBLE_DEVICES: ${ALL_GPUS[*]}"
else
  for ((i=0; i<TOTAL_GPUS; i++)); do ALL_GPUS+=("$i"); done
  echo "Using default GPU indices: ${ALL_GPUS[*]}"
fi

if (( TOTAL_GPUS % GPUS_PER_JOB != 0 )); then
  echo "Error: Number of GPUs (${TOTAL_GPUS}) must be divisible by --gpus-per-job (${GPUS_PER_JOB})" >&2
  exit 1
fi

MAX_CONCURRENT=$(( TOTAL_GPUS / GPUS_PER_JOB ))
echo "GPUs/job=$GPUS_PER_JOB -> up to $MAX_CONCURRENT concurrent jobs per node"

# Build an array of GPU groups from ALL_GPUS, e.g., [0], [1], [2], ...
GPU_GROUPS=()
for ((i=0; i<MAX_CONCURRENT; i++)); do
  start=$(( i * GPUS_PER_JOB ))
  group_elems=()
  for ((k=0; k<GPUS_PER_JOB; k++)); do
    group_elems+=("${ALL_GPUS[$(( start + k ))]}")
  done
  GROUP=$(IFS=,; echo "${group_elems[*]}")
  GPU_GROUPS+=("$GROUP")
done

# Track GPU group occupancy and PID->group mapping
declare -a IN_USE
for ((i=0; i<MAX_CONCURRENT; i++)); do IN_USE[$i]=0; done
declare -A PID_TO_GROUP

find_free_group() {
  for ((i=0; i<MAX_CONCURRENT; i++)); do
    if [[ "${IN_USE[$i]}" -eq 0 ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

# Track background PIDs to limit concurrency
PIDS=()

launch_job() {
  local job_id="$1"
  local group_index="$2"
  local cuda_visible="$3"
  local s_idx="$4"
  local e_idx="$5"

  echo "[task $TASK_ID job $job_id] GPUs={$cuda_visible} range=[$s_idx,$e_idx)" >&2
  CUDA_VISIBLE_DEVICES="$cuda_visible" \
  python ../src/ultrafeedback_multi_preference/generate.py \
    --output_repo "${OUTPUT_PREFIX}${s_idx}" \
    --model "$MODEL" \
    --start_idx "$s_idx" \
    --end_idx "$e_idx" \
    --world_size "$GPUS_PER_JOB" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    $EXTRA_ARGS &

  local pid="$!"
  PIDS+=("$pid")
  IN_USE[$group_index]=1
  PID_TO_GROUP["$pid"]="$group_index"
}

wait_for_slot() {
  while true; do
    # If any group is free, we can launch immediately
    for ((i=0; i<MAX_CONCURRENT; i++)); do
      if [[ "${IN_USE[$i]}" -eq 0 ]]; then
        return 0
      fi
    done
    # Otherwise compact PIDs and free finished groups
    local new_pids=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      else
        local grp="${PID_TO_GROUP[$pid]:-}"
        if [[ -n "$grp" ]]; then
          IN_USE[$grp]=0
          unset 'PID_TO_GROUP[$pid]'
        fi
      fi
    done
    PIDS=("${new_pids[@]}")
    sleep 1
  done
}

# Distribute missing shards across the 4 tasks:
# task 0 gets indices where i % 4 == 0, etc.
SHARDS_FOR_TASK=()
for ((i=0; i<${#MISSING_STARTS[@]}; i++)); do
  if (( i % NUM_TASKS == TASK_ID )); then
    SHARDS_FOR_TASK+=("${MISSING_STARTS[$i]}")
  fi
done

if [[ "${#SHARDS_FOR_TASK[@]}" -eq 0 ]]; then
  echo "No missing shards assigned to array task $TASK_ID; exiting."
  exit 0
fi

echo "Array task $TASK_ID will regenerate shards: ${SHARDS_FOR_TASK[*]}"

job_index=0
for s_idx in "${SHARDS_FOR_TASK[@]}"; do
  e_idx=$(( s_idx + CHUNK_SIZE ))

  wait_for_slot
  group_index="$(find_free_group)"
  cuda_set="${GPU_GROUPS[$group_index]}"

  launch_job "$job_index" "$group_index" "$cuda_set" "$s_idx" "$e_idx"
  job_index=$(( job_index + 1 ))
done

trap 'kill "${PIDS[@]}" 2>/dev/null || true' INT TERM
echo "Waiting for ${#PIDS[@]} running job(s) to finish..."
wait
echo "All jobs for array task $TASK_ID completed."
