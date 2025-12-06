#!/bin/bash

#SBATCH --job-name=3b_rlcf_iter2_generate_patch
#SBATCH --output=logs/3b_rlcf_iter2_generate_patch_%A_%a.out
#SBATCH --error=logs/3b_rlcf_iter2_generate_patch_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --partition=ml.p4d.24xlarge
#SBATCH --exclusive
# Only rerun missing shards 8000 and 16000
#SBATCH --array=0-1
# Use two nodes, one per shard
#SBATCH --nodelist=ip-10-1-231-85,ip-10-1-235-38


# Array task mapping (rerun only):
#   Task 0 -> shard start_idx=8000
#   Task 1 -> shard start_idx=16000

set -euo pipefail

cd /fsx/gstevenw/testing_alignment_algos/BoN_REBEL
# Create logs directory if it doesn't exist
mkdir -p logs

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Per-task configuration (rerun only shards 8000 and 16000)
START_OFFSETS=(8000 16000)
NUM_JOBS_LIST=(1 1)
JOB_INDEX_START=(4 8)

# Available nodes for assignment (one task per node)
AVAILABLE_NODES=("ip-10-1-231-85" "ip-10-1-235-38")
NODE_INDEX="$TASK_ID"
ASSIGNED_NODE="${AVAILABLE_NODES[$NODE_INDEX]}"

# Validate array index
if (( TASK_ID < 0 || TASK_ID >= ${#START_OFFSETS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID; must be 0..$(( ${#START_OFFSETS[@]} - 1 ))" >&2
  exit 1
fi

# Common configuration (can be tweaked if needed)
TOTAL_GPUS="8"
# Each shard uses all 8 GPUs on its node
GPUS_PER_JOB="8"
CHUNK_SIZE="2000"
MODEL="zjhhhh/3b_rerun_rlcf_expand_eta_1e4_step_373_final"
OUTPUT_PREFIX="Qwen3b_rlcf_iter1_"
GPU_MEM_UTIL="0.8"
EXTRA_ARGS=""
# Comma-separated logical GPU IDs on the node
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

START_IDX="${START_OFFSETS[$TASK_ID]}"
NUM_JOBS="${NUM_JOBS_LIST[$TASK_ID]}"
job_index="${JOB_INDEX_START[$TASK_ID]}"

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
echo "Task plan -> start_idx=$START_IDX, num_jobs=$NUM_JOBS, job_index_start=$job_index"

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
echo "Launching $NUM_JOBS jobs with up to $MAX_CONCURRENT concurrent (GPUs/job=$GPUS_PER_JOB)"

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
next_start=$START_IDX

launch_job() {
  local job_id="$1"
  local group_index="$2"
  local cuda_visible="$3"
  local s_idx="$4"
  local e_idx="$5"

  echo "[task $TASK_ID job $job_id] GPUs={$cuda_visible} range=[$s_idx,$e_idx)" >&2
  CUDA_VISIBLE_DEVICES="$cuda_visible" \
  python src/ultrafeedback_multi_preference/generate.py \
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

for ((j=0; j<NUM_JOBS; j++)); do
  wait_for_slot
  group_index="$(find_free_group)"
  cuda_set="${GPU_GROUPS[$group_index]}"

  s_idx=$next_start
  e_idx=$(( s_idx + CHUNK_SIZE ))
  next_start=$e_idx

  launch_job "$job_index" "$group_index" "$cuda_set" "$s_idx" "$e_idx"
  job_index=$(( job_index + 1 ))
done

trap 'kill "${PIDS[@]}" 2>/dev/null || true' INT TERM
echo "Waiting for ${#PIDS[@]} running job(s) to finish..."
wait
echo "All jobs for array task $TASK_ID completed."

