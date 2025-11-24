#!/bin/bash

#SBATCH --job-name=7b_multi_generate_iter2_node4
#SBATCH --output=logs/7b_multi_generate_iter2_node4_%j.out
#SBATCH --error=logs/7b_multi_generate_iter2_node4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --partition=ml.p4d.24xlarge
#SBATCH --nodelist=ip-10-1-226-48
#SBATCH --exclusive

# Create logs directory if it doesn't exist
mkdir -p logs

# Node 4: Handle jobs 24-25 (data indices 48000-51999)
# TOTAL_GPUS=8, START_IDX=48000, NUM_JOBS=2, CHUNK_SIZE=2000

set -euo pipefail

# Configuration for Node 4 (only 2 jobs remaining)
TOTAL_GPUS="8"
GPUS_PER_JOB="1"
START_IDX="48000"
CHUNK_SIZE="2000"
NUM_JOBS="2"
MODEL="zjhhhh/7b_sweep_eta_1e4_step_401"
OUTPUT_PREFIX="Qwen7b_iter1_multi_"
GPU_MEM_UTIL="0.8"
EXTRA_ARGS=""
GPU_LIST="0,1,2,3,4,5,6,7"

# Activate conda environment
echo "Activating conda environment: new_vllm"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

# Verify environment is activated
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

ALL_GPUS=()
if [[ -n "$GPU_LIST" ]]; then
  IFS=',' read -r -a ALL_GPUS <<< "$GPU_LIST"
  TOTAL_GPUS="${#ALL_GPUS[@]}"
  echo "Using GPU list from --gpu-list: ${ALL_GPUS[*]}"
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
  IFS=',' read -r -a __dump <<< "${group_elems[*]}" # no-op to satisfy shellcheck-like linters
  GROUP=$(IFS=,; echo "${group_elems[*]}")
  GPU_GROUPS+=("$GROUP")
done

# Track background PIDs to limit concurrency
PIDS=()

job_index=24
next_start=$START_IDX

launch_job() {
  local job_id="$1"
  local cuda_visible="$2"
  local s_idx="$3"
  local e_idx="$4"

  echo "[job $job_id] GPUs={$cuda_visible} range=[$s_idx,$e_idx)" >&2
  CUDA_VISIBLE_DEVICES="$cuda_visible" \
  python ../src/ultrafeedback_multi_preference/generate.py \
    --output_repo "${OUTPUT_PREFIX}${s_idx}" \
    --model "$MODEL" \
    --start_idx "$s_idx" \
    --end_idx "$e_idx" \
    --world_size "$GPUS_PER_JOB" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    $EXTRA_ARGS &

  PIDS+=("$!")
}

wait_for_slot() {
  # If below concurrency limit, return immediately; else wait for any PID
  while (( ${#PIDS[@]} >= MAX_CONCURRENT )); do
    # wait -n is not POSIX; emulate: wait for each and compact
    local new_pids=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      fi
    done
    if (( ${#new_pids[@]} < ${#PIDS[@]} )); then
      PIDS=("${new_pids[@]}")
      break
    fi
    sleep 1
  done
}

for ((j=0; j<NUM_JOBS; j++)); do
  wait_for_slot
  group_index=$(( j % MAX_CONCURRENT ))
  cuda_set="${GPU_GROUPS[$group_index]}"

  s_idx=$next_start
  e_idx=$(( s_idx + CHUNK_SIZE ))
  next_start=$e_idx

  launch_job "$job_index" "$cuda_set" "$s_idx" "$e_idx"
  job_index=$(( job_index + 1 ))
done

echo "Waiting for ${#PIDS[@]} running job(s) to finish..."
wait
echo "All jobs completed on Node 4."
