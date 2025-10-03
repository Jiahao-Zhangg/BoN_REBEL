#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./scripts/run_multi_generate.sh \
#     --total-gpus 8 \
#     --gpus-per-job 2 \
#     --start-idx 0 \
#     --chunk-size 1000 \
#     --num-jobs 4 \
#     [--model Qwen/Qwen2.5-7B-Instruct] \
#     [--output-prefix Qwen7b_] \
#     [--gpu-list 0,2,4,5]
#
# This launches multiple jobs in parallel, each bound to a disjoint set of GPUs.
# Each job runs:
#   python src/ultrafeedback_multi_preference/generate.py \
#     --output_repo <output-prefix><job_id> \
#     --model <model> \
#     --start_idx <start> --end_idx <end> \
#     --world_size <gpus-per-job>
#
# Ranges are computed as [start, start+chunk), advancing per job.

print_help() {
  cat <<EOF
Run generate.py in parallel across GPUs.

Required flags:
  --total-gpus N         Total available GPUs on the node
  --gpus-per-job N       GPUs allocated per job (tensor parallel world size)
  --start-idx N          Start index for the first job
  --chunk-size N         Number of data items each job processes
  --num-jobs N           Number of jobs to launch (batches of chunk-size)

Optional flags:
  --model STR            Model name (default: Qwen/Qwen2.5-7B-Instruct)
  --output-prefix STR    Output repo prefix (default: Qwen7b_)
  --gpu-mem-util F       vLLM gpu memory utilization (default: 0.8)
  --extra "ARGS"         Extra args to append to generate.py (quoted)
  --gpu-list CSV         Explicit GPU indices (e.g., 0,2,4). If provided,
                         overrides --total-gpus and CUDA_VISIBLE_DEVICES.
                         If omitted, uses CUDA_VISIBLE_DEVICES if set; else
                         assumes indices 0..(--total-gpus-1).

Example:
  ./scripts/run_multi_generate.sh \
    --total-gpus 8 --gpus-per-job 2 --start-idx 0 --chunk-size 1000 --num-jobs 4 \
    --model Qwen/Qwen2.5-7B-Instruct --output-prefix Qwen7b_
EOF
}

TOTAL_GPUS="3"
GPUS_PER_JOB="1"
START_IDX="0"
CHUNK_SIZE="2000"
NUM_JOBS="3"
MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PREFIX="Qwen7b_"
GPU_MEM_UTIL="0.8"
EXTRA_ARGS=""
GPU_LIST="4,5,7"

# Activate conda environment
echo "Activating conda environment" 
source /work2/$USER/Anaconda3/etc/profile.d/conda.sh
conda activate new_vllm

while [[ $# -gt 0 ]]; do
  case "$1" in
    --total-gpus) TOTAL_GPUS="$2"; shift 2;;
    --gpus-per-job) GPUS_PER_JOB="$2"; shift 2;;
    --start-idx) START_IDX="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --num-jobs) NUM_JOBS="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --output-prefix) OUTPUT_PREFIX="$2"; shift 2;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2;;
    --extra) EXTRA_ARGS="$2"; shift 2;;
    --gpu-list) GPU_LIST="$2"; shift 2;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 1;;
  esac
done

if [[ -z "$GPUS_PER_JOB" || -z "$START_IDX" || -z "$CHUNK_SIZE" || -z "$NUM_JOBS" ]]; then
  echo "Missing required flags" >&2
  print_help
  exit 1
fi

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

# Build an array of GPU groups from ALL_GPUS, e.g., [0,1], [2,3], ...
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

job_index=0
next_start=$START_IDX

launch_job() {
  local job_id="$1"
  local cuda_visible="$2"
  local s_idx="$3"
  local e_idx="$4"

  echo "[job $job_id] GPUs={$cuda_visible} range=[$s_idx,$e_idx)" >&2
  CUDA_VISIBLE_DEVICES="$cuda_visible" \
  python src/ultrafeedback_multi_preference/generate.py \
    --output_repo "${OUTPUT_PREFIX}${job_id}" \
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
echo "All jobs completed."


