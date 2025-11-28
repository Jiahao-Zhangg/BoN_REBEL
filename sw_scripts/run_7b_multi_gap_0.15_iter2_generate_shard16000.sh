#!/bin/bash

# Rerun only the shard starting at index 16000

#SBATCH --job-name=7b_multi_gap_0.15_iter2_generate_16000
#SBATCH --output=logs/7b_multi_gap_0.15_iter2_generate_16000_%j.out
#SBATCH --error=logs/7b_multi_gap_0.15_iter2_generate_16000_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --partition=ml.p4d.24xlarge
#SBATCH --exclusive

set -euo pipefail

mkdir -p logs

echo "Activating conda environment: new_vllm"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"

# Configuration copied from run_7b_multi_gap_0.15_iter2_generate.sh
CHUNK_SIZE="2000"
MODEL="zjhhhh/7b_gap_0.15_multi_step_350_final"
OUTPUT_PREFIX="Qwen7b_multi_gap_0.15_iter1_"
GPU_MEM_UTIL="0.8"
EXTRA_ARGS=""

# Shard we want to rerun
START_IDX=16000
END_IDX=$(( START_IDX + CHUNK_SIZE ))

# Default to all visible GPUs unless GPU_LIST is provided
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GPUS_PER_JOB=1

IFS=',' read -r -a ALL_GPUS <<< "$GPU_LIST"
TOTAL_GPUS="${#ALL_GPUS[@]}"

if (( TOTAL_GPUS < GPUS_PER_JOB )); then
  echo "Error: Need at least $GPUS_PER_JOB GPU(s), but only ${TOTAL_GPUS} provided in GPU_LIST=${GPU_LIST}" >&2
  exit 1
fi

# For a single shard, just use the first GPU in the list
CUDA_VISIBLE_DEVICES="${ALL_GPUS[0]}"

echo "Rerunning shard with start_idx=${START_IDX}, end_idx=${END_IDX}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python ../src/ultrafeedback_multi_preference/generate.py \
  --output_repo "${OUTPUT_PREFIX}${START_IDX}" \
  --model "$MODEL" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --world_size "$GPUS_PER_JOB" \
  --gpu_memory_utilization "$GPU_MEM_UTIL" \
  $EXTRA_ARGS

echo "Shard 16000 rerun completed."

