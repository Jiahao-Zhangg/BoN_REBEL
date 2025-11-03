#!/bin/bash
# Rerun specific fullcheck shards on H100 using 4 GPUs per shard.
# This runs up to two shards concurrently on a single node (GPUs 0-3 and 4-7).
# Usage:
#   sbatch rerun_inference_fullcheck_H100.sh            # reruns shards 45 and 50
#   SHARDS="45 50 60" sbatch rerun_inference_fullcheck_H100.sh  # schedules shards in pairs

#SBATCH --job-name=inference_fullcheck_rerun_H100
#SBATCH --output=logs/inference_fullcheck_rerun_H100_%A.out
#SBATCH --error=logs/inference_fullcheck_rerun_H100_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --exclusive
# Optionally constrain nodes (same as main H100 script); scheduler will choose one
#SBATCH --nodelist=ip-10-1-38-11,ip-10-1-81-8

set -euo pipefail

mkdir -p logs

echo "Activating conda environment: new_vllm"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Config paths to match fullcheck runs
SCRIPT_PATH="src/checklist_judge_data_parallel/run_inference_on_shard_robust_fullchecks.py"
SHARD_DIR="./fullcheck_shards"
OUTPUT_DIR="./inference_scores_fullcheck"
JUDGE_MODEL="Qwen/Qwen3-14B"

# Shards to rerun (space-separated). Allow override via SHARDS env var.
read -r -a TARGET_SHARDS <<< "${SHARDS:-45 50}"
if [ ${#TARGET_SHARDS[@]} -eq 0 ]; then
  echo "No shards specified. Set SHARDS env var or edit script."
  exit 1
fi

# Define two 4-GPU groups per node
GPU_GROUPS=("0,1,2,3" "4,5,6,7")
MAX_CONCURRENT=${#GPU_GROUPS[@]}

echo "=== Rerun Info ==="
echo "Job ID: ${SLURM_JOB_ID:-no-slurm}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Target shards: ${TARGET_SHARDS[*]}"
echo "GPU groups: ${GPU_GROUPS[*]}"

run_inference_group() {
  local shard_idx=$1
  local gpu_group=$2
  local group_id=$3

  local shard_path="${SHARD_DIR}/shard_$(printf "%05d" "${shard_idx}")"
  if [ ! -d "${shard_path}" ]; then
    echo "Shard directory not found, skipping: ${shard_path}"
    return 0
  fi

  local shard_idx_padded
  shard_idx_padded=$(printf "%05d" "${shard_idx}")
  echo "Starting inference for shard ${shard_idx} on GPUs ${gpu_group} (group ${group_id})"

  local shard_log_out="logs/shard_${shard_idx_padded}_gpus${group_id}_${SLURM_JOB_ID:-manual}.out"
  local shard_log_err="logs/shard_${shard_idx_padded}_gpus${group_id}_${SLURM_JOB_ID:-manual}.err"

  (
    export CUDA_VISIBLE_DEVICES=${gpu_group}
    python "${SCRIPT_PATH}" \
      --idx "${shard_idx}" \
      --shard_dir "${SHARD_DIR}" \
      --judge_model "${JUDGE_MODEL}" \
      --world_size 4 \
      --judge_type preference_5score \
      --selection_pairs 4 \
      --base_pairs 2 \
      --current_pairs 2 \
      --switch_position \
      --push_to_hub \
      --hf_repo_template zjhhhh/fullcheck_scores_{target}_{shard_idx} \
      --output_dir "${OUTPUT_DIR}"
  ) >"${shard_log_out}" 2>"${shard_log_err}" &

  echo "Launched inference for shard ${shard_idx} on GPUs ${gpu_group} (PID: $!)"
}

# Schedule shards in pairs (2 at a time), each using a 4-GPU group
idx=0
while [ ${idx} -lt ${#TARGET_SHARDS[@]} ]; do
  for g in $(seq 0 $(( MAX_CONCURRENT - 1 ))); do
    s_idx=$(( idx + g ))
    if [ ${s_idx} -ge ${#TARGET_SHARDS[@]} ]; then
      break
    fi
    shard=${TARGET_SHARDS[$s_idx]}
    run_inference_group "${shard}" "${GPU_GROUPS[$g]}" "$g"
  done
  echo "Launched up to ${MAX_CONCURRENT} shard(s); waiting for completion before next batch..."
  wait
  idx=$(( idx + MAX_CONCURRENT ))
done

echo "Waiting for all rerun shard jobs to complete..."
wait
echo "=== Rerun complete for shards: ${TARGET_SHARDS[*]} ==="
