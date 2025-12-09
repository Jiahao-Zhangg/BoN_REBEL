#!/bin/bash

#SBATCH --job-name=7b_model_game_value
#SBATCH --output=logs/7b_model_game_value_%j.out
#SBATCH --error=logs/7b_model_game_value_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodelist=ip-10-1-81-8

set -euo pipefail

# Ensure logs dir exists for Slurm outputs
mkdir -p logs

# Verify environment is activated
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
# Change to BoN_REBEL directory
cd /fsx/gstevenw/testing_alignment_algos/BoN_REBEL
echo "Switched to $(pwd)"

# Activate conda environment
if [[ -f "/fsx/gstevenw/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /fsx/gstevenw/miniconda3/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate new_vllm
echo "Activated conda env: $(conda info --envs | awk '/\*/ {print $1}')"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# -----------------------------
# Parameters
# -----------------------------
DATASET_REPO="zjhhhh/stage2_preprocessed"
JUDGE_MODEL="Qwen/Qwen3-14B"
OUTPUT_REPO_PREFIX="zjhhhh/game_7b"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
N_RESPONSE=8
N_JUDGE_SAMPLES=5
SAMPLE_SIZE=500

# Sample checkpoints
CHECKPOINT_REPO_1="zjhhhh/7b_iter2_multi_0.17_eta_1e4_step_322_final"
CHECKPOINT_REPO_2="zjhhhh/7b_fullcheck_gap_0.17_iter2_eta_1e2_step_322_final"
CHECKPOINT_REPO_3="zjhhhh/7b_rlcf_gap_0.17_iter2_step_335_final"
CHECKPOINT_REPO_4="zjhhhh/7b_iter2_minmin_step_301"
CHECKPOINT_REPO_5="viswavi/qwen2.5_rlcf"

# Group checkpoints per GPU
CHECK_POINT_GROUPS="${CHECKPOINT_REPO_1} ${CHECKPOINT_REPO_2} ${CHECKPOINT_REPO_3} ${CHECKPOINT_REPO_4} ${CHECKPOINT_REPO_5}"

# Use 1 GPU per judge worker; specify GPU IDs as space-separated groups
GPU_GROUPS="0 1 2 3 4 5"

# Derive a base repo for stage 2
BASE_OUTPUT_REPO="${OUTPUT_REPO_PREFIX}_base"

# -----------------------------
# Run three-stage pipeline
# -----------------------------
bash sw_scripts/game_value_parallel.sh \
  --dataset_repo "$DATASET_REPO" \
  --base_output_repo "$BASE_OUTPUT_REPO" \
  --base_model "$BASE_MODEL" \
  --check_points "$CHECK_POINT_GROUPS" \
  --output_repo_prefix "$OUTPUT_REPO_PREFIX" \
  --judge_model "$JUDGE_MODEL" \
  --gpu_groups "$GPU_GROUPS" \
  --extra_args_base "--n_response $N_RESPONSE --world_size 2 --base_gpus 0,1 --temperature 0.8 --end_idx $SAMPLE_SIZE" \
  --extra_args_eval "--world_size 1 --n_response $N_RESPONSE --n_judge_samples $N_JUDGE_SAMPLES --model_temperature 0.1 --switch_position" \
  --extra_args_score "--beta 1.0"

echo "Job finished at $(date)"


