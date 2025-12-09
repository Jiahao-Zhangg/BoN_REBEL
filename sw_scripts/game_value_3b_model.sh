#!/bin/bash

#SBATCH --job-name=3b_model_game_value
#SBATCH --output=logs/3b_model_game_value_%j.out
#SBATCH --error=logs/3b_model_game_value_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodelist=ip-10-1-38-11

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
OUTPUT_REPO_PREFIX="zjhhhh/game_3b"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
N_RESPONSE=8
N_JUDGE_SAMPLES=5
SAMPLE_SIZE=500

# Sample checkpoints
CHECKPOINT_REPO_1="zjhhhh/iter2_multi_adversary_step_367_final"
CHECKPOINT_REPO_2="zjhhhh/iter2_ver2_pad_adversary_min_step_367_final"
CHECKPOINT_REPO_3="zjhhhh/iter2_fullcheck_multi_adversary_eta_1e3_step_367_final"
CHECKPOINT_REPO_4="zjhhhh/3b_rerun_rlcf_iter2_step_396_final"

# Group checkpoints per GPU
CHECK_POINT_GROUPS="${CHECKPOINT_REPO_1} ${CHECKPOINT_REPO_2} ${CHECKPOINT_REPO_3} ${CHECKPOINT_REPO_4}"

# Use 1 GPU per judge worker; specify GPU IDs as space-separated groups
GPU_GROUPS="0 1 2 3 4"

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


