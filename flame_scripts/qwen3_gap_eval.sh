#!/bin/bash

#SBATCH --job-name=game_value_eval
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=zhiweiw
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:4
#SBATCH --output=game_value_eval_%j.out
#SBATCH --error=game_value_eval_%j.err

# Game value evaluation for local checkpoints using vLLM judge

set -euo pipefail

export HF_HOME="/project/flame/jiahaoz4/.cache/huggingface"
export HF_DATASETS_CACHE="/project/flame/jiahaoz4/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="/project/flame/jiahaoz4/.cache/huggingface/hub"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to project directory
cd /project/flame/$USER/BoN_REBEL/src/ultrafeedback_judge

echo "Switched to $(pwd)"

# Activate conda environment
source /project/flame/$USER/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

echo "Activated conda env: $(conda info --envs | awk '/\*/ {print $1}')"

CHECKPOINT_REPO="zjhhhh/qwen2.5_3B_Instruct_fixed_gap_beta_1_eta_1e4_step_382_final"
JUDGE_MODEL="Qwen/Qwen3-14B"
OUTPUT_REPO_PREFIX="zjhhhh/qwen3_eval"
WORLD_SIZE=4
N_JUDGE_SAMPLES=5
DEFAULT_DATASET_REPO="zjhhhh/choose_gap_beta_1_tokenized_logprob"
DATASET_REPO="${DATASET_REPO:-$DEFAULT_DATASET_REPO}"

python game_value_evaluate_local.py \
    --check_points "$CHECKPOINT_REPO" \
    --judge_model "$JUDGE_MODEL" \
    --n_judge_samples "$N_JUDGE_SAMPLES" \
    --dataset_repo "$DATASET_REPO" \
    --switch_position \
    --output_repo_prefix "$OUTPUT_REPO_PREFIX" \
    --world_size "$WORLD_SIZE" \

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi

echo "Job finished at $(date)"
exit $EXIT_CODE
