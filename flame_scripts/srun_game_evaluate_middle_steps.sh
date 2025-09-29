#!/bin/bash

#SBATCH --job-name=game_value_eval
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=zhiweiw
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:4
#SBATCH --output=game_value_eval_middle_steps%j.out
#SBATCH --error=game_value_eval_middle_steps%j.err

# Game value evaluation for local checkpoints using vLLM judge

set -euo pipefail

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to project directory
cd /home/$USER/BoN_REBEL

echo "Switched to $(pwd)"

# Activate conda environment
source /project/flame/$USER/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

echo "Activated conda env: $(conda info --envs | awk '/\*/ {print $1}')"

CHECKPOINT_REPO_1=""Qwen/Qwen2.5-3B-Instruct""
CHECKPOINT_REPO_2=""zjhhhh/qwen2.5_3B_Instruct_multi_gap_seed_555134_beta_1_eta_1e4_step_101""
CHECKPOINT_REPO_3=""zjhhhh/qwen2.5_3B_Instruct_multi_gap_seed_555134_beta_1_eta_1e4_step_201""
CHECKPOINT_REPO_4=""zjhhhh/qwen2.5_3B_Instruct_multi_gap_seed_555134_beta_1_eta_1e4_step_301""
CHECKPOINT_REPO_5=""zjhhhh/qwen2.5_3B_Instruct_multi_gap_seed_555134_beta_1_eta_1e4_step_382_final""
SCORE_JSON_PATH="/home/$USER/BoN_REBEL/src/ultrafeedback_judge/scores.json"
DATASET_REPO=""zjhhhh/choose_gap_beta_1_multi_tokenized""
JUDGE_MODEL="Qwen/Qwen3-14B"
OUTPUT_REPO_PREFIX="MisDrifter/928"
WORLD_SIZE=4
N_JUDGE_SAMPLES=5

python src/ultrafeedback_judge/game_value_evaluate_local_v2.py \
    --dataset_repo "$DATASET_REPO" \
    --check_points "$CHECKPOINT_REPO_1" "$CHECKPOINT_REPO_2" "$CHECKPOINT_REPO_3" "$CHECKPOINT_REPO_4" "$CHECKPOINT_REPO_5" \
    --n_response 1 \
    --judge_model "$JUDGE_MODEL" \
    --n_judge_samples "$N_JUDGE_SAMPLES" \
    --switch_position \
    --output_repo_prefix "$OUTPUT_REPO_PREFIX" \
    --world_size "$WORLD_SIZE" \
    --score_json_path "$SCORE_JSON_PATH" \
    --end_idx 500

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi

echo "Job finished at $(date)"
exit $EXIT_CODE
