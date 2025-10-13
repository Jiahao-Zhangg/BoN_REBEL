# Game value evaluation for local checkpoints using vLLM judge

set -euo pipefail

LOG_OUT="/work2/$USER/BoN_REBEL/log.out"
LOG_ERR="/work2/$USER/BoN_REBEL/log.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

# Change to project directory
cd /work2/$USER/BoN_REBEL

echo "Switched to $(pwd)"

# Activate conda environment
source /work2/$USER/Anaconda3/etc/profile.d/conda.sh
conda activate new_vllm

echo "Activated conda env: $(conda info --envs | awk '/\*/ {print $1}')"

CHECKPOINT_REPO_1=""Qwen/Qwen2.5-3B-Instruct""
CHECKPOINT_REPO_2=""zjhhhh/qwen2.5_3B_Instruct_min_gap_seed_555134_eta_1e4_step_101""
CHECKPOINT_REPO_3=""zjhhhh/qwen2.5_3B_Instruct_min_gap_seed_555134_eta_1e4_step_201""
CHECKPOINT_REPO_4=""zjhhhh/qwen2.5_3B_Instruct_min_gap_seed_555134_eta_1e4_step_301""
CHECKPOINT_REPO_5=""zjhhhh/qwen2.5_3B_Instruct_min_gap_seed_555134_eta_1e4_step_382_final""
SCORE_JSON_PATH="/work2/$USER/BoN_REBEL/src/ultrafeedback_judge/scores_min_response_2.json"
DATASET_REPO=""zjhhhh/choose_gap_min_ver2_tokenized""
JUDGE_MODEL="Qwen/Qwen3-14B"
OUTPUT_REPO_PREFIX="MisDrifter/1005"
WORLD_SIZE=4
N_JUDGE_SAMPLES=5

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE-1)))
fi
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python src/ultrafeedback_judge/game_value_evaluate_local_v2.py \
    --dataset_repo "$DATASET_REPO" \
    --check_points "$CHECKPOINT_REPO_1" "$CHECKPOINT_REPO_2" "$CHECKPOINT_REPO_3" "$CHECKPOINT_REPO_4" "$CHECKPOINT_REPO_5" \
    --n_response 2 \
    --base_temperature 0.8 \
    --model_temperature 0.1 \
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
