# Game value evaluation for local checkpoints using vLLM judge

set -euo pipefail

LOG_OUT="/work2/$USER/BoN_REBEL/log.out"
LOG_ERR="/work2/$USER/BoN_REBEL/log.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

# Change to project directory
cd /work2/$USER/BoN_REBEL

echo "Switched to $(pwd)"

# Activate conda environment
source /work2/$USER/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

echo "Activated conda env: $(conda info --envs | awk '/\*/ {print $1}')"

CHECKPOINT_REPO_1=""zjhhhh/qwen2.5_3B_Instruct_multi_stage1_seed_555134_eta_1e4_step_1""
CHECKPOINT_REPO_2=""zjhhhh/qwen2.5_3B_Instruct_multi_stage1_seed_555134_eta_1e4_step_101""
CHECKPOINT_REPO_3=""zjhhhh/qwen2.5_3B_Instruct_multi_stage1_seed_555134_eta_1e4_step_201""
CHECKPOINT_REPO_4=""zjhhhh/qwen2.5_3B_Instruct_multi_stage1_seed_555134_eta_1e4_step_301""
CHECKPOINT_REPO_5=""zjhhhh/qwen2.5_3B_Instruct_multi_stage1_seed_555134_eta_1e4_step_382_final""

SCORE_BASE_PATH="game_values"
mkdir -p "$SCORE_BASE_PATH"
SCORE_JSON_PATH="$SCORE_BASE_PATH/scores_stage1_multi_largesample.json"


DATASET_REPO=""zjhhhh/stage1_preprocessed""
JUDGE_MODEL="Qwen/Qwen3-14B"
WORLD_SIZE=4
N_JUDGE_SAMPLES=5
N_RESPONSE=4
SAMPLE_SIZE=500
OUTPUT_REPO_PREFIX="zjhhhh/game_stage1_largesample"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE-1)))
fi
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python src/ultrafeedback_judge/game_value_evaluate_local_v2.py \
    --dataset_repo "$DATASET_REPO" \
    --check_points "$CHECKPOINT_REPO_1" "$CHECKPOINT_REPO_2" "$CHECKPOINT_REPO_3" "$CHECKPOINT_REPO_4" "$CHECKPOINT_REPO_5" \
    --n_response "$N_RESPONSE" \
    --base_temperature 0.8 \
    --model_temperature 0.1 \
    --judge_model "$JUDGE_MODEL" \
    --n_judge_samples "$N_JUDGE_SAMPLES" \
    --switch_position \
    --output_repo_prefix "$OUTPUT_REPO_PREFIX" \
    --world_size "$WORLD_SIZE" \
    --score_json_path "$SCORE_JSON_PATH" \
    --end_idx "$SAMPLE_SIZE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi

echo "Job finished at $(date)"
exit $EXIT_CODE
