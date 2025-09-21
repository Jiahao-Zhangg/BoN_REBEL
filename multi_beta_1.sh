#!/bin/bash
#SBATCH --job-name=reward-pipeline
#SBATCH --time=08:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=zhiweiw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -o slurm-%j.out

set -euo pipefail

# Hugging Face caches (match interactive environment)
export HF_HOME="/project/flame/jiahaoz4/.cache/huggingface"
export HF_DATASETS_CACHE="/project/flame/jiahaoz4/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="/project/flame/jiahaoz4/.cache/huggingface/hub"

SECONDS=0

# Change to BoN_REBEL directory as base path
cd /project/flame/$USER/BoN_REBEL

# Activate conda base tooling
source /project/flame/$USER/miniconda3/etc/profile.d/conda.sh

# Configuration variables (modify as needed)
ETA="1e4"                     # REBEL eta parameter
BETA="1"
TOTAL_EPISODES="40000"         # Episodes fed to rebel.py --total_episodes
TEST_MODE="false"             # If true, pass --test to rebel.py for quick dataset sampling
WORLD_SIZE="4"               # Number of GPUs/processes to use
BON="true"                  # Toggle Best-of-N training behaviour for rebel.py
OUTPUT_DIR="/project/flame/$USER/multi_rebel/outputs_beta_${BETA}_eta_${ETA}"
HF_REPO_NAME="zjhhhh/qwen2.5_3B_Instruct_multi_beta_${BETA}_eta_${ETA}"
EVAL_RESULTS_DIR="/project/flame/$USER/qwen2.5_3B_Instruct_multi_evaluation_results"

TRAIN_INPUT_REPO="zjhhhh/whole_sw_maxlen_8192_rescale_mean_maxlenp_1024_beta_1.0_multi_tokenized"

# Evaluation configuration
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
EVAL_DATASET_NAME="zjhhhh/whole_sw_maxlen_8192_rescale_mean_maxlenp_1024_beta_1.0_multi_tokenized"
REWARD_MODEL="RLHFlow/ArmoRM-Llama3-8B-v0.1"
EVAL_MAXLEN=2048
BEST_OF_N=1
EVAL_WORLD_SIZE=4
EVAL_MAX_SAMPLES="1000"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_RESULTS_DIR"


echo "Starting reward training pipeline from $(pwd)"

# Step 1: Training
echo "Step 1: Running training..."
set +u
conda activate rebel
set -u
TRAIN_CMD=(
    python -m accelerate.commands.launch
    --config_file accelerate_cfgs/deepspeed_config_stage_3.yaml
    --main-process-port 29081
    --num_processes "$WORLD_SIZE"
    src/ultrafeedback_judge/rebel_save.py
    --output_dir "$OUTPUT_DIR"
    --rebel.eta "$ETA"
    --total_episodes "$TOTAL_EPISODES"
    --task.input_repo "$TRAIN_INPUT_REPO"
)

case "${BON,,}" in
    true)
        TRAIN_CMD+=(--rebel.bon)
        ;;
    false)
        TRAIN_CMD+=(--rebel.no-bon)
        ;;
    *)
        echo "Error: BON must be 'true' or 'false', got '$BON'"
        exit 1
        ;;
esac

if [[ "$TEST_MODE" == "true" ]]; then
    TRAIN_CMD+=(--test)
fi

"${TRAIN_CMD[@]}"

echo "Training completed successfully!"

# Step 2: Convert all checkpoints to FP32 and upload each to Hugging Face
echo "Step 2: Converting all checkpoints to FP32 and uploading to Hugging Face..."

mapfile -d '' CHECKPOINT_DIRS < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "ultrafeedback_rebel_*" -print0 | sort -z -V)

if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    echo "Error: No checkpoint directories found in $OUTPUT_DIR"
    exit 1
fi

CHECKPOINT_ARGS=()
for CKPT_DIR in "${CHECKPOINT_DIRS[@]}"; do
    echo "Processing checkpoint: $CKPT_DIR"

    if [ ! -f "$CKPT_DIR/zero_to_fp32.py" ]; then
        echo "Warning: zero_to_fp32.py not found in $CKPT_DIR, skipping conversion"
        continue
    fi

    if [ ! -f "$CKPT_DIR/pytorch_model.bin" ]; then
        python "$CKPT_DIR/zero_to_fp32.py" "$CKPT_DIR" "$CKPT_DIR/pytorch_model.bin"
        echo "Completed conversion for $CKPT_DIR"
    else
        echo "pytorch_model.bin already exists for $CKPT_DIR, reusing"
    fi

    if [ ! -f "$CKPT_DIR/pytorch_model.bin" ]; then
        echo "Conversion failed for $CKPT_DIR, skipping upload"
        continue
    fi

    BASENAME=$(basename "$CKPT_DIR")
    STEP_SEGMENT=$(echo "$BASENAME" | sed -E 's/.*(step_[0-9]+(_final)?)$/\1/')
    if [[ -z "$STEP_SEGMENT" || "$STEP_SEGMENT" == "$BASENAME" ]]; then
        STEP_SEGMENT=$(echo "$BASENAME" | tr '/:' '__')
    fi
    HF_STEP_REPO="${HF_REPO_NAME}_${STEP_SEGMENT}"

    python save_model.py \
        --checkpoint_path "$CKPT_DIR" \
        --hf_repo "$HF_STEP_REPO" \
        --model "Qwen/Qwen2.5-3B-Instruct"

    CHECKPOINT_ARGS+=("$HF_STEP_REPO")
done

if [ ${#CHECKPOINT_ARGS[@]} -eq 0 ]; then
    echo "Error: No checkpoints were successfully converted/uploaded"
    exit 1
fi

# Step 4: Reward evaluation with simple_evaluate.py
echo "Step 4: Running reward evaluation with simple_evaluate.py..."
cd src/ultrafeedback_largebatch

set +u
conda activate new_vllm
set -u

export VLLM_DISABLE_MEMORY_PROFILING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$EVAL_RESULTS_DIR"
EVAL_RESULTS_FILE="$EVAL_RESULTS_DIR/reward_evaluation_beta_${BETA}_eta_${ETA}_$(date +%Y%m%d_%H%M%S).txt"

if [ ${#CHECKPOINT_ARGS[@]} -eq 0 ]; then
    echo "Error: No checkpoints available for evaluation."
    exit 1
fi

LATEST_CHECKPOINT_REPO="${CHECKPOINT_ARGS[$((${#CHECKPOINT_ARGS[@]} - 1))]}"

MODELS=("$BASE_MODEL" "$LATEST_CHECKPOINT_REPO")
MODEL_NAMES=("Base" "$(basename "$LATEST_CHECKPOINT_REPO")")

echo "Evaluating the following models:"
for idx in "${!MODELS[@]}"; do
    echo "  $((idx + 1)). ${MODEL_NAMES[$idx]} -> ${MODELS[$idx]}"
done

EVAL_CMD=(python simple_evaluate.py)
EVAL_CMD+=(--models)
EVAL_CMD+=("${MODELS[@]}")

if [ ${#MODEL_NAMES[@]} -eq ${#MODELS[@]} ]; then
    EVAL_CMD+=(--model_names)
    EVAL_CMD+=("${MODEL_NAMES[@]}")
fi

EVAL_CMD+=(--dataset_name "$EVAL_DATASET_NAME")
EVAL_CMD+=(--reward_model "$REWARD_MODEL")
EVAL_CMD+=(--maxlen "$EVAL_MAXLEN")
EVAL_CMD+=(--n "$BEST_OF_N")
EVAL_CMD+=(--world_size "$EVAL_WORLD_SIZE")

if [ -n "$EVAL_MAX_SAMPLES" ]; then
    EVAL_CMD+=(--max_samples "$EVAL_MAX_SAMPLES")
fi

echo "Running: ${EVAL_CMD[*]}"
"${EVAL_CMD[@]}" | tee "$EVAL_RESULTS_FILE"

echo "Evaluation completed! Results saved to: $EVAL_RESULTS_FILE"

cd - > /dev/null

# Summary output
echo "=========================================="
echo "Reward Pipeline Finished Successfully!"
echo "=========================================="
echo "Training ETA: $ETA"
echo "Training BETA: $BETA"
echo "Total episodes: $TOTAL_EPISODES"
echo "Uploaded checkpoints: ${CHECKPOINT_ARGS[*]}"
echo "Evaluation results saved to: $EVAL_RESULTS_FILE"
echo "=========================================="

ELAPSED_SECONDS=$SECONDS
ELAPSED_HOURS=$((ELAPSED_SECONDS / 3600))
ELAPSED_MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
ELAPSED_REMAINDER=$((ELAPSED_SECONDS % 60))
printf "Total elapsed time: %02dh:%02dm:%02ds\n" "$ELAPSED_HOURS" "$ELAPSED_MINUTES" "$ELAPSED_REMAINDER"
