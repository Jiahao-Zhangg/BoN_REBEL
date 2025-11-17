#!/bin/bash
#SBATCH --job-name=7b_sweep_eta_1e6
#SBATCH --output=logs/7b_sweep_eta_1e6_%A.out
#SBATCH --error=logs/7b_sweep_eta_1e6_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodelist=ip-10-1-38-11

set -euo pipefail

mkdir -p logs

# Hugging Face caches (match interactive environment)
# export HF_HOME="/work2/jiahaoz4/.cache/huggingface"
# export HF_DATASETS_CACHE="/work2/jiahaoz4/.cache/huggingface/datasets"
# export HUGGINGFACE_HUB_CACHE="/work2/jiahaoz4/.cache/huggingface/hub"

# Change to your BoN_REBEL directory
cd /fsx/gstevenw/testing_alignment_algos/BoN_REBEL

# Change to your conda environment path
source /fsx/gstevenw/miniconda3/etc/profile.d/conda.sh

# Configuration variables (modify as needed)
ETA="1e6"                     # REBEL eta parameter
BETA="1"
TOTAL_EPISODES="56000"         # Episodes fed to rebel.py --total_episodes
TEST_MODE="false"             # If true, pass --test to rebel.py for quick dataset sampling
WORLD_SIZE="4"               # Number of GPUs/processes to use
export CUDA_VISIBLE_DEVICES="0,1,2,3"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
BON="true"                  # Toggle Best-of-N training behaviour for rebel.py
SEED="555134"               # Random seed for reproducible runs
JOB_RUN_ID="${JOB_RUN_ID:-$(date +%s)}"
TMP_BASE="/fsx/gstevenw/testing_alignment_algos/BoN_REBEL/tmp"
TMP_RUN_ROOT="${TMP_BASE%/}/7b_sweep_eta_1e6_${USER}/${JOB_RUN_ID}"
mkdir -p "$TMP_RUN_ROOT"

# Optional explicit WandB run name; keep original prefix, append compact timestamp
# If RUN_NAME is preset in the environment, it is used as-is.
RUN_ID="${RUN_ID:-$(date +%y%m%d%H%M)}"
RUN_NAME="${RUN_NAME:-7b_sweep_eta_1e6_${RUN_ID}}"

LOG_DIR="${REBEL_LOG_DIR:-../logs}"
LOG_OUT="${LOG_DIR%/}/7b_sweep_eta_1e6.out"
LOG_ERR="${LOG_DIR%/}/7b_sweep_eta_1e6.err"
mkdir -p "$LOG_DIR"

# Mirror stdout/stderr to log files while keeping console output
exec > >(tee -a "$LOG_OUT")
exec 2> >(tee -a "$LOG_ERR" >&2)

SECONDS=0
OUTPUT_DIR="${TMP_RUN_ROOT}/outputs_seed_${SEED}_eta_${ETA}"
HF_REPO_NAME="zjhhhh/7b_sweep_eta_1e6"

############################
# Training configuration   #
############################
GRADIENT_ACCUMULATION_STEPS=$((128 / WORLD_SIZE))
TRAIN_INPUT_REPO="zjhhhh/7b_iter1_mean_beta_1.0_multi_expand_tokenized_gap_ratio_0.22"
# Base model used to initialize training
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

mkdir -p "$OUTPUT_DIR"


echo "Starting reward training pipeline from $(pwd)"
echo "Checkpoints will be written to temporary storage: $OUTPUT_DIR"

if [[ "${TEST_MODE,,}" == "true" ]]; then
    echo "Test mode enabled: reducing total episodes to 256 for a quick run"
    TOTAL_EPISODES="256"
fi

# Step 1: Training
echo "Step 1: Running training..."
set +u
conda activate rebel
set -u
TRAIN_CMD=(
    python -m accelerate.commands.launch
    --config_file accelerate_cfgs/deepspeed_config_stage_3.yaml
    --main-process-port 29082
    --num_processes "$WORLD_SIZE"
    src/ultrafeedback_judge/rebel_save.py
    --output_dir "$OUTPUT_DIR"
    --base_model "$BASE_MODEL"
    --rebel.eta "$ETA"
    --total_episodes "$TOTAL_EPISODES"
    --task.input_repo "$TRAIN_INPUT_REPO"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --run_name "$RUN_NAME"
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

# Checkpoints are saved as "${RUN_NAME}_step_<N>[_final]" under $OUTPUT_DIR
mapfile -d '' CHECKPOINT_DIRS < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "${RUN_NAME}_step_*" -print0 | sort -z -V)

# Fallback to legacy pattern if none are found (backward compatibility)
if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    mapfile -d '' CHECKPOINT_DIRS < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "ultrafeedback_rebel_*" -print0 | sort -z -V)
fi

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

    if python save_model.py \
        --checkpoint_path "$CKPT_DIR" \
        --hf_repo "$HF_STEP_REPO" \
        --model "$BASE_MODEL"; then
        CHECKPOINT_ARGS+=("$HF_STEP_REPO")
        echo "Upload succeeded for $CKPT_DIR; deleting checkpoint to save space"
        rm -rf "$CKPT_DIR"
    else
        echo "Error: Upload failed for $CKPT_DIR; leaving checkpoint in place" >&2
        exit 1
    fi
done

if [ ${#CHECKPOINT_ARGS[@]} -eq 0 ]; then
    echo "Error: No checkpoints were successfully converted/uploaded"
    exit 1
fi

# Summary output
echo "=========================================="
echo "Reward Pipeline Finished Successfully!"
echo "=========================================="
echo "Training ETA: $ETA"
echo "Total episodes: $TOTAL_EPISODES"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Seed: $SEED"
echo "Uploaded checkpoints: ${CHECKPOINT_ARGS[*]}"
echo "=========================================="

ELAPSED_SECONDS=$SECONDS
ELAPSED_HOURS=$((ELAPSED_SECONDS / 3600))
ELAPSED_MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
ELAPSED_REMAINDER=$((ELAPSED_SECONDS % 60))
printf "Total elapsed time: %02dh:%02dm:%02ds\n" "$ELAPSED_HOURS" "$ELAPSED_MINUTES" "$ELAPSED_REMAINDER"
