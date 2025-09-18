#!/bin/bash

# Training Pipeline Script
# This script runs the complete training pipeline including:
# 1. Training with accelerate
# 2. Converting checkpoint to FP32
# 3. Saving model to HuggingFace format

set -e  # Exit on any error

# Track total elapsed time  
SECONDS=0

# Change to BoN_REBEL directory as base path
cd /home/$USER/BoN_REBEL

# Activate conda environment
source /project/flame/$USER/miniconda3/etc/profile.d/conda.sh

# Configuration variables (modify as needed)
ETA="1e4"  # REBEL eta parameter
WORLD_SIZE="4"  # Number of GPUs/processes to use
OUTPUT_DIR="/project/flame/$USER/original_rebel/outputs_${ETA}"
HF_REPO_NAME="MisDrifter/qwen2.5_3B_Instruct_rebel_${ETA}"  # Change this to your desired repo name
EVAL_RESULTS_DIR="/project/flame/$USER/qwen2.5_3B_Instruct_rebel_evaluation_results"  # Directory to save evaluation results

echo "Starting training pipeline from BoN_REBEL directory..."
echo "Current working directory: $(pwd)"

# Create necessary directories
echo "Creating output directories..."
mkdir -p "$OUTPUT_DIR"        # Create full output directory (e.g., ../original_rebel/outputs_1e5)
mkdir -p "$EVAL_RESULTS_DIR"  # Create evaluation results directory

# Step 1: Training
echo "Step 1: Running training..."
conda activate rebel && \
accelerate launch \
    --config_file accelerate_cfgs/deepspeed_config_stage_3.yaml \
    --main-process-port 29081 \
    --num_processes "$WORLD_SIZE" \
    src/ultrafeedback_judge/rebel.py \
    --output_dir "$OUTPUT_DIR" \
    --rebel.eta "$ETA"

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "Training completed successfully!"

# Find the latest checkpoint directory (assuming it follows the pattern shown in the image)
LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -type d -name "ultrafeedback_rebel_*_step_*_final" | sort | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No final checkpoint found in $OUTPUT_DIR"
    exit 1
fi

echo "Found latest checkpoint: $LATEST_CHECKPOINT"

# Step 2: Convert checkpoint to FP32
echo "Step 2: Converting checkpoint to FP32..."
cd "$LATEST_CHECKPOINT"

if [ ! -f "zero_to_fp32.py" ]; then
    echo "Error: zero_to_fp32.py not found in $LATEST_CHECKPOINT"
    exit 1
fi

python "$LATEST_CHECKPOINT/zero_to_fp32.py" "$LATEST_CHECKPOINT" "$LATEST_CHECKPOINT/pytorch_model.bin"

if [ $? -ne 0 ]; then
    echo "Error: Checkpoint conversion failed"
    exit 1
fi

echo "Checkpoint conversion completed successfully!"

# Go back to BoN_REBEL directory
cd - > /dev/null

# Step 3: Save model using save_model.py (we're already in BoN_REBEL directory)
echo "Step 3: Saving model to HuggingFace format..."
python save_model.py \
    --checkpoint_path "$LATEST_CHECKPOINT" \
    --hf_repo "$HF_REPO_NAME" \
    --model "Qwen/Qwen2.5-3B-Instruct"

if [ $? -ne 0 ]; then
    echo "Error: Model saving failed"
    exit 1
fi

echo "Model saved successfully to HuggingFace repo: $HF_REPO_NAME"

# Step 4: Evaluation
echo "Step 4: Running evaluation..."
cd src/ultrafeedback_largebatch

# Evaluation configuration
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
TRAINED_MODEL="$LATEST_CHECKPOINT"  # Use the converted checkpoint path (contains pytorch_model.bin)
DATASET_NAME="zjhhhh/sw_maxlen_8192_mean_maxlenprompt_1024_tokenized_logprob"
REWARD_MODEL="RLHFlow/ArmoRM-Llama3-8B-v0.1"
MAXLEN=2048
N=3  # Best-of-N sampling
EVAL_WORLD_SIZE=4
MAX_SAMPLES=1000  # Limit samples for faster testing (remove for full evaluation)

echo "Evaluating trained model against base model..."
echo "Base model: $BASE_MODEL"
echo "Trained model: $TRAINED_MODEL"
echo "Dataset: $DATASET_NAME"

# Activate vllm environment for evaluation
echo "Activating new_vllm environment for evaluation..."
conda activate new_vllm

# Create evaluation results directory and file
mkdir -p "$EVAL_RESULTS_DIR"
EVAL_RESULTS_FILE="$EVAL_RESULTS_DIR/evaluation_eta_${ETA}_$(date +%Y%m%d_%H%M%S).txt"

echo "Evaluation results will be saved to: $EVAL_RESULTS_FILE"

# Run evaluation comparing base model vs trained model and save to file
python simple_evaluate.py \
    --models "$BASE_MODEL" "$TRAINED_MODEL" \
    --model_names "Base-3B" "REBEL-3B-eta-${ETA}" \
    --dataset_name "$DATASET_NAME" \
    --reward_model "$REWARD_MODEL" \
    --maxlen $MAXLEN \
    --n $N \
    --world_size $EVAL_WORLD_SIZE \
    --max_samples $MAX_SAMPLES | tee "$EVAL_RESULTS_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo "Evaluation completed successfully!"
echo "Results saved to: $EVAL_RESULTS_FILE"

# Go back to BoN_REBEL directory
cd - > /dev/null

echo "=========================================="
echo "Complete Pipeline Finished Successfully!"
echo "=========================================="
echo "Training ETA: $ETA"
echo "Model saved to HuggingFace repo: $HF_REPO_NAME"
echo "Final checkpoint location: $LATEST_CHECKPOINT"
echo "Evaluation results saved to: $EVAL_RESULTS_FILE"
echo "=========================================="

# Print total elapsed time
ELAPSED_SECONDS=$SECONDS
ELAPSED_HOURS=$((ELAPSED_SECONDS / 3600))
ELAPSED_MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
ELAPSED_REMAINDER=$((ELAPSED_SECONDS % 60))
printf "Total elapsed time: %02dh:%02dm:%02ds\n" "$ELAPSED_HOURS" "$ELAPSED_MINUTES" "$ELAPSED_REMAINDER"
