#!/bin/bash

# Enhanced evaluation script runner
# This script runs the enhanced evaluation with multiple models and best-of-n sampling

# Change to the correct directory
cd /work2/lujingz/BoN_REBEL/src/ultrafeedback_largebatch

# Configuration - Multiple models to compare
MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "/work2/lujingz/model3b_v2"
    "/work2/lujingz/model3b_bon"
    # Add more modelsq here as needed
    # "/path/to/another/model"
)

# Optional: Custom names for the models (must match the number of models above)
MODEL_NAMES=(
    "Base-3B"
    "REBEL-3B"
    "REBEL-3B-bo2"
    # Add corresponding names here
    # "Other-Model"
)

# Other configuration
DATASET_NAME="MisDrifter/eval_3B_whole_data_armo_REBEL_tokenized_logprob"
REWARD_MODEL="RLHFlow/ArmoRM-Llama3-8B-v0.1"
MAXLEN=2048
N=1  # Best-of-N sampling (set to >1 for best-of-n)
WORLD_SIZE=1
MAX_SAMPLES=1000  # Set to limit samples for testing (set to empty for full dataset)

echo "============================================"
echo "Starting Enhanced Multi-Model Evaluation"
echo "============================================"
echo "Number of models: ${#MODELS[@]}"
echo "Models to compare:"
for i in "${!MODELS[@]}"; do
    if [ ${#MODEL_NAMES[@]} -gt $i ]; then
        echo "  $((i+1)). ${MODEL_NAMES[$i]}: ${MODELS[$i]}"
    else
        echo "  $((i+1)). Model_$((i+1)): ${MODELS[$i]}"
    fi
done
echo "Dataset: $DATASET_NAME"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Using first $MAX_SAMPLES samples for testing"
fi
echo "Best-of-$N sampling enabled"
echo "World size: $WORLD_SIZE"
echo ""

# Validate that we have at least 2 models
if [ ${#MODELS[@]} -lt 2 ]; then
    echo "Error: At least 2 models are required for comparison"
    echo "Please add more models to the MODELS array in this script"
    exit 1
fi

# Calculate number of pairwise comparisons
num_models=${#MODELS[@]}
num_comparisons=$((num_models * (num_models - 1) / 2))
echo "Total pairwise comparisons: $num_comparisons"
echo ""

# Build the command arguments
models_args="--models"
for model in "${MODELS[@]}"; do
    models_args="$models_args \"$model\""
done

names_args=""
if [ ${#MODEL_NAMES[@]} -eq ${#MODELS[@]} ]; then
    names_args="--model_names"
    for name in "${MODEL_NAMES[@]}"; do
        names_args="$names_args \"$name\""
    done
fi

# Debug output
echo "Debug - models_args: $models_args"
echo "Debug - names_args: $names_args"
echo ""

# Activate the vllm environment and run the evaluation
echo "Activating vllm environment..."
source /work2/lujingz/Anaconda3/etc/profile.d/conda.sh
conda activate vllm

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=2
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "Running enhanced evaluation..."
echo ""

# Construct and run the command
max_samples_arg=""
if [ -n "$MAX_SAMPLES" ]; then
    max_samples_arg="--max_samples $MAX_SAMPLES"
fi

if [ -n "$names_args" ]; then
    eval "python simple_evaluate.py $models_args $names_args --dataset_name \"$DATASET_NAME\" --reward_model \"$REWARD_MODEL\" --maxlen $MAXLEN --n $N --world_size $WORLD_SIZE $max_samples_arg"
else
    eval "python simple_evaluate.py $models_args --dataset_name \"$DATASET_NAME\" --reward_model \"$REWARD_MODEL\" --maxlen $MAXLEN --n $N --world_size $WORLD_SIZE $max_samples_arg"
fi

echo ""
echo "============================================"
echo "Enhanced Evaluation Completed!"
echo "============================================"

# Instructions for customization
echo ""
echo "To customize this evaluation:"
echo "1. Edit the MODELS array to add/remove models"
echo "2. Edit the MODEL_NAMES array to set custom names"
echo "3. Change N for best-of-N sampling (N>1)"
echo "4. Modify other parameters as needed"
echo ""
echo "Example usage with best-of-4 sampling:"
echo "N=4 ./run_simple_evaluate.sh"

# Optional: You can modify the variables above or pass them as command line arguments
# Usage examples:
# ./run_simple_evaluate.sh
# 
# Or modify the script to accept command line arguments:
# MY_MODEL="/path/to/your/model" ./run_simple_evaluate.sh 