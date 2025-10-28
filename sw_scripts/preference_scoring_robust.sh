#!/bin/bash

# SLURM script to run inference across all 26 shards using 7 jobs
# Each job runs 4 Python scripts on different GPU pairs (0,1; 2,3; 4,5; 6,7)
# Usage: sbatch run_all_shards_block.sh

#SBATCH --job-name=inference_multi_gpu
#SBATCH --output=logs/inference_%A_%a.out
#SBATCH --error=logs/inference_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=ml.p5.48xlarge,ml.p4d.24xlarge
#SBATCH --exclusive

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
echo "Activating conda environment: new_vllm"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

# Verify environment is activated
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Available nodes for assignment
AVAILABLE_NODES=("ip-10-1-38-11" "ip-10-1-81-8" "ip-10-1-173-179" "ip-10-1-184-205" "ip-10-1-196-96" "ip-10-1-226-48" "ip-10-1-231-85")
ASSIGNED_NODE=${AVAILABLE_NODES[$SLURM_ARRAY_TASK_ID]}

# GPU pairs for 4 parallel jobs per node
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned Node: $ASSIGNED_NODE"
echo "Actual Node: $SLURMD_NODENAME"

# Define paths
SCRIPT_PATH="src/checklist_judge_data_parallel/run_inference_on_shard_block_robust.py"
SHARD_DIR="./shards_qwen2.5_3b_base_488_2000"
OUTPUT_DIR="./inference_scores_qwen2.5_3b_base_488"
JUDGE_MODEL="Qwen/Qwen3-14B"

# Calculate shard indices for this job (4 shards per job)
BASE_SHARD=$((SLURM_ARRAY_TASK_ID * 4))
SHARD_INDICES=($BASE_SHARD $((BASE_SHARD + 1)) $((BASE_SHARD + 2)) $((BASE_SHARD + 3)))

echo "=== Starting 4 parallel inference jobs ==="
echo "Base shard: $BASE_SHARD"
echo "Shard indices: ${SHARD_INDICES[@]}"
echo "GPU pairs: ${GPU_PAIRS[@]}"

# Function to run inference on specific GPU pair with error isolation
run_inference() {
    local shard_idx=$1
    local gpu_pair=$2
    local gpu_id=$3
    
    # Skip if shard index exceeds 25 (we only have 26 shards: 0-25)
    if [ $shard_idx -gt 25 ]; then
        echo "Skipping shard $shard_idx (exceeds maximum shard index 25)"
        return 0
    fi
    
    local shard_idx_padded=$(printf "%05d" $shard_idx)
    
    echo "Starting inference for shard $shard_idx on GPUs $gpu_pair (GPU ID: $gpu_id)"
    
    # Create separate log files for this shard
    local shard_log_out="logs/shard_${shard_idx_padded}_gpu${gpu_id}_${SLURM_JOB_ID}.out"
    local shard_log_err="logs/shard_${shard_idx_padded}_gpu${gpu_id}_${SLURM_JOB_ID}.err"
    
    # Run the Python script in a subshell with error isolation
    (
        # Set GPU assignment for this subprocess only
        export CUDA_VISIBLE_DEVICES=$gpu_pair
        
        # Run the Python script
        python $SCRIPT_PATH \
            --idx $shard_idx \
            --shard_dir $SHARD_DIR \
            --judge_model $JUDGE_MODEL \
            --world_size 2 \
            --judge_type preference_5score \
            --selection_pairs 4 \
            --base_pairs 8 \
            --current_pairs 8 \
            --n_samples 5 \
            --max_tokens 256 \
            --temperature 0.6 \
            --top_p 0.95 \
            --top_k 20 \
            --switch_position \
            --push_to_hub \
            --hf_repo_template zjhhhh/qwen3b_base_488_shard_${shard_idx_padded} \
            --output_dir $OUTPUT_DIR
    ) > $shard_log_out 2> $shard_log_err &
    
    echo "Launched inference for shard $shard_idx on GPUs $gpu_pair (PID: $!)"
}

# Launch 4 parallel inference jobs on different GPU pairs
for i in {0..3}; do
    shard_idx=${SHARD_INDICES[$i]}
    gpu_pair=${GPU_PAIRS[$i]}
    run_inference $shard_idx $gpu_pair $i
done

# Wait for all background jobs to complete
echo "Waiting for all 4 inference jobs to complete..."
wait

echo "=== All inference jobs completed ==="
echo "Processed shards: ${SHARD_INDICES[@]}"