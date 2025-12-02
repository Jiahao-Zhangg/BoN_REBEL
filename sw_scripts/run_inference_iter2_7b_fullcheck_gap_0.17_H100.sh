#!/bin/bash
# SLURM script to run fullcheck robust inference across 48 shards using 6 jobs on H100
# Each job runs 8 Python scripts on different single GPUs (0,1,2,3,4,5,6,7)
# Shards 0-47 (48 total shards)
# Usage: sbatch run_inference_iter2_7b_fullcheck_gap_0.17_H100.sh
#SBATCH --job-name=inference_iter2_7b_fullcheck_gap_0.17_H100
#SBATCH --output=logs/inference_iter2_7b_fullcheck_gap_0.17_H100_%A_%a.out
#SBATCH --error=logs/inference_iter2_7b_fullcheck_gap_0.17_H100_%A_%a.err
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --exclusive
#SBATCH --nodelist=ip-10-1-38-11,ip-10-1-81-8

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
# Jobs 0,2,4 -> Node 0; Jobs 1,3,5 -> Node 1
AVAILABLE_NODES=("ip-10-1-38-11" "ip-10-1-81-8")
NODE_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
ASSIGNED_NODE=${AVAILABLE_NODES[$NODE_INDEX]}

# Single GPUs for 8 parallel jobs per node
GPUS=("0" "1" "2" "3" "4" "5" "6" "7")

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node Index: $NODE_INDEX"
echo "Assigned Node: $ASSIGNED_NODE"
echo "Actual Node: $SLURMD_NODENAME"

# Verify we're on the correct node
if [ "$SLURMD_NODENAME" != "$ASSIGNED_NODE" ]; then
    echo "WARNING: Running on $SLURMD_NODENAME but expected $ASSIGNED_NODE"
fi

# Define paths
SCRIPT_PATH="src/checklist_judge_data_parallel/run_inference_on_shard_iter2_robust_fullchecks.py"
SHARD_DIR="./iter2_7b_fullcheck_gap_0.17_shards"
OUTPUT_DIR="./inference_scores_iter2_7b_fullcheck_gap_0.17"
JUDGE_MODEL="Qwen/Qwen3-14B"

# Calculate shard indices for this job (8 shards per job)
BASE_SHARD=$((SLURM_ARRAY_TASK_ID * 8))
SHARD_INDICES=($BASE_SHARD $((BASE_SHARD + 1)) $((BASE_SHARD + 2)) $((BASE_SHARD + 3)) $((BASE_SHARD + 4)) $((BASE_SHARD + 5)) $((BASE_SHARD + 6)) $((BASE_SHARD + 7)))

echo "=== Starting 8 parallel inference jobs ==="
echo "Base shard: $BASE_SHARD"
echo "Shard indices: ${SHARD_INDICES[@]}"
echo "GPUs: ${GPUS[@]}"

# Function to run inference on specific GPU with error isolation
run_inference() {
    local shard_idx=$1
    local gpu=$2
    local gpu_id=$3

    # Skip if shard index exceeds 47 (we only have 48 shards: 0-47)
    if [ $shard_idx -gt 47 ]; then
        echo "Skipping shard $shard_idx (exceeds maximum shard index 47)"
        return 0
    fi

    local shard_idx_padded=$(printf "%05d" $shard_idx)
    echo "Starting inference for shard $shard_idx on GPU $gpu (GPU ID: $gpu_id)"

    # Create separate log files for this shard
    local shard_log_out="logs/shard_${shard_idx_padded}_gpu${gpu_id}_${SLURM_JOB_ID}.out"
    local shard_log_err="logs/shard_${shard_idx_padded}_gpu${gpu_id}_${SLURM_JOB_ID}.err"

    # Run the Python script in a subshell with error isolation
    (
        # Set GPU assignment for this subprocess only
        export CUDA_VISIBLE_DEVICES=$gpu
        # Run the Python script
        python $SCRIPT_PATH \
            --idx $shard_idx \
            --shard_dir $SHARD_DIR \
            --judge_model $JUDGE_MODEL \
            --world_size 1 \
            --judge_type preference_5score \
            --selection_pairs 4 \
            --base_pairs 2 \
            --current_pairs 2 \
            --adversary_pairs 2 \
            --switch_position \
            --push_to_hub \
            --hf_repo_template zjhhhh/iter2_7b_fullcheck_gap_0.17_scores_{target}_{shard_idx} \
            --output_dir $OUTPUT_DIR
    ) > $shard_log_out 2> $shard_log_err &

    echo "Launched inference for shard $shard_idx on GPU $gpu (PID: $!)"
}

# Launch 8 parallel inference jobs on different single GPUs
for i in {0..7}; do
    shard_idx=${SHARD_INDICES[$i]}
    gpu=${GPUS[$i]}
    run_inference $shard_idx $gpu $i
done

# Wait for all background jobs to complete
echo "Waiting for all 8 inference jobs to complete..."
wait
echo "=== All inference jobs completed ==="
echo "Processed shards: ${SHARD_INDICES[@]}"


