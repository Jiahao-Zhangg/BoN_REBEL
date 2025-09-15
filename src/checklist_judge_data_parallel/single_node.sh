#!/bin/bash
#SBATCH --job-name=judge-shard0
#SBATCH --time=12:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=zhiweiw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -o slurm-%j.out

set -euo pipefail

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_vllm

# Set Hugging Face cache directory (from .bashrc)
export HF_HOME="/project/flame/jiahaoz4/.cache/huggingface"
export HF_DATASETS_CACHE="/project/flame/jiahaoz4/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="/project/flame/jiahaoz4/.cache/huggingface/hub"

DIR="/home/jiahaoz4/BoN_REBEL/src/checklist_judge_data_parallel"
PY="${DIR}/run_inference_on_shard.py"

# Config
SHARD_DIR="/project/flame/jiahaoz4/filtered_shards"
OUT_DIR="/project/flame/jiahaoz4/filtered_outputs"
JUDGE_MODEL="Qwen/Qwen3-14B"
JUDGE_TYPE="preference_5score"
WORLD_SIZE=8  # tensor_parallel_size

mkdir -p "${OUT_DIR}"

echo "Running shard idx=0 on single node with 8 GPUs"
python "${PY}" \
  --idx 0 \
  --shard_dir "${SHARD_DIR}" \
  --output_dir "${OUT_DIR}" \
  --judge_model "${JUDGE_MODEL}" \
  --judge_type "${JUDGE_TYPE}" \
  --selection_pairs 3 --base_pairs 2 --current_pairs 2 \
  --world_size "${WORLD_SIZE}" \
  --n_samples 5 --max_tokens 256 --temperature 0.6 --top_p 0.95 --top_k 20\
  --switch_position \
  --push_to_hub \
  --hf_repo_template "zjhhhh/filtered_subsampling_{shard_idx}"

echo "Done. Outputs are in ${OUT_DIR}"

