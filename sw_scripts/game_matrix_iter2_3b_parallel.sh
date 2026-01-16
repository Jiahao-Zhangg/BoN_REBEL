#!/bin/bash
#SBATCH --job-name=game_matrix_iter2_3b_parallel
#SBATCH --output=logs/game_matrix_iter2_3b_parallel_%A.out
#SBATCH --error=logs/game_matrix_iter2_3b_parallel_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodelist=ip-10-1-81-8
#SBATCH --exclusive

set -euo pipefail

mkdir -p logs

# Change to your BoN_REBEL directory
cd /fsx/gstevenw/testing_alignment_algos/BoN_REBEL
# Change to your conda environment path
source /fsx/gstevenw/miniconda3/etc/profile.d/conda.sh

PY=${PYTHON:-python}

conda activate new_vllm

INPUT_REPO="zjhhhh/stage2_preprocessed"
TEST_SPLIT="test"
START_IDX=0
#For full run
NUM_PROMPTS=500 
#For testing
# NUM_PROMPTS=2
if (( NUM_PROMPTS % 2 != 0 )); then
  echo "NUM_PROMPTS must be divisible by 2 for equal shards" >&2
  exit 1
fi
SHARD_SIZE=$(( NUM_PROMPTS / 2 ))

MODELS=(
  "zjhhhh/iter2_multi_adversary_step_367_final"
  "zjhhhh/iter2_ver2_pad_adversary_min_step_367_final"
  "zjhhhh/iter2_fullcheck_multi_adversary_eta_1e3_step_367_final"
  "Qwen/Qwen2.5-3B-Instruct"
  "zjhhhh/3b_rerun_rlcf_iter2_step_396_final"
)
ALIASES=(
  "3b_multi_iter2_game_matrix"
  "3b_min_iter2_game_matrix"
  "3b_fullcheck_iter2_game_matrix"
  "3b_base_game_matrix"
  "3b_rlcf_game_matrix"
)

if [[ "${#MODELS[@]}" -ne "${#ALIASES[@]}" ]]; then
  echo "MODELS and ALIASES length mismatch" >&2
  exit 1
fi

OUTPUT_BASE="outputs/game_matrix_iter2_3b"
# For full run
JUDGE_MODEL="Qwen/Qwen3-14B"
#For testing
# JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"

# Each shard writes to its own output directory to keep local indices aligned.

# -------------------------
# Step 1: Generation
# -------------------------
GPU_IDS_GEN=(0 1 2 3 4 5 6 7)
FREE_GPUS=("${GPU_IDS_GEN[@]}")
PIDS=()
declare -A PID_GPU

collect_finished() {
  local new_pids=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
    else
      local gpu="${PID_GPU[$pid]}"
      FREE_GPUS+=("$gpu")
      unset PID_GPU[$pid]
    fi
  done
  PIDS=("${new_pids[@]}")
}

wait_for_free_gpu() {
  while (( ${#FREE_GPUS[@]} == 0 )); do
    collect_finished
    if (( ${#FREE_GPUS[@]} == 0 )); then
      sleep 1
    fi
  done
}

for shard in 0 1; do
  shard_start=$(( START_IDX + shard * SHARD_SIZE ))
  shard_num=$SHARD_SIZE
  shard_out="${OUTPUT_BASE}/shard${shard}"
  mkdir -p "$shard_out"

  for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    alias="${ALIASES[$i]}"
    wait_for_free_gpu
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    echo "[gen] shard=${shard} model=${model} alias=${alias} gpu=${gpu} range=[$shard_start,$((shard_start + shard_num)))"
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PY" "src/checklist_judge_data_parallel/run_multi_model_generate_responses_robust.py" \
      --input_repo "$INPUT_REPO" \
      --test_split "$TEST_SPLIT" \
      --start_idx "$shard_start" \
      --num_prompts "$shard_num" \
      --models "$model" \
      --aliases "$alias" \
      --output_dir "$shard_out" \
      --push_to_hub \
      --hf_postfix "shard${shard}" \
      --response_world_size 1 &
    pid="$!"
    PIDS+=("$pid")
    PID_GPU[$pid]="$gpu"
  done
done

echo "Waiting for generation jobs (${#PIDS[@]})..."
wait
echo "Generation completed."

# -------------------------
# Step 2: Scoring
# -------------------------
GPU_IDS_SCORE=(0 1 2 3 4 5 6 7)
FREE_GPUS=("${GPU_IDS_SCORE[@]}")
PIDS=()
unset PID_GPU
declare -A PID_GPU

PAIR_IDX=(
  "0 1"
  "0 2"
  "0 3"
  "0 4"
  "1 2"
  "1 3"
  "1 4"
  "2 3"
  "2 4"
  "3 4"
)

for shard in 0 1; do
  shard_start=$(( START_IDX + shard * SHARD_SIZE ))
  shard_num=$SHARD_SIZE
  shard_out="${OUTPUT_BASE}/shard${shard}"
  responses_dir="${shard_out}/responses"

  for pair in "${PAIR_IDX[@]}"; do
    read -r i j <<< "$pair"
    model_a="${MODELS[$i]}"
    model_b="${MODELS[$j]}"
    alias_a="${ALIASES[$i]}"
    alias_b="${ALIASES[$j]}"
    wait_for_free_gpu
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    echo "[score] shard=${shard} pair=${alias_a}_vs_${alias_b} gpu=${gpu} range=[$shard_start,$((shard_start + shard_num)))"
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PY" "src/checklist_judge_data_parallel/run_multi_model_pairwise_robust.py" \
      --input_repo "$INPUT_REPO" \
      --test_split "$TEST_SPLIT" \
      --start_idx "$shard_start" \
      --num_prompts "$shard_num" \
      --models "${model_a},${model_b}" \
      --aliases "${alias_a},${alias_b}" \
      --output_dir "$shard_out" \
      --responses_dir "$responses_dir" \
      --judge_model "$JUDGE_MODEL" \
      --push_to_hub \
      --hf_postfix "shard${shard}" \
      --switch_position \
      --judge_world_size 1 &
    pid="$!"
    PIDS+=("$pid")
    PID_GPU[$pid]="$gpu"
  done
done

echo "Waiting for scoring jobs (${#PIDS[@]})..."
wait
echo "Scoring completed."
