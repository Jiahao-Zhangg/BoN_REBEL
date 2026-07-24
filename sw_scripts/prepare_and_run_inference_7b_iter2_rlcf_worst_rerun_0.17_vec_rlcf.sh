#!/bin/bash
# Prepare the filtered worst-RLCF iter2 dataset and launch all 72 scoring shards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

INPUT_REPO="zjhhhh/generation_for_iter2_7b_rlcf_worst_rerun_0.17_filtered"
SHARD_DIR="./7b_iter2_rlcf_worst_rerun_0.17_vec_rlcf_shards"
SHARD_SIZE=700
EXPECTED_SHARDS=72

python src/checklist_judge_data_parallel/prepare_shards.py \
    --input_repo "$INPUT_REPO" \
    --split train \
    --shard_size "$SHARD_SIZE" \
    --out_dir "$SHARD_DIR"

ACTUAL_SHARDS=$(find "$SHARD_DIR" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | wc -l)
if [[ "$ACTUAL_SHARDS" -ne "$EXPECTED_SHARDS" ]]; then
    echo "Error: expected $EXPECTED_SHARDS shards in $SHARD_DIR, found $ACTUAL_SHARDS" >&2
    exit 1
fi

A100_JOB_ID=$(sbatch --parsable \
    sw_scripts/run_inference_7b_iter2_rlcf_worst_rerun_0.17_vec_rlcf_A100.sh)
H100_JOB_ID=$(sbatch --parsable \
    sw_scripts/run_inference_7b_iter2_rlcf_worst_rerun_0.17_vec_rlcf_H100.sh)

echo "Prepared $ACTUAL_SHARDS shards from $INPUT_REPO"
echo "Submitted A100 job array: $A100_JOB_ID (shards 48-71)"
echo "Submitted H100 job array: $H100_JOB_ID (shards 0-47)"
