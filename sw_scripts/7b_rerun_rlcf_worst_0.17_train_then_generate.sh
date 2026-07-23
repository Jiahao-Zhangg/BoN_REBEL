#!/bin/bash

# Submit worst-coordinate RLCF training, then start iteration-2 generation only
# after the training job (including checkpoint conversion and upload) succeeds.
# Run this submission wrapper from a login node with:
#   bash sw_scripts/7b_rerun_rlcf_worst_0.17_train_then_generate.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/7b_rerun_rlcf_worst_0.17.sh"
GENERATION_SCRIPT="${SCRIPT_DIR}/run_7b_rlcf_worst_rerun_iter2_generate.sh"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch is not available; run this script from a SLURM login node." >&2
    exit 1
fi

for script in "$TRAIN_SCRIPT" "$GENERATION_SCRIPT"; do
    if [[ ! -f "$script" ]]; then
        echo "Error: required script not found: $script" >&2
        exit 1
    fi
done

TRAIN_SUBMISSION="$(sbatch --parsable "$TRAIN_SCRIPT")"
TRAIN_JOB_ID="${TRAIN_SUBMISSION%%;*}"
if [[ ! "$TRAIN_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: could not parse training job ID from: $TRAIN_SUBMISSION" >&2
    exit 1
fi
echo "Submitted training job: $TRAIN_JOB_ID"

GENERATION_SUBMISSION="$(
    sbatch \
        --parsable \
        "--dependency=afterok:${TRAIN_JOB_ID}" \
        "$GENERATION_SCRIPT"
)"
GENERATION_JOB_ID="${GENERATION_SUBMISSION%%;*}"
if [[ ! "$GENERATION_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: could not parse generation job ID from: $GENERATION_SUBMISSION" >&2
    exit 1
fi

echo "Submitted generation array: $GENERATION_JOB_ID"
echo "Dependency: generation starts only after training job $TRAIN_JOB_ID succeeds."
echo "Monitor with: squeue -j ${TRAIN_JOB_ID},${GENERATION_JOB_ID}"
