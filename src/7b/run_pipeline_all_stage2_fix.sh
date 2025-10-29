#!/usr/bin/env bash
set -euo pipefail

# Common configuration (override via environment variables)
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
RAW_REPO=${RAW_REPO:-"zjhhhh/qwen3b_sel4_cur6_base6_rescaled"}
# NOCHECK_REPO=${NOCHECK_REPO:-"zjhhhh/whole_sw_maxlen_8192_nocheck_rescale"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"zjhhhh/stage2"}

# Derived repos
PREPROCESSED_REPO=${PREPROCESSED_REPO:-"${OUTPUT_PREFIX}_preprocessed"}
MATCHED_NOCHECK_REPO=${MATCHED_NOCHECK_REPO:-"${PREPROCESSED_REPO}_nocheck_matched"}

# Tokenization and scoring
MAXLEN=${MAXLEN:-2048}
MAXLEN_PROMPT=${MAXLEN_PROMPT:-1024}
BETA=${BETA:-1.0}
SLICING_IDX=${SLICING_IDX:-24}
SCORE_TYPE=${SCORE_TYPE:-mean}

# Preprocess split control
TEST_SIZE=${TEST_SIZE:-1000}
SEED=${SEED:-42}
LIMIT_ROWS=${LIMIT_ROWS:-0}

# Stage2 intersection test control
REFERENCE_TEST_REPO=${REFERENCE_TEST_REPO:-"zjhhhh/stage1_preprocessed"}
ID_COLUMN=${ID_COLUMN:-"prompt"}

# Optional gap filtering
GAP_RATIO=${GAP_RATIO:-0}
GAP_SHUFFLE_SEED=${GAP_SHUFFLE_SEED:-}

# Fixed check text (used by fixed_expand)
FIXED_CHECK=${FIXED_CHECK:-"Does the response satisfy the following two criteria: 1) The response directly address the request without excessive or off-topic information not necessary for addressing the user's instruction? 2) The response should match the context and the instruction, whether it requires professionalism, friendliness, formality, or neutrality."}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# echo "[1/3] Preprocessing common dataset (stage2, intersection test) -> $PREPROCESSED_REPO"
# python preprocess_common_stage2.py \
#   --model "$MODEL" \
#   --input_repo "$RAW_REPO" \
#   --output_repo "$PREPROCESSED_REPO" \
#   --maxlen "$MAXLEN" \
#   --maxlen_prompt "$MAXLEN_PROMPT" \
#   --slicing_idx "$SLICING_IDX" \
#   --limit_rows "$LIMIT_ROWS" \
#   --use_intersection_test \
#   --reference_test_repo "$REFERENCE_TEST_REPO" \
#   --id_column "$ID_COLUMN"

echo "[1/2] Running fixed_expand filter (uses preprocessed splits)"
python filter_tokenize_judge_fixed_expand.py \
  --model "$MODEL" \
  --input_repo "$PREPROCESSED_REPO" \
  --maxlen "$MAXLEN" \
  --maxlen_prompt "$MAXLEN_PROMPT" \
  --beta "$BETA" \
  --slicing_idx "$SLICING_IDX" \
  --score_type "$SCORE_TYPE"

# echo "[3/3] Running fixed_expand filter -> prefix $OUTPUT_PREFIX (gap_ratio=$GAP_RATIO)"
# python filter_tokenize_judge_fixed_expand.py \
#   --model "$MODEL" \
#   --input_repo "$PREPROCESSED_REPO" \
#   --maxlen "$MAXLEN" \
#   --maxlen_prompt "$MAXLEN_PROMPT" \
#   --beta "$BETA" \
#   --slicing_idx "$SLICING_IDX" \
#   --score_type "$SCORE_TYPE" \
#   --fixed_check "$FIXED_CHECK" \
#   --output_repo_prefix "$OUTPUT_PREFIX" \
#   --limit_rows "$LIMIT_ROWS" \
#   --gap_ratio "$GAP_RATIO" \
#   ${GAP_SHUFFLE_SEED:+--gap_shuffle_seed "$GAP_SHUFFLE_SEED"}

echo "[2/2] Running min noexpand filter (uses preprocessed splits)"
python filter_tokenize_judge_min_fixed_expand.py \
  --model "$MODEL" \
  --input_repo "$PREPROCESSED_REPO" \
  --maxlen "$MAXLEN" \
  --maxlen_prompt "$MAXLEN_PROMPT" \
  --beta "$BETA" \
  --slicing_idx "$SLICING_IDX" \
  --score_type "$SCORE_TYPE"

# echo "[5/5] Matching nocheck with preprocessed prompts -> $MATCHED_NOCHECK_REPO and running nocheck_expand"
# python match_nocheck.py \
#   --preprocessed_repo "$PREPROCESSED_REPO" \
#   --nocheck_repo "$NOCHECK_REPO" \
#   --output_repo "$MATCHED_NOCHECK_REPO"

# python filter_tokenize_judge_nocheck_expand.py \
#   --model "$MODEL" \
#   --input_repo "$MATCHED_NOCHECK_REPO" \
#   --maxlen "$MAXLEN" \
#   --maxlen_prompt "$MAXLEN_PROMPT" \
#   --beta "$BETA" \
#   --slicing_idx "$SLICING_IDX" \
#   --score_type "$SCORE_TYPE" \
#   --output_repo_prefix "$OUTPUT_PREFIX" \
#   --limit_rows "$LIMIT_ROWS" \
#   --gap_ratio "$GAP_RATIO" \
#   ${GAP_SHUFFLE_SEED:+--gap_shuffle_seed "$GAP_SHUFFLE_SEED"}

echo "✓ All pipelines completed."
