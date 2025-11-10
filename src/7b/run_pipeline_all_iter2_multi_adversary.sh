#!/usr/bin/env bash
set -euo pipefail

# Common configuration (override via environment variables)
MODEL=${MODEL:-"Qwen/Qwen2.5-3B-Instruct"}
RAW_REPO=${RAW_REPO:-"zjhhhh/iter2_multi_scores_adversary_rescaled"}
# NOCHECK_REPO=${NOCHECK_REPO:-"zjhhhh/whole_sw_maxlen_8192_nocheck_rescale"}
# OUTPUT_PREFIX=${OUTPUT_PREFIX:-"zjhhhh/iter2_adversary"}
ADVERSARY_OUTPUT_PREFIX=${ADVERSARY_OUTPUT_PREFIX:-"zjhhhh/iter2_multi_adversary"}
# BASE_OUTPUT_PREFIX=${BASE_OUTPUT_PREFIX:-"zjhhhh/iter2_ver2_base"}

# Derived repos
# BASE_PREPROCESSED_REPO=${BASE_PREPROCESSED_REPO:-"zjhhhh/iter2_ver2_base_preprocessed"}
ADVERSARY_PREPROCESSED_REPO=${ADVERSARY_PREPROCESSED_REPO:-"${ADVERSARY_OUTPUT_PREFIX}_preprocessed"}
# MATCHED_NOCHECK_REPO=${MATCHED_NOCHECK_REPO:-"${PREPROCESSED_REPO}_nocheck_matched"}

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
REFERENCE_TEST_REPO=${REFERENCE_TEST_REPO:-"zjhhhh/stage2_preprocessed"}
ID_COLUMN=${ID_COLUMN:-"prompt"}

# Optional gap filtering
GAP_RATIO=${GAP_RATIO:-0.22}
GAP_SHUFFLE_SEED=${GAP_SHUFFLE_SEED:-42}

# Fixed check text (used by fixed_expand)
FIXED_CHECK=${FIXED_CHECK:-"Does the response satisfy the following two criteria: 1) The response directly address the request without excessive or off-topic information not necessary for addressing the user's instruction? 2) The response should match the context and the instruction, whether it requires professionalism, friendliness, formality, or neutrality."}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[1/2] Preprocessing common dataset (iter2, intersection test) -> $ADVERSARY_PREPROCESSED_REPO"
python preprocess_common_stage2.py \
  --model "$MODEL" \
  --input_repo "$RAW_REPO" \
  --output_repo "$ADVERSARY_PREPROCESSED_REPO" \
  --maxlen "$MAXLEN" \
  --maxlen_prompt "$MAXLEN_PROMPT" \
  --slicing_idx "$SLICING_IDX" \
  --limit_rows "$LIMIT_ROWS" \
  --use_intersection_test \
  --reference_test_repo "$REFERENCE_TEST_REPO" \
  --id_column "$ID_COLUMN"

# echo "[2/3] Running expand filter -> prefix $OUTPUT_PREFIX (gap_ratio=$GAP_RATIO)"
# python filter_tokenize_judge_fullcheck_expand.py \
#   --model "$MODEL" \
#   --input_repo "$PREPROCESSED_REPO" \
#   --maxlen "$MAXLEN" \
#   --maxlen_prompt "$MAXLEN_PROMPT" \
#   --beta "$BETA" \
#   --slicing_idx "$SLICING_IDX" \
#   --score_type "$SCORE_TYPE" \
#   --output_repo_prefix "$OUTPUT_PREFIX" \
#   --limit_rows "$LIMIT_ROWS" \
#   --gap_ratio "$GAP_RATIO" \
#   ${GAP_SHUFFLE_SEED:+--gap_shuffle_seed "$GAP_SHUFFLE_SEED"}

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
# echo "[2/2] Running min_expand filter for base -> prefix $BASE_OUTPUT_PREFIX (gap_ratio=$GAP_RATIO)"
# python filter_tokenize_judge_min_expand.py \
#   --model "$MODEL" \
#   --input_repo "$BASE_PREPROCESSED_REPO" \
#   --maxlen "$MAXLEN" \
#   --maxlen_prompt "$MAXLEN_PROMPT" \
#   --beta "$BETA" \
#   --slicing_idx "$SLICING_IDX" \
#   --score_type "$SCORE_TYPE" \
#   --output_repo_prefix "$BASE_OUTPUT_PREFIX" \
#   --limit_rows "$LIMIT_ROWS" \
#   --gap_ratio "$GAP_RATIO" \
#   ${GAP_SHUFFLE_SEED:+--gap_shuffle_seed "$GAP_SHUFFLE_SEED"}

echo "[2/2] Running multi_expand filter for adversary -> prefix $ADVERSARY_OUTPUT_PREFIX (gap_ratio=$GAP_RATIO)"
python filter_tokenize_judge_expand.py \
  --model "$MODEL" \
  --input_repo "$ADVERSARY_PREPROCESSED_REPO" \
  --maxlen "$MAXLEN" \
  --maxlen_prompt "$MAXLEN_PROMPT" \
  --beta "$BETA" \
  --slicing_idx "$SLICING_IDX" \
  --score_type "$SCORE_TYPE" \
  --output_repo_prefix "$ADVERSARY_OUTPUT_PREFIX" \
  --limit_rows "$LIMIT_ROWS" \
  --gap_ratio "$GAP_RATIO" \
  ${GAP_SHUFFLE_SEED:+--gap_shuffle_seed "$GAP_SHUFFLE_SEED"}

# echo "[3/3] Running min_expand filter for base -> prefix $OUTPUT_PREFIX (gap_ratio=$GAP_RATIO)"
# python filter_tokenize_judge_min_expand.py \
#   --model "$MODEL" \
#   --input_repo "$PREPROCESSED_REPO" \
#   --maxlen "$MAXLEN" \
#   --maxlen_prompt "$MAXLEN_PROMPT" \
#   --beta "$BETA" \
#   --slicing_idx "$SLICING_IDX" \
#   --score_type "$SCORE_TYPE" \
#   --output_repo_prefix "$BASE_OUTPUT_PREFIX" \
#   --limit_rows "$LIMIT_ROWS" \
#   --gap_ratio "$GAP_RATIO" \
#   ${GAP_SHUFFLE_SEED:+--gap_shuffle_seed "$GAP_SHUFFLE_SEED"}

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

