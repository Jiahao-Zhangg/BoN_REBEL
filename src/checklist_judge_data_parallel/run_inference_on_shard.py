import os
import json
import time
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from datasets import load_from_disk
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# ------------------ Structured outputs for guided decoding ------------------
class PreferenceBinaryOutput(BaseModel):
    explanation: str
    verdict: Literal["A", "B"]


PREFERENCE_BINARY_GUIDED_DECODING = GuidedDecodingParams(
    json=PreferenceBinaryOutput.model_json_schema()
)


class PreferenceTernaryOutput(BaseModel):
    explanation: str
    verdict: Literal["A", "B", "Tie"]


PREFERENCE_TERNARY_GUIDED_DECODING = GuidedDecodingParams(
    json=PreferenceTernaryOutput.model_json_schema()
)


class PreferenceScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=0, le=10)


PREFERENCE_SCORE_GUIDED_DECODING = GuidedDecodingParams(
    json=PreferenceScoreOutput.model_json_schema()
)


class Preference5ScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=-1, le=4)


PREFERENCE_5SCORE_GUIDED_DECODING = GuidedDecodingParams(
    json=Preference5ScoreOutput.model_json_schema()
)


# ------------------ Utilities ------------------
def set_seed(seed=5775709):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_winner(values: List[str]):
    counts = Counter(values)
    counts = {k: counts.get(k, 0) for k in ["A", "B", "Tie"]}

    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]

    if len(winners) == 3:
        return "Tie"
    if "Tie" in winners and len(winners) == 2:
        return next(k for k in winners if k != "Tie")
    if set(winners) == {"A", "B"}:
        return "Tie"
    return winners[0]


def is_valid_response(response, judge_type):
    if judge_type == "reward":
        try:
            score = int(response)
            return 0 <= score <= 100
        except Exception:
            return False
    elif judge_type == "preference_binary":
        return response in ["A", "B"]
    elif judge_type == "preference_ternary":
        return response in ["A", "B", "Tie"]
    elif judge_type == "preference_score":
        try:
            score = int(response)
            return 0 <= score <= 10
        except Exception:
            return False
    elif judge_type == "preference_5score":
        try:
            score = int(response)
            return -1 <= score <= 4
        except Exception:
            return False
    else:
        return True


def filter_valid_responses(responses, judge_type):
    return [r for r in responses if is_valid_response(r, judge_type)]


def reverse_score(score, judge_type):
    if judge_type == "preference_5score":
        if score == -1:
            return -1
        else:
            return 4 - int(score)
    elif judge_type in ["preference_binary", "preference_ternary"]:
        if score == "A":
            return "B"
        elif score == "B":
            return "A"
        else:
            return "Tie"
    else:
        return score


def extract_verdict(response_text: str):
    try:
        parsed = json.loads(response_text)
    except Exception:
        return None
    return parsed.get("verdict", None)


def get_message(instruction: str):
    return [{"role": "user", "content": instruction}]


def majority_numeric(ints):
    """Return majority of a numeric list; if tie among modes, return mean of tied values.
    Expects a non-empty list of ints. Returns int when unique mode; float when tie-averaged.
    """
    counts = Counter(ints)
    max_c = max(counts.values())
    modes = [k for k, c in counts.items() if c == max_c]
    if len(modes) == 1:
        return int(modes[0])
    # Average of tied modes
    return float(sum(modes) / len(modes))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run judge inference on a prepared shard and output 20 scores per prompt (mean and majority per pair).")
    parser.add_argument("--idx", type=int, required=True,
                        help="Shard index to load (matches prepare_shards naming)")
    parser.add_argument("--shard_dir", type=str, default="./local_shards",
                        help="Directory containing shard_* folders")

    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--judge_type", type=str, default="preference_5score",
                        choices=["preference_binary", "preference_ternary", "preference_score", "preference_5score"])

    parser.add_argument("--selection_pairs", type=int, default=3, help="number of selection responses")
    parser.add_argument("--base_pairs", type=int, default=2, help="number of base responses")
    parser.add_argument("--current_pairs", type=int, default=2, help="number of current responses")

    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--switch_position", action="store_true", default=False,
                        help="Collect preferences in both directions to mitigate positional bias")

    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to write JSONL results; one row per prompt with 10 scores")
    return parser.parse_args()


def main():
    st = time.time()
    args = parse_args()

    shard_path = os.path.join(args.shard_dir, f"shard_{args.idx:05d}")
    if not os.path.isdir(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    ds = load_from_disk(shard_path)
    n_rows = len(ds)
    print(f"Loaded shard {args.idx} from {shard_path} with {n_rows} prompts")

    # Prepare prompt template and guided decoding
    if args.judge_type == "preference_binary":
        filename = "prompt_preference_binary.txt"
        guided_decoding = PREFERENCE_BINARY_GUIDED_DECODING
    elif args.judge_type == "preference_ternary":
        filename = "prompt_preference_ternary.txt"
        guided_decoding = PREFERENCE_TERNARY_GUIDED_DECODING
    elif args.judge_type == "preference_score":
        filename = "prompt_preference_score.txt"
        guided_decoding = PREFERENCE_SCORE_GUIDED_DECODING
    elif args.judge_type == "preference_5score":
        filename = "prompt_preference_5score_explanation.txt"
        guided_decoding = PREFERENCE_5SCORE_GUIDED_DECODING
    with open(Path(__file__).parent / filename, "r") as f:
        prompt_template = f.read()

    # Validate expected response columns exist
    expected_cols = ["prompt"]
    selection_cols = [f"selection_response_{i+1}" for i in range(args.selection_pairs)]
    base_cols = [f"base_response_{j+1}" for j in range(args.base_pairs)]
    current_cols = [f"current_response_{k+1}" for k in range(args.current_pairs)]
    needed = selection_cols + base_cols + current_cols
    for col in expected_cols + needed:
        if col not in ds.column_names:
            raise ValueError(
                f"Missing required column '{col}' in shard dataset. Available columns: {ds.column_names}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.world_size,
    )

    # Common sampling params
    set_seed(0)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n_samples,
        max_tokens=args.max_tokens,
        seed=0,
        guided_decoding=guided_decoding,
    )

    # Pre-allocate result holders for each pair
    # We'll fill per pair with length-n_rows lists for both mean and majority
    pair_results = {}

    # Helper to run a batch of prompts and reduce to per-row score
    def run_pair_and_reduce(resp_a_list: List[str], resp_b_list: List[str], label: str):
        prompts = []
        for row_idx in range(n_rows):
            row = ds[row_idx]
            prompt = row["prompt"]
            resp_a = resp_a_list[row_idx]
            resp_b = resp_b_list[row_idx]
            check_val = row.get("check", "")  # Some templates include {check}
            filled = prompt_template.format(
                prompt=prompt,
                response_a=resp_a,
                response_b=resp_b,
                check=check_val,
            )
            messages = get_message(filled)
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        responses = llm.generate(prompts, sampling_params)
        # Convert to list of per-row raw texts per sample
        texts = [[o.text for o in r.outputs] for r in responses]
        # Extract verdicts and filter
        verdict_lists = []
        for per_row in texts:
            vals = [extract_verdict(t) for t in per_row]
            vals = [v for v in vals if v is not None]
            vals = filter_valid_responses(vals, args.judge_type)
            verdict_lists.append(vals)

        if args.judge_type in ["preference_score", "preference_5score"]:
            reduced_mean = []
            reduced_majority = []
            for vals in verdict_lists:
                # Convert to ints and filter -1 for 5score
                ints = [int(v) for v in vals]
                if args.judge_type == "preference_5score":
                    ints = [v for v in ints if v != -1]
                if len(ints) == 0:
                    reduced_mean.append(None)
                    reduced_majority.append(None)
                else:
                    reduced_mean.append(float(np.mean(ints)))
                    reduced_majority.append(majority_numeric(ints))
        else:
            # Categorical preference: we only define majority via get_winner; mean is not applicable
            reduced_majority = [get_winner(vals) if len(vals) > 0 else None for vals in verdict_lists]
            reduced_mean = [None for _ in range(n_rows)]

        if args.switch_position:
            # Also judge reversed A/B and combine
            prompts_sw = []
            for row_idx in range(n_rows):
                row = ds[row_idx]
                prompt = row["prompt"]
                resp_a = resp_b_list[row_idx]
                resp_b = resp_a_list[row_idx]
                check_val = row.get("check", "")
                filled = prompt_template.format(
                    prompt=prompt,
                    response_a=resp_a,
                    response_b=resp_b,
                    check=check_val,
                )
                messages = get_message(filled)
                prompts_sw.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            responses_sw = llm.generate(prompts_sw, sampling_params)
            texts_sw = [[o.text for o in r.outputs] for r in responses_sw]
            verdict_lists_sw = []
            for per_row in texts_sw:
                vals = [extract_verdict(t) for t in per_row]
                vals = [v for v in vals if v is not None]
                vals = filter_valid_responses(vals, args.judge_type)
                verdict_lists_sw.append(vals)

            if args.judge_type in ["preference_score", "preference_5score"]:
                # Build combined lists of ints per row, then compute both mean and majority
                combined_int_lists = []
                for orig_vals, sw_vals in zip(verdict_lists, verdict_lists_sw):
                    ints = [int(v) for v in orig_vals]
                    sw_ints = [int(v) for v in sw_vals]
                    if args.judge_type == "preference_5score":
                        ints = [v for v in ints if v != -1]
                        sw_ints = [v for v in sw_ints if v != -1]
                    # Reverse switch scores for positional bias
                    sw_ints = [reverse_score(v, args.judge_type) for v in sw_ints]
                    combined_int_lists.append(ints + sw_ints)

                new_mean = []
                new_majority = []
                for ints in combined_int_lists:
                    if len(ints) == 0:
                        new_mean.append(None)
                        new_majority.append(None)
                    else:
                        new_mean.append(float(np.mean(ints)))
                        new_majority.append(majority_numeric(ints))

                reduced_mean = new_mean
                reduced_majority = new_majority
            else:
                # Categorical: recompute winner on doubled sample lists
                combined_samples = []
                for vals_a, vals_b in zip(verdict_lists, verdict_lists_sw):
                    rb = [reverse_score(v, args.judge_type) for v in vals_b]
                    combined_samples.append(vals_a + rb)
                reduced_majority = [get_winner(vals) if len(vals) > 0 else None for vals in combined_samples]
                reduced_mean = [None for _ in range(n_rows)]

        pair_results[label + "_mean"] = reduced_mean
        pair_results[label + "_majority"] = reduced_majority

    # Prepare lists of response strings per row for each column group
    sel_lists = {col: [ds[i][col] for i in range(n_rows)] for col in selection_cols}
    base_lists = {col: [ds[i][col] for i in range(n_rows)] for col in base_cols}
    cur_lists = {col: [ds[i][col] for i in range(n_rows)] for col in current_cols}

    # Run selection vs base (3x2=6)
    for i, sel_col in enumerate(selection_cols, start=1):
        for j, base_col in enumerate(base_cols, start=1):
            label = f"selection_{i}_base_{j}_score"
            run_pair_and_reduce(sel_lists[sel_col], base_lists[base_col], label)

    # Run current vs base (2x2=4)
    for k, cur_col in enumerate(current_cols, start=1):
        for j, base_col in enumerate(base_cols, start=1):
            label = f"current_{k}_base_{j}_score"
            run_pair_and_reduce(cur_lists[cur_col], base_lists[base_col], label)

    # Assemble output rows: prompt + 10 scores
    base_labels = [
        "selection_1_base_1_score",
        "selection_1_base_2_score",
        "selection_2_base_1_score",
        "selection_2_base_2_score",
        "selection_3_base_1_score",
        "selection_3_base_2_score",
        "current_1_base_1_score",
        "current_1_base_2_score",
        "current_2_base_1_score",
        "current_2_base_2_score",
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard_{args.idx:05d}_scores.jsonl")
    with open(out_path, "w") as f:
        for row_idx in range(n_rows):
            row = {"prompt": ds[row_idx]["prompt"]}
            for key in base_labels:
                row[key.replace("_score", "_mean")] = pair_results.get(key + "_mean", [None] * n_rows)[row_idx]
                row[key.replace("_score", "_majority")] = pair_results.get(key + "_majority", [None] * n_rows)[row_idx]
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {n_rows} rows -> {out_path}")
    print(f"Time taken: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
