import os
import json
import time
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from datasets import load_from_disk, Dataset
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# ------------------ Structured outputs for guided decoding ------------------
PREFERENCE_BASELINE_GUIDED_DECODING = GuidedDecodingParams(choice=[str(i) for i in range(-1, 101)])

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
    if judge_type == "baseline":
        try:
            score = int(response)
            return 0 <= score <= 100
        except Exception:
            return False


def filter_valid_responses(responses, judge_type):
    return [r for r in responses if is_valid_response(r, judge_type)]


def get_message(instruction: str):
    return [{"role": "user", "content": instruction}]


def get_numeric_mode(values, score_range=None):
    """
    Get the mode (most frequent value) from a list of numeric values.
    If multiple modes exist, return the mean of the tied modes.
    """
    if not values:
        return None

    # Convert to ints and optionally filter by a valid range
    if score_range is not None:
        min_s, max_s = score_range
        values = [int(v) for v in values if min_s <= int(v) <= max_s]
    else:
        values = [int(v) for v in values]

    if not values:
        return None

    counts = Counter(values)
    max_c = max(counts.values())
    modes = [k for k, c in counts.items() if c == max_c]
    if len(modes) == 1:
        return modes[0]
    # mean tie-break among tied modes
    return float(sum(modes) / len(modes))


def weighted_average(values, weights):
    pairs = []
    for v, w in zip(values, weights):
        if v is None or w is None:
            continue
        if w <= 0:
            continue
        pairs.append((float(v), float(w)))

    if not pairs:
        return None

    total_weight = sum(w for _, w in pairs)
    if total_weight == 0:
        return None

    return sum(v * w for v, w in pairs) / total_weight


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run judge inference on a prepared shard and output 20 scores per prompt (mean and majority per pair).")
    parser.add_argument("--idx", type=int, required=True,
                        help="Shard index to load (matches prepare_shards naming)")
    parser.add_argument("--shard_dir", type=str, default="./local_shards",
                        help="Directory containing shard_* folders")

    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--judge_type", type=str, default="baseline",
                        choices=["baseline"])

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
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="If set, also pushes the output rows to Hugging Face Hub as a dataset")
    parser.add_argument("--hf_repo_template", type=str, default="zjhhhh/subsampling_{shard_idx}",
                        help="Template for target HF repo id. {shard_idx} will be replaced with the shard index.")
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
    if args.judge_type == "baseline":
        filename = "prompt_baseline.txt"
        guided_decoding = PREFERENCE_BASELINE_GUIDED_DECODING
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

    if "requirements" not in ds.column_names:
        raise ValueError("Input shard is missing 'requirements' column needed to extract checks per prompt.")

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

    # Expand each row by splitting its requirements into per-check rows.
    # Keep track of mapping back to the original row via 'orig_index'.
    expanded = []
    for orig_idx in range(n_rows):
        row = ds[orig_idx]
        req_str = row["requirements"]
        counter = 1
        chunks: List[str] = []
        while len(req_str) > 0:
            assert req_str.startswith(f"{counter})"), (
                f"Malformed requirements format at row {orig_idx}: expected prefix '{counter})' but got: {req_str[:20]}...")
            marker = f"/100)\n{counter+1})"
            pos = req_str.find(marker)
            if pos > 0:
                curr = req_str[len(f"{counter})"): pos + len("/100)\n")]
            else:
                curr = req_str[len(f"{counter})"):]
            chunks.append(curr)
            # advance
            req_str = req_str[len(curr) + len(f"{counter})"):]
            counter += 1
        # normalize
        chunks = [c.strip() for c in chunks]

        for c in chunks:
            new_row = {k: row[k] for k in ds.column_names}
            new_row["check"] = c.split("(importance:")[0].strip()
            try:
                new_row["importance"] = int(c.split("(importance:")[1].split("/")[0].strip())
            except Exception:
                new_row["importance"] = None
            new_row["orig_index"] = orig_idx
            expanded.append(new_row)

    eds: Dataset = Dataset.from_list(expanded)
    n_expanded = len(eds)
    print(f"Expanded to {n_expanded} check-rows across {n_rows} prompts")

    # Pre-allocate result holders for each pair over expanded rows
    pair_results = {}

    # Helper to run a batch of prompts and reduce to per-row score
    def run_pair_and_reduce(resp_list: List[str], label: str):
        prompts = []
        for row_idx in range(n_expanded):
            row = eds[row_idx]
            prompt = row["prompt"]
            resp = resp_list[row_idx]
            check_val = row.get("check", "")
            filled = prompt_template.format(
                instruction=prompt,
                response=resp,
                requirement=check_val,
            )
            messages = get_message(filled)
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        responses = llm.generate(prompts, sampling_params)

        reduced_mean = []
        reduced_majority = []
        score_range = (0, 100)
        expected_samples = sampling_params.n or args.n_samples

        for req_out in responses:
            raw_vals = []
            for output in req_out.outputs[:expected_samples]:
                text = output.text.strip()
                if not text:
                    continue
                raw_vals.append(text.splitlines()[0].strip())

            valid_vals = filter_valid_responses(raw_vals, args.judge_type)
            if not valid_vals:
                reduced_mean.append(None)
                reduced_majority.append(None)
                continue

            numeric_vals = [int(v) for v in valid_vals]
            reduced_mean.append(float(np.mean(numeric_vals)))
            reduced_majority.append(get_numeric_mode(numeric_vals, score_range))

        if len(reduced_mean) != n_expanded:
            # Maintain alignment with expanded rows; fill with None if generation under-produced.
            if len(reduced_mean) < n_expanded:
                deficit = n_expanded - len(reduced_mean)
                reduced_mean.extend([None] * deficit)
                reduced_majority.extend([None] * deficit)
            else:
                reduced_mean = reduced_mean[:n_expanded]
                reduced_majority = reduced_majority[:n_expanded]

        pair_results[label + "_mean"] = reduced_mean
        pair_results[label + "_majority"] = reduced_majority

    # Prepare lists of response strings per expanded row
    sel_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in selection_cols}
    base_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in base_cols}
    cur_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in current_cols}
    
    # Run selection
    for i, sel_col in enumerate(selection_cols, start=1):
        run_pair_and_reduce(sel_lists[sel_col], f"selection_{i}_score")

    # Run base
    for j, base_col in enumerate(base_cols, start=1):
        run_pair_and_reduce(base_lists[base_col], f"base_{j}_score")
    
    # Run current
    for k, cur_col in enumerate(current_cols, start=1):
        run_pair_and_reduce(cur_lists[cur_col], f"current_{k}_score")


    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard_{args.idx:05d}_scores.jsonl")
    out_rows = []
    with open(out_path, "w") as f:
        # Build mapping from orig_index -> list of expanded indices in order
        idx_map = {}
        for i in range(n_expanded):
            oi = eds[i]["orig_index"]
            idx_map.setdefault(oi, []).append(i)

        for oi in range(n_rows):
            # Start with all original columns
            base_row = {k: ds[oi][k] for k in ds.column_names}
            # For each pair key, collect vector over the checks of this prompt
            for key in ["selection_1_score", "base_1_score", "current_1_score", "selection_2_score", "base_2_score", "current_2_score", "selection_3_score"]:
                mean_vals = pair_results.get(key + "_mean", [])
                maj_vals = pair_results.get(key + "_majority", [])

                mean_numbers = []
                mean_weights = []
                maj_numbers = []
                maj_weights = []

                for exp_i in idx_map.get(oi, []):
                    importance = eds[exp_i].get("importance")
                    if importance is None:
                        continue

                    if exp_i < len(mean_vals):
                        mv = mean_vals[exp_i]
                        if mv is not None:
                            mean_numbers.append(mv)
                            mean_weights.append(importance)

                    if exp_i < len(maj_vals):
                        mj = maj_vals[exp_i]
                        if mj is not None:
                            maj_numbers.append(mj)
                            maj_weights.append(importance)

                base_row[key.replace("_score", "_mean")] = weighted_average(mean_numbers, mean_weights)
                base_row[key.replace("_score", "_majority")] = weighted_average(maj_numbers, maj_weights)

            f.write(json.dumps(base_row) + "\n")
            out_rows.append(base_row)

    print(f"Wrote {n_rows} rows -> {out_path}")

    if args.push_to_hub:
        try:
            repo_id = args.hf_repo_template.format(shard_idx=args.idx)
            print(f"Pushing to HF Hub: {repo_id}")
            ds_out = Dataset.from_list(out_rows)
            ds_out.push_to_hub(repo_id)
            print(f"Pushed dataset to hub: {repo_id}")
        except Exception as e:
            print(f"Failed to push to Hugging Face Hub: {e}")

    print(f"Time taken: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
