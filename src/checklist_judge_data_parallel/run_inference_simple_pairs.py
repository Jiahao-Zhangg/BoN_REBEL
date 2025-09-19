import os
import json
import time
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import load_from_disk, Dataset
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# ------------------ Structured outputs for guided decoding ------------------
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


def is_valid_response(response: str) -> bool:
    try:
        score = int(response)
        return -1 <= score <= 4
    except Exception:
        return False


def reverse_score(score: int):
    # For 5-score preference, -1 means invalid/missing and stays as -1
    if score == -1:
        return -1
    return 4 - int(score)


def extract_verdict(response_text: str):
    try:
        parsed = json.loads(response_text)
    except Exception:
        return None
    return parsed.get("verdict", None)


def get_message(instruction: str):
    return [{"role": "user", "content": instruction}]


# Majority util: mode with mean-of-modes tie break and -1 excluded

def get_numeric_mode(values: List[int]):
    values = [int(v) for v in values if int(v) != -1]
    if not values:
        return None
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    max_c = max(counts.values())
    modes = [k for k, c in counts.items() if c == max_c]
    if len(modes) == 1:
        return modes[0]
    return float(sum(modes) / len(modes))


# ------------------ Argument parsing ------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simplified judge inference per prompt per pair (single score, no checks)."
    )
    parser.add_argument("--idx", type=int, required=True,
                        help="Shard index to load (matches prepare_shards naming)")
    parser.add_argument("--shard_dir", type=str, default="./local_shards",
                        help="Directory containing shard_* folders")

    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    # We focus on 5-score preference for this simplified script
    parser.add_argument("--judge_type", type=str, default="preference_5score",
                        choices=["preference_5score"],
                        help="Simplified script supports only 5-score preference")

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
                        help="Directory to write JSONL results; one row per prompt with per-pair scores")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="If set, also pushes the output rows to Hugging Face Hub as a dataset")
    parser.add_argument("--hf_repo_template", type=str, default="zjhhhh/subsampling_simple_{shard_idx}",
                        help="Template for target HF repo id. {shard_idx} will be replaced with the shard index.")
    return parser.parse_args()


# ------------------ Main ------------------

def main():
    st = time.time()
    args = parse_args()

    shard_path = os.path.join(args.shard_dir, f"shard_{args.idx:05d}")
    if not os.path.isdir(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    ds: Dataset = load_from_disk(shard_path)
    n_rows = len(ds)
    print(f"Loaded shard {args.idx} from {shard_path} with {n_rows} prompts")

    # Prepare prompt template (no check)
    filename = "prompt_preference_5score_no_check.txt"
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
        guided_decoding=PREFERENCE_5SCORE_GUIDED_DECODING,
    )

    # Pre-allocate result holders for each pair
    pair_results = {}

    # Helper to run a batch of prompts and reduce to per-row mean and majority
    def run_pair_and_reduce(resp_a_list: List[str], resp_b_list: List[str], label: str):
        prompts = []
        for row_idx in range(n_rows):
            row = ds[row_idx]
            prompt = row["prompt"]
            resp_a = resp_a_list[row_idx]
            resp_b = resp_b_list[row_idx]
            filled = prompt_template.format(
                prompt=prompt,
                response_a=resp_a,
                response_b=resp_b,
            )
            messages = get_message(filled)
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        responses = llm.generate(prompts, sampling_params)
        texts = [[o.text for o in r.outputs] for r in responses]

        # Extract verdicts and filter
        verdict_lists = []
        for per_row in texts:
            vals = [extract_verdict(t) for t in per_row]
            vals = [v for v in vals if v is not None]
            vals = [v for v in vals if is_valid_response(v)]
            verdict_lists.append(vals)

        # Compute per-row mean and majority on original direction (ignores -1)
        orig_mean = []
        orig_majority = []
        for vals in verdict_lists:
            if len(vals) == 0:
                orig_mean.append(None)
                orig_majority.append(None)
            else:
                ints = [int(v) for v in vals]
                ints_no_missing = [x for x in ints if x != -1]
                if len(ints_no_missing) == 0:
                    orig_mean.append(None)
                    orig_majority.append(None)
                else:
                    orig_mean.append(float(np.mean(ints_no_missing)))
                    orig_majority.append(get_numeric_mode(ints_no_missing))

        reduced_mean = orig_mean
        reduced_majority = orig_majority

        if args.switch_position:
            # Also judge reversed A/B and combine
            prompts_sw = []
            for row_idx in range(n_rows):
                row = ds[row_idx]
                prompt = row["prompt"]
                resp_a = resp_b_list[row_idx]
                resp_b = resp_a_list[row_idx]
                filled = prompt_template.format(
                    prompt=prompt,
                    response_a=resp_a,
                    response_b=resp_b,
                )
                messages = get_message(filled)
                prompts_sw.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            responses_sw = llm.generate(prompts_sw, sampling_params)
            texts_sw = [[o.text for o in r.outputs] for r in responses_sw]

            sw_mean = []
            sw_majority = []
            for per_row in texts_sw:
                vals = [extract_verdict(t) for t in per_row]
                vals = [v for v in vals if v is not None]
                vals = [v for v in vals if is_valid_response(v)]
                if len(vals) == 0:
                    sw_mean.append(None)
                    sw_majority.append(None)
                else:
                    sw_ints = [int(v) for v in vals]
                    sw_ints = [reverse_score(v) for v in sw_ints]
                    sw_no_missing = [x for x in sw_ints if x != -1]
                    if len(sw_no_missing) == 0:
                        sw_mean.append(None)
                        sw_majority.append(None)
                    else:
                        sw_mean.append(float(np.mean(sw_no_missing)))
                        sw_majority.append(get_numeric_mode(sw_no_missing))

            # Average original and reversed statistics per row
            new_mean = []
            new_majority = []
            for om, sm, oj, sj in zip(reduced_mean, sw_mean, reduced_majority, sw_majority):
                # Mean
                if om is None and sm is None:
                    new_mean.append(None)
                elif om is None:
                    new_mean.append(sm)
                elif sm is None:
                    new_mean.append(om)
                else:
                    new_mean.append(0.5 * (om + sm))
                # Majority (numeric) already tie-broken via mean-of-modes; average two sides
                if oj is None and sj is None:
                    new_majority.append(None)
                elif oj is None:
                    new_majority.append(sj)
                elif sj is None:
                    new_majority.append(oj)
                else:
                    new_majority.append(0.5 * (float(oj) + float(sj)))

            reduced_mean = new_mean
            reduced_majority = new_majority

        pair_results[label + "_mean"] = reduced_mean
        pair_results[label + "_majority"] = reduced_majority

    # Prepare lists of response strings per row
    sel_lists = {col: [ds[i][col] for i in range(n_rows)] for col in selection_cols}
    base_lists = {col: [ds[i][col] for i in range(n_rows)] for col in base_cols}
    cur_lists = {col: [ds[i][col] for i in range(n_rows)] for col in current_cols}

    # Run selection vs base
    for i, sel_col in enumerate(selection_cols, start=1):
        for j, base_col in enumerate(base_cols, start=1):
            label = f"selection_{i}_base_{j}"
            run_pair_and_reduce(sel_lists[sel_col], base_lists[base_col], label)

    # Run current vs base
    for k, cur_col in enumerate(current_cols, start=1):
        for j, base_col in enumerate(base_cols, start=1):
            label = f"current_{k}_base_{j}"
            run_pair_and_reduce(cur_lists[cur_col], base_lists[base_col], label)

    # Assemble output rows: original columns + per-pair mean/majority (not lists)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard_{args.idx:05d}_scores_simple.jsonl")
    out_rows = []
    with open(out_path, "w") as f:
        for oi in range(n_rows):
            base_row = {k: ds[oi][k] for k in ds.column_names}
            for label, scores in pair_results.items():
                base_row[label] = scores[oi] if oi < len(scores) else None
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
