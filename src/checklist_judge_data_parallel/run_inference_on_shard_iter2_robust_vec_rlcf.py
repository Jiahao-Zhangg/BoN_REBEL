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
class Preference101ScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=-1, le=100)
 
 
PREFERENCE_BASELINE_GUIDED_DECODING = GuidedDecodingParams(
    json=Preference101ScoreOutput.model_json_schema()
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
    if judge_type == "baseline":
        try:
            score = int(response)
            return -1 <= score <= 100
        except Exception:
            return False
    return False


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run judge inference on a prepared shard and compute reward-style scores for selection, current, base, and adversary responses (per-check vectors plus scalar rewards).")
    parser.add_argument("--idx", type=int, required=True,
                        help="Shard index to load (matches prepare_shards naming)")
    parser.add_argument("--shard_dir", type=str, default="./local_shards",
                        help="Directory containing shard_* folders")
 
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--judge_type", type=str, default="baseline",
                        choices=["baseline"])

    parser.add_argument("--selection_pairs", type=int, default=4, help="number of selection responses")
    parser.add_argument("--base_pairs", type=int, default=2, help="number of base responses")
    parser.add_argument("--current_pairs", type=int, default=2, help="number of current responses")
    parser.add_argument("--adversary_pairs", type=int, default=2, help="number of adversary responses")

    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)

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
    adversary_cols = [f"adversary_response_{t+1}" for t in range(args.adversary_pairs)]
    needed = selection_cols + base_cols + current_cols + adversary_cols
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

    # Checkpointing setup: store per-pair results as we go so we can resume mid-stage
    ckpt_root = os.path.join(args.output_dir, f"shard_{args.idx:05d}_checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    def _pair_ckpt_path(label: str):
        safe = label.replace("/", "_")
        return os.path.join(ckpt_root, f"{safe}.json")

    def _atomic_write_json(path: str, obj: dict):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    # Load any existing checkpoints into memory so we skip recomputation
    def _load_existing_pair_checkpoints():
        loaded = 0
        try:
            for name in os.listdir(ckpt_root):
                if not name.endswith(".json"):
                    continue
                p = os.path.join(ckpt_root, name)
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue
                label = data.get("label")
                if not label:
                    continue
                n_exp = data.get("n_expanded")
                # Only load if sizes match this run's expanded dataset size
                if n_exp is not None and n_exp != n_expanded:
                    continue
                mean_vals = data.get("mean")
                maj_vals = data.get("majority")
                if mean_vals is None or maj_vals is None:
                    continue
                pair_results[label + "_mean"] = mean_vals
                pair_results[label + "_majority"] = maj_vals
                loaded += 1
        except FileNotFoundError:
            pass
        if loaded > 0:
            print(f"Resumed {loaded} pair checkpoints from {ckpt_root}")

    # Save a single pair's results so we can resume mid-stage if interrupted
    def _save_pair_checkpoint(label: str):
        payload = {
            "label": label,
            "judge_type": args.judge_type,
            "n_expanded": n_expanded,
            "mean": pair_results.get(label + "_mean", []),
            "majority": pair_results.get(label + "_majority", []),
        }
        _atomic_write_json(_pair_ckpt_path(label), payload)

    # Weighted average helper for aggregation
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
        result = sum(v * w for v, w in pairs) / total_weight
        return result / 100.0
 
    # Helper to run a batch of prompts (pointwise) and reduce to per-row score
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
        # Convert to list of per-row raw texts per sample
        texts = [[o.text for o in r.outputs] for r in responses]
        # Extract verdicts and filter
        verdict_lists = []
        for per_row in texts:
            vals = [extract_verdict(t) for t in per_row]
            vals = [v for v in vals if v is not None]
            vals = filter_valid_responses(vals, args.judge_type)
            verdict_lists.append(vals)
        # Compute per-row stats (baseline numeric with -1 as missing)
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
                else:
                    orig_mean.append(float(np.mean(ints_no_missing)))
                score_range = (0, 100)
                orig_majority.append(get_numeric_mode(vals, score_range))
        reduced_mean = orig_mean
        reduced_majority = orig_majority

        pair_results[label + "_mean"] = reduced_mean
        pair_results[label + "_majority"] = reduced_majority
        # Persist this label immediately for robust recovery
        _save_pair_checkpoint(label)

    # Prepare lists of response strings per expanded row
    sel_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in selection_cols}
    base_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in base_cols}
    cur_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in current_cols}
    adv_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in adversary_cols}

    # Load any existing per-pair checkpoints before computing
    _load_existing_pair_checkpoints()

    # Compute all pointwise scores once
    # selection
    for i in range(1, args.selection_pairs + 1):
        sel_col = selection_cols[i - 1]
        label = f"selection_{i}_score"
        if label + "_mean" not in pair_results:
            run_pair_and_reduce(sel_lists[sel_col], label)
    # base
    for j in range(1, args.base_pairs + 1):
        base_col = base_cols[j - 1]
        label = f"base_{j}_score"
        if label + "_mean" not in pair_results:
            run_pair_and_reduce(base_lists[base_col], label)
    # current
    for k in range(1, args.current_pairs + 1):
        cur_col = current_cols[k - 1]
        label = f"current_{k}_score"
        if label + "_mean" not in pair_results:
            run_pair_and_reduce(cur_lists[cur_col], label)
    # adversary
    for t in range(1, args.adversary_pairs + 1):
        adv_col = adversary_cols[t - 1]
        label = f"adversary_{t}_score"
        if label + "_mean" not in pair_results:
            run_pair_and_reduce(adv_lists[adv_col], label)

    # Prepare for writing outputs (aggregate per prompt using importance weights)
    os.makedirs(args.output_dir, exist_ok=True)
    # Build mapping from orig_index -> list of expanded indices in order once
    idx_map = {}
    for i in range(n_expanded):
        oi = eds[i]["orig_index"]
        idx_map.setdefault(oi, []).append(i)

    out_path = os.path.join(
        args.output_dir,
        f"shard_{args.idx:05d}_scores.jsonl",
    )
    out_rows = []
    tmp_out_path = out_path + ".tmp"
    with open(tmp_out_path, "w") as f:
        for oi in range(n_rows):
            row_out = {k: ds[oi][k] for k in ds.column_names}
            # Aggregate for each computed label
            all_labels = []
            all_labels += [f"selection_{i}_score" for i in range(1, args.selection_pairs + 1)]
            all_labels += [f"base_{j}_score" for j in range(1, args.base_pairs + 1)]
            all_labels += [f"current_{k}_score" for k in range(1, args.current_pairs + 1)]
            all_labels += [f"adversary_{t}_score" for t in range(1, args.adversary_pairs + 1)]
            for key in all_labels:
                mean_vals = pair_results.get(key + "_mean", [])
                maj_vals = pair_results.get(key + "_majority", [])
                # Per-check vectors for this prompt (one entry per checklist item)
                mean_vec = []
                maj_vec = []
                mean_numbers = []
                mean_weights = []
                for exp_i in idx_map.get(oi, []):
                    mv = None
                    mj = None
                    if exp_i < len(mean_vals):
                        mv = mean_vals[exp_i]
                    if exp_i < len(maj_vals):
                        mj = maj_vals[exp_i]
                    mean_vec.append(mv)
                    maj_vec.append(mj)
                    # Build inputs for scalar reward: importance-weighted mean score
                    if mv is not None:
                        importance = eds[exp_i].get("importance")
                        if importance is not None and importance > 0:
                            mean_numbers.append(mv)
                            mean_weights.append(importance)

                # Store vectors (per-check scores, padded with None where missing)
                row_out[key.replace("_score", "_mean")] = mean_vec
                row_out[key.replace("_score", "_majority")] = maj_vec
                # Store scalar reward (importance-weighted 1D score)
                row_out[key.replace("_score", "_reward")] = weighted_average(mean_numbers, mean_weights)
            f.write(json.dumps(row_out) + "\n")
            out_rows.append(row_out)
 
    os.replace(tmp_out_path, out_path)
    print(f"Wrote {n_rows} rows -> {out_path}")

    if args.push_to_hub:
        try:
            repo_id = args.hf_repo_template.format(shard_idx=args.idx)
            print(f"Pushing dataset to HF Hub -> {repo_id}")
            ds_out = Dataset.from_list(out_rows)
            ds_out.push_to_hub(repo_id)
            print(f"Pushed dataset to hub: {repo_id}")
        except Exception as e:
            print(f"Failed to push dataset to Hugging Face Hub: {e}")

    print(f"Time taken: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
