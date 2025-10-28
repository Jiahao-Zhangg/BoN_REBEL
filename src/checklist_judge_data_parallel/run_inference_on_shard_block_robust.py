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
        description="Run judge inference on a prepared shard and output 20 scores per prompt (mean and majority per pair).")
    parser.add_argument("--idx", type=int, required=True,
                        help="Shard index to load (matches prepare_shards naming)")
    parser.add_argument("--shard_dir", type=str, default="./local_shards",
                        help="Directory containing shard_* folders")

    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--judge_type", type=str, default="preference_5score",
                        choices=["preference_binary", "preference_ternary", "preference_score", "preference_5score"])

    parser.add_argument("--selection_pairs", type=int, default=4, help="number of selection responses")
    parser.add_argument("--base_pairs", type=int, default=8, help="number of base responses")
    parser.add_argument("--current_pairs", type=int, default=8, help="number of current responses")

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

    # Helper to run a batch of prompts and reduce to per-row score
    def run_pair_and_reduce(resp_a_list: List[str], resp_b_list: List[str], label: str):
        prompts = []
        for row_idx in range(n_expanded):
            row = eds[row_idx]
            prompt = row["prompt"]
            resp_a = resp_a_list[row_idx]
            resp_b = resp_b_list[row_idx]
            check_val = row.get("check", "")
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

        if args.judge_type in ["preference_score", "preference_5score", "reward"]:
            # Compute per-row stats on original direction only for now
            orig_mean = []
            orig_majority = []
            for vals in verdict_lists:
                if len(vals) == 0:
                    orig_mean.append(None)
                    orig_majority.append(None)
                else:
                    ints = [int(v) for v in vals]
                    # For 5-score preference, filter out -1 (missing/invalid) when computing stats
                    if args.judge_type == "preference_5score":
                        ints_no_missing = [x for x in ints if x != -1]
                        # Mean over non-missing only
                        if len(ints_no_missing) == 0:
                            orig_mean.append(None)
                        else:
                            orig_mean.append(float(np.mean(ints_no_missing)))
                        # Exclude -1 from majority by restricting range to 0..4
                        score_range = (0, 4)
                    else:
                        orig_mean.append(float(np.mean(ints)))
                        if args.judge_type == "preference_score":
                            score_range = (0, 10)
                        elif args.judge_type == "reward":
                            score_range = (0, 100)
                        else:
                            score_range = None
                    orig_majority.append(get_numeric_mode(vals, score_range))
            reduced_mean = orig_mean
            reduced_majority = orig_majority
        else:
            # Categorical preference: winner is used for both mean and majority to mirror reference behavior
            winners = [get_winner(vals) if len(vals) > 0 else None for vals in verdict_lists]
            reduced_majority = winners
            reduced_mean = winners

        if args.switch_position:
            # Also judge reversed A/B and combine
            prompts_sw = []
            for row_idx in range(n_expanded):
                row = eds[row_idx]
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

            if args.judge_type in ["preference_score", "preference_5score", "reward"]:
                # Compute stats on reversed direction separately, then average with original
                sw_mean = []
                sw_majority = []
                for sw_vals in verdict_lists_sw:
                    if len(sw_vals) == 0:
                        sw_mean.append(None)
                        sw_majority.append(None)
                    else:
                        sw_ints = [int(v) for v in sw_vals]
                        # Reverse switch scores for positional bias
                        sw_ints = [reverse_score(v, args.judge_type) for v in sw_ints]
                        if args.judge_type == "preference_5score":
                            # Filter out -1 when computing mean and majority
                            sw_no_missing = [x for x in sw_ints if x != -1]
                            if len(sw_no_missing) == 0:
                                sw_mean.append(None)
                            else:
                                sw_mean.append(float(np.mean(sw_no_missing)))
                            score_range = (0, 4)
                            sw_majority.append(get_numeric_mode(sw_no_missing, score_range))
                        else:
                            sw_mean.append(float(np.mean(sw_ints)))
                            if args.judge_type == "preference_score":
                                score_range = (0, 10)
                            elif args.judge_type == "reward":
                                score_range = (0, 100)
                            else:
                                score_range = None
                            sw_majority.append(get_numeric_mode(sw_ints, score_range))

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
                    # Majority (numeric) with mean tie-break already baked in per side; average the two sides
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
            else:
                # Categorical: compute winner on original and reversed separately, then combine:
                # if both present and equal -> keep; if only one present -> keep that; else -> Tie
                winners_orig = [get_winner(vals) if len(vals) > 0 else None for vals in verdict_lists]
                winners_sw = []
                for vals_b in verdict_lists_sw:
                    rb = [reverse_score(v, args.judge_type) for v in vals_b]
                    winners_sw.append(get_winner(rb) if len(rb) > 0 else None)

                winners_final = []
                for wo, ws in zip(winners_orig, winners_sw):
                    if wo is None and ws is None:
                        winners_final.append(None)
                    elif wo is None:
                        winners_final.append(ws)
                    elif ws is None:
                        winners_final.append(wo)
                    else:
                        winners_final.append(wo if wo == ws else "Tie")

                reduced_majority = winners_final
                reduced_mean = winners_final

        pair_results[label + "_mean"] = reduced_mean
        pair_results[label + "_majority"] = reduced_majority
        # Persist this pair immediately for robust recovery
        _save_pair_checkpoint(label)

    # Prepare lists of response strings per expanded row
    sel_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in selection_cols}
    base_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in base_cols}
    cur_lists = {col: [eds[i][col] for i in range(n_expanded)] for col in current_cols}

    # Build staged expansion targets so we can release (4,4,4) -> (4,6,6) -> (4,8,8)
    # Automatically deduplicate stages that collapse due to small pair counts.
    stage_candidates = [
        (args.selection_pairs, min(4, args.current_pairs), min(4, args.base_pairs)),
        (args.selection_pairs, min(6, args.current_pairs), min(6, args.base_pairs)),
        (args.selection_pairs, args.current_pairs, args.base_pairs),
    ]
    stages = []
    for s_sel, s_cur, s_base in stage_candidates:
        # Skip invalid or duplicate stages
        if s_sel == 0 or s_cur == 0 or s_base == 0:
            continue
        stage = (min(args.selection_pairs, s_sel),
                 min(args.current_pairs, s_cur),
                 min(args.base_pairs, s_base))
        if not stages or stage != stages[-1]:
            stages.append(stage)

    if not stages:
        raise ValueError("No valid stages inferred from provided pair counts.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build mapping from orig_index -> list of expanded indices in order once
    idx_map = {}
    for i in range(n_expanded):
        oi = eds[i]["orig_index"]
        idx_map.setdefault(oi, []).append(i)

    def resolve_repo_id(stage_suffix: str):
        template = args.hf_repo_template
        if "{stage" in template:
            return template.format(shard_idx=args.idx, stage=stage_suffix)
        base_repo = template.format(shard_idx=args.idx)
        return f"{base_repo}_{stage_suffix}"

    def write_stage_output(stage_idx: int, sel_count: int, cur_count: int, base_count: int):
        stage_suffix = f"sel{sel_count}_cur{cur_count}_base{base_count}"
        out_path = os.path.join(
            args.output_dir,
            f"shard_{args.idx:05d}_scores_{stage_suffix}.jsonl",
        )
        out_rows = []
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            for oi in range(n_rows):
                stage_row = {k: ds[oi][k] for k in ds.column_names}
                for i in range(1, sel_count + 1):
                    for j in range(1, base_count + 1):
                        key = f"selection_{i}_base_{j}_score"
                        mean_vals = pair_results.get(key + "_mean", [])
                        maj_vals = pair_results.get(key + "_majority", [])
                        mean_list = []
                        maj_list = []
                        for exp_i in idx_map.get(oi, []):
                            mean_list.append(mean_vals[exp_i] if exp_i < len(mean_vals) else None)
                            maj_list.append(maj_vals[exp_i] if exp_i < len(maj_vals) else None)
                        stage_row[key.replace("_score", "_mean")] = mean_list
                        stage_row[key.replace("_score", "_majority")] = maj_list
                for k in range(1, cur_count + 1):
                    for j in range(1, base_count + 1):
                        key = f"current_{k}_base_{j}_score"
                        mean_vals = pair_results.get(key + "_mean", [])
                        maj_vals = pair_results.get(key + "_majority", [])
                        mean_list = []
                        maj_list = []
                        for exp_i in idx_map.get(oi, []):
                            mean_list.append(mean_vals[exp_i] if exp_i < len(mean_vals) else None)
                            maj_list.append(maj_vals[exp_i] if exp_i < len(maj_vals) else None)
                        stage_row[key.replace("_score", "_mean")] = mean_list
                        stage_row[key.replace("_score", "_majority")] = maj_list

                f.write(json.dumps(stage_row) + "\n")
                out_rows.append(stage_row)

        # Atomic rename to avoid partial files
        os.replace(tmp_path, out_path)
        print(f"Stage {stage_idx}: wrote {n_rows} rows -> {out_path}")

        if args.push_to_hub:
            try:
                repo_id = resolve_repo_id(stage_suffix)
                print(f"Stage {stage_idx}: pushing to HF Hub -> {repo_id}")
                ds_out = Dataset.from_list(out_rows)
                ds_out.push_to_hub(repo_id)
                print(f"Stage {stage_idx}: pushed dataset to hub: {repo_id}")
            except Exception as e:
                print(f"Stage {stage_idx}: failed to push to Hugging Face Hub: {e}")

    # Sequentially compute the required pairs and emit staged outputs.
    # Load any existing per-pair checkpoints before computing
    _load_existing_pair_checkpoints()

    for stage_idx, (sel_count, cur_count, base_count) in enumerate(stages, start=1):
        # Ensure selection/base scores for this stage exist
        for i in range(1, sel_count + 1):
            sel_col = selection_cols[i - 1]
            for j in range(1, base_count + 1):
                label = f"selection_{i}_base_{j}_score"
                if label + "_mean" not in pair_results:
                    base_col = base_cols[j - 1]
                    run_pair_and_reduce(sel_lists[sel_col], base_lists[base_col], label)
        # Ensure current/base scores for this stage exist
        for k in range(1, cur_count + 1):
            cur_col = current_cols[k - 1]
            for j in range(1, base_count + 1):
                label = f"current_{k}_base_{j}_score"
                if label + "_mean" not in pair_results:
                    base_col = base_cols[j - 1]
                    run_pair_and_reduce(cur_lists[cur_col], base_lists[base_col], label)

        write_stage_output(stage_idx, sel_count, cur_count, base_count)

    print(f"Time taken: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
