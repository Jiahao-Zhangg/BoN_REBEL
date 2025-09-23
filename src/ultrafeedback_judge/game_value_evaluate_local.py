#!/usr/bin/env python3
import argparse
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from pydantic import BaseModel, Field
import gc
import re


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int = 5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data and models
    parser.add_argument("--dataset_repo", type=str, default="zjhhhh/sw_maxlen_8192_mean_maxlenprompt_1024_fixed_tokenized_logprob", help="Hugging Face dataset repo id (split=train)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model id/path to sample base responses")
    parser.add_argument("--check_points", type=str, nargs='+', default=["Qwen/Qwen2.5-3B-Instruct"], help="One or more model ids/paths to evaluate")
    parser.add_argument("--output_repo_prefix", type=str, required=True, help="Prefix for output repo; final repo is {prefix}_{model_name}")

    # Generation
    parser.add_argument("--n_response", type=int, default=1, help="Number of responses to sample per prompt for base/model")
    parser.add_argument("--maxlen", type=int, default=8192)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument(
        "--model_temperature", "--temperature", type=float, default=0.1,
        help="Sampling temperature for base/checkpoint generations"
    )
    parser.add_argument(
        "--model_top_p", "--top_p", type=float, default=0.9,
        help="Top-p nucleus sampling value for base/checkpoint generations"
    )
    parser.add_argument("--gpu_memory_utilization", "--gpu-memory-utilization", type=float, default=0.95, help="vLLM GPU memory utilization fraction [0-1]")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    # Judge (local)
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--n_judge_samples", type=int, default=5)
    parser.add_argument("--judge_max_tokens", type=int, default=256)
    parser.add_argument(
        "--judge_temperature", type=float, default=0.6,
        help="Sampling temperature for the judge model"
    )
    parser.add_argument(
        "--judge_top_p", type=float, default=0.95,
        help="Top-p nucleus sampling value for the judge model"
    )
    parser.add_argument(
        "--judge_top_k", "--top_k", type=int, default=20,
        help="Top-k value for the judge model (alias --top_k for backward compatibility)"
    )
    parser.add_argument("--switch_position", action="store_true", default=False, help="Collect preferences in both directions and reverse bias")

    # Debug
    parser.add_argument("--max_prompts", type=int, default=None, help="Optional cap on number of unique prompts for testing")

    return parser.parse_args()


def get_message(instruction: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": instruction}]


def sanitize_model_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__").replace(" ", "_")


def load_fix_criteria_template() -> str:
    template_path = Path(__file__).parent / "prompt_fix_criteria.txt"
    with open(template_path, 'r') as f:
        return f.read()


def get_numeric_mode(values: List[int], score_range: Tuple[int, int]) -> Optional[int]:
    if not values:
        return None
    min_score, max_score = score_range
    filtered = [int(v) for v in values if min_score <= int(v) <= max_score]
    if not filtered:
        return None
    from collections import Counter
    counts = Counter(filtered)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    modes.sort()
    return modes[len(modes) // 2]


class Preference5ScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=-1, le=4)


PREFERENCE_5SCORE_GUIDED_DECODING = GuidedDecodingParams(json=Preference5ScoreOutput.model_json_schema())


def is_valid_5score(value: Optional[int]) -> bool:
    try:
        if value is None:
            return False
        v = int(value)
        return -1 <= v <= 4
    except Exception:
        return False


def reverse_score_5score(score: int) -> int:
    # Keep -1 as -1. Otherwise reverse on 0..4 via 4 - s
    if score == -1:
        return -1
    return 4 - int(score)


def parse_verdict_safely(text: str) -> Optional[int]:
    """
    Try to parse a verdict integer from model output. Prefer JSON, but
    fall back to regex if JSON is malformed or truncated.
    Accepts values in range [-1, 4]. Returns None if not found/invalid.
    """
    if not text:
        return None
    # First try strict JSON
    try:
        obj = json.loads(text)
        # obj may be dict (preferred) or scalar
        if isinstance(obj, dict):
            value = obj.get("verdict", None)
        else:
            value = obj
        if is_valid_5score(value):
            return int(value)
    except Exception:
        pass

    # Fall back: try to find 'verdict' key-like or bare score in text
    # 1) key-based: "verdict": 3 or 'verdict': 3
    match = re.search(r"verdict\s*[:=]\s*([-]?\d)", text, flags=re.IGNORECASE)
    if match:
        try:
            cand = int(match.group(1))
            return cand if is_valid_5score(cand) else None
        except Exception:
            pass

    # 2) JSON-like minimal prefix, e.g., {"verdict": "3" ... truncated
    match = re.search(r"\b([-]?\d)\b", text)
    if match:
        try:
            cand = int(match.group(1))
            return cand if is_valid_5score(cand) else None
        except Exception:
            pass
    return None


def aggregate_numeric_samples(samples: List[Optional[int]]) -> Tuple[Optional[float], Optional[int]]:
    parsed = [int(s) for s in samples if is_valid_5score(s)]
    if not parsed:
        return None, None
    # Mean excluding -1 (confused/noise)
    valid_for_mean = [x for x in parsed if x != -1]
    mean_val: Optional[float] = (sum(valid_for_mean) / len(valid_for_mean)) if valid_for_mean else None
    # Majority over 0..4 only
    majority_val: Optional[int] = get_numeric_mode(parsed, (0, 4))
    return mean_val, majority_val


def generate_n_responses(model_id: str, prompts: List[str], world_size: int, maxlen: int, n_response: int, temperature: float, top_p: float, gpu_memory_utilization: float) -> List[List[str]]:
    print(f"Generating {n_response} responses per prompt with model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=world_size,
        max_model_len=maxlen,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    chat_prompts = [tokenizer.apply_chat_template([{ "role": "user", "content": p }], tokenize=False, add_generation_prompt=True) for p in tqdm(prompts)]

    all_pass_outputs: List[List[str]] = []  # shape [n_response][num_prompts]
    for i in range(n_response):
        set_seed(i * 50)
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=maxlen,
            seed=i * 50,
        )
        outputs = llm.generate(chat_prompts, params)
        texts = [o.outputs[0].text for o in outputs]
        all_pass_outputs.append(texts)

    # transpose -> [num_prompts][n_response]
    per_prompt: List[List[str]] = []
    for idx in range(len(prompts)):
        per_prompt.append([all_pass_outputs[j][idx] for j in range(n_response)])
    # Free generation model and tokenizer to avoid OOM
    try:
        del llm
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    # import pdb; pdb.set_trace()
    cleanup_memory()
    return per_prompt


def add_static_response_columns(dataset: Dataset, rows: List[dict], prompt_to_idx: Dict[str, int], base_resps: List[List[str]], model_resps: List[List[str]]):
    num_rows = len(rows)
    n_base = len(base_resps[0]) if base_resps else 0
    n_model = len(model_resps[0]) if model_resps else 0
    # Base responses
    for i in range(n_base):
        col = [base_resps[prompt_to_idx[rows[r]["prompt"]]][i] for r in range(num_rows)]
        dataset = dataset.add_column(f"base_response_{i}", col)
    # Model responses
    for j in range(n_model):
        col = [model_resps[prompt_to_idx[rows[r]["prompt"]]][j] for r in range(num_rows)]
        dataset = dataset.add_column(f"model_response_{j}", col)
    return dataset


def compute_simple_winrate_from_columns(ds: Dataset) -> Tuple[List[int], List[int], List[Optional[float]], Optional[float], List[Optional[float]], Optional[float]]:
    # Compute per-prompt winrate from existing judge_{a}_{b}_majority columns.
    # A win if majority > 2, loss if < 2, ignore == 2 or None.
    # Also compute per-prompt average of judge_{a}_{b}_mean scores and its overall average.
    # Build prompt order
    prompts: List[str] = [row["prompt"] for row in ds]
    prompt_to_idxs: Dict[str, List[int]] = {}
    for i, p in enumerate(prompts):
        prompt_to_idxs.setdefault(p, []).append(i)

    # infer n_base/n_model from columns
    base_cols = sorted([c for c in ds.column_names if c.startswith("base_response_")], key=lambda x: int(x.split("_")[-1]))
    model_cols = sorted([c for c in ds.column_names if c.startswith("model_response_")], key=lambda x: int(x.split("_")[-1]))
    n_base = len(base_cols)
    n_model = len(model_cols)

    unique_prompts = list(prompt_to_idxs.keys())
    wins_per_prompt: List[int] = [0] * len(unique_prompts)
    totals_per_prompt: List[int] = [0] * len(unique_prompts)
    per_prompt_avg_mean: List[Optional[float]] = [None] * len(unique_prompts)

    for p_idx, p in enumerate(unique_prompts):
        row_indices = prompt_to_idxs[p]
        # winrate via majority
        for a in range(n_model):
            for b in range(n_base):
                col = f"judge_{a}_{b}_majority"
                if col not in ds.column_names:
                    continue
                for i in row_indices:
                    val = ds[i][col]
                    if val is None:
                        continue
                    try:
                        v = int(val)
                    except Exception:
                        continue
                    if v > 2:
                        wins_per_prompt[p_idx] += 1
                        totals_per_prompt[p_idx] += 1
                    elif v < 2:
                        totals_per_prompt[p_idx] += 1
                    # v == 2 -> tie -> ignore

        # average of mean scores
        mean_values: List[float] = []
        for a in range(n_model):
            for b in range(n_base):
                col_mean = f"judge_{a}_{b}_mean"
                if col_mean not in ds.column_names:
                    continue
                for i in row_indices:
                    valm = ds[i][col_mean]
                    if valm is None:
                        continue
                    try:
                        mean_values.append(float(valm))
                    except Exception:
                        continue
        if mean_values:
            per_prompt_avg_mean[p_idx] = sum(mean_values) / len(mean_values)

    per_prompt_wr: List[Optional[float]] = []
    for w, t in zip(wins_per_prompt, totals_per_prompt):
        per_prompt_wr.append((w / t) if t > 0 else None)

    valid = [v for v in per_prompt_wr if v is not None]
    avg_wr: Optional[float] = (sum(valid) / len(valid)) if valid else None

    valid_means = [m for m in per_prompt_avg_mean if m is not None]
    avg_of_means: Optional[float] = (sum(valid_means) / len(valid_means)) if valid_means else None

    return wins_per_prompt, totals_per_prompt, per_prompt_wr, avg_wr, per_prompt_avg_mean, avg_of_means


def main():
    st = time.time()
    args = parse_arguments()

    # Load dataset and build unique prompts list
    try:
        raw = load_dataset(args.dataset_repo, split='test')
    except Exception:
        raw = load_dataset(args.dataset_repo, split='train')
    if args.end_idx != -1:
        raw = raw.select(range(args.start_idx, min(args.end_idx, len(raw))))
    # Prepare unique prompts
    unique_prompts: List[str] = list(dict.fromkeys([row["prompt"] for row in raw]))
    if args.max_prompts is not None:
        unique_prompts = unique_prompts[: args.max_prompts]
    expanded = Dataset.from_list([{"prompt": p} for p in unique_prompts])
    prompt_to_idx: Dict[str, int] = {p: i for i, p in enumerate(unique_prompts)}

    print(f"Num prompts: {len(unique_prompts)}")

    # Generate base responses once
    base_resps = generate_n_responses(
        args.base_model,
        unique_prompts,
        args.world_size,
        args.maxlen,
        args.n_response,
        args.model_temperature,
        args.model_top_p,
        args.gpu_memory_utilization,
    )

    # Load judge template (local)
    template = load_fix_criteria_template()

    # Iterate over candidate models
    expanded_rows = [row for row in expanded]
    for model_id in args.check_points:
        model_name = sanitize_model_name(model_id.split(":")[0])
        print(f"\n=== Evaluating {model_name} ===")

        # Generate candidate model responses (judge model NOT loaded yet)
        model_resps = generate_n_responses(
            model_id,
            unique_prompts,
            args.world_size,
            args.maxlen,
            args.n_response,
            args.model_temperature,
            args.model_top_p,
            args.gpu_memory_utilization,
        )

        # Instantiate judge per model to reduce peak memory
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
        judge_llm = LLM(
            model=args.judge_model,
            tensor_parallel_size=args.world_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )

        num_rows = len(expanded_rows)
        n_base = len(base_resps[0]) if base_resps else 0
        n_model = len(model_resps[0]) if model_resps else 0

        # Prepare storage for judge outputs
        mean_cols: Dict[Tuple[int, int], List[Optional[float]]] = {}
        maj_cols: Dict[Tuple[int, int], List[Optional[int]]] = {}
        for ai in range(n_model):
            for bj in range(n_base):
                mean_cols[(ai, bj)] = [None] * num_rows
                maj_cols[(ai, bj)] = [None] * num_rows

        # For each pair (ai, bj), build prompts for all rows and evaluate once
        for ai in range(n_model):
            for bj in range(n_base):
                # Build prompts for original A=model, B=base
                prompts: List[str] = []
                for row in expanded_rows:
                    p = row["prompt"]
                    pid = prompt_to_idx[p]
                    a_text = model_resps[pid][ai]
                    b_text = base_resps[pid][bj]
                    instruction = template.format(prompt=p, response_a=a_text, response_b=b_text)
                    chat = get_message(instruction)
                    prompts.append(judge_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

                # Sampling params for judge
                set_seed(0)
                sampling_params = SamplingParams(
                    temperature=args.judge_temperature,
                    top_p=args.judge_top_p,
                    top_k=args.judge_top_k,
                    n=args.n_judge_samples,
                    max_tokens=args.judge_max_tokens,
                    seed=0,
                    guided_decoding=PREFERENCE_5SCORE_GUIDED_DECODING,
                )

                print(f"Judging pair A=model[{ai}] vs B=base[{bj}] for {len(prompts)} rows ...")
                response = judge_llm.generate(prompts, sampling_params)

                # Collect all samples (optionally add switched direction)
                all_samples: List[List[int]] = []
                orig_texts = [[parse_verdict_safely(r.text) for r in result.outputs] for result in response]
                for orig_samples in orig_texts:
                    filtered = [int(s) for s in orig_samples if is_valid_5score(s)]
                    all_samples.append(filtered)

                if args.switch_position:
                    print(f"Judging pair (switched) A=base[{bj}] vs B=model[{ai}] ...")
                    prompts_switched: List[str] = []
                    for row in expanded_rows:
                        p = row["prompt"]
                        pid = prompt_to_idx[p]
                        a_text = base_resps[pid][bj]
                        b_text = model_resps[pid][ai]
                        instruction = template.format(prompt=p, response_a=a_text, response_b=b_text)
                        chat = get_message(instruction)
                        prompts_switched.append(judge_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

                    response_switched = judge_llm.generate(prompts_switched, sampling_params)
                    switched_texts = [[parse_verdict_safely(r.text) for r in result.outputs] for result in response_switched]
                    for idx, switched_samples in enumerate(switched_texts):
                        filtered = [int(s) for s in switched_samples if is_valid_5score(s)]
                        reversed_scores = [reverse_score_5score(s) for s in filtered]
                        all_samples[idx].extend(reversed_scores)
                    try:
                        del response_switched
                    except Exception:
                        pass
                    cleanup_memory()

                # Aggregate to mean and majority
                for row_idx, samples in enumerate(all_samples):
                    mean_val, maj_val = aggregate_numeric_samples(samples)
                    mean_cols[(ai, bj)][row_idx] = mean_val
                    maj_cols[(ai, bj)][row_idx] = maj_val
                try:
                    del response
                except Exception:
                    pass
                cleanup_memory()

        # Build dataset: one row per prompt with static responses and judged scores
        ds = expanded
        ds = add_static_response_columns(ds, expanded_rows, prompt_to_idx, base_resps, model_resps)
        for (ai, bj), col in maj_cols.items():
            ds = ds.add_column(f"judge_{ai}_{bj}_majority", col)
        for (ai, bj), col in mean_cols.items():
            ds = ds.add_column(f"judge_{ai}_{bj}_mean", col)

        # Compute simple winrate per prompt from columns and push
        wins, totals, per_prompt_wr, avg_wr, per_prompt_avg_mean, avg_of_means = compute_simple_winrate_from_columns(ds)
        ds = ds.add_column("win_count", wins)
        ds = ds.add_column("pair_count", totals)
        ds = ds.add_column("winrate", per_prompt_wr)
        ds = ds.add_column("avg_mean", per_prompt_avg_mean)
        print(f"Average winrate over prompts: {avg_wr}")
        print(f"Average of mean scores over prompts: {avg_of_means}")

        repo_id = f"{args.output_repo_prefix}_{model_name}"
        print(f"Pushing results to {repo_id} ...")
        ds.push_to_hub(repo_id)

        # Free judge to avoid OOM before next model
        try:
            del judge_llm
        except Exception:
            pass
        try:
            del judge_tokenizer
        except Exception:
            pass
        cleanup_memory()

        # Free model responses from CPU RAM before next checkpoint
        try:
            del model_resps
        except Exception:
            pass
        cleanup_memory()

    print(f"Done. Total time: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
    
