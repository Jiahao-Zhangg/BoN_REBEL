#!/usr/bin/env python3
import os
import argparse
import asyncio
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from openai import AsyncOpenAI
import math


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int = 5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Data and models
    parser.add_argument("--dataset_repo", type=str, required=True, help="Hugging Face dataset repo id (split=train)")
    parser.add_argument("--base_model", type=str, required=True, help="Model id/path to sample base responses")
    parser.add_argument("--check_points", type=str, nargs='+', required=True, help="One or more model ids/paths to evaluate")
    parser.add_argument("--output_repo_prefix", type=str, required=True, help="Prefix for output repo; final repo is {prefix}_{model_name}")

    # Generation
    parser.add_argument("--n_response", type=int, default=2, help="Number of responses to sample per prompt for base/model")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    # Judge
    parser.add_argument("--judge_model", type=str, default="qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--n_judge_samples", type=int, default=5)
    parser.add_argument("--max_concurrent", type=int, default=50)
    parser.add_argument("--openrouter_api_key", type=str, default=None, help="Optional; else use env OPENROUTER_API_KEY/OPENAI_API_KEY")

    # Scoring
    parser.add_argument("--beta", type=float, default=1.0)

    # Debug
    parser.add_argument("--max_prompts", type=int, default=None, help="Optional cap on number of unique prompts for testing")

    return parser.parse_args()


def get_message(instruction: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": instruction}]


def sanitize_model_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__").replace(" ", "_")


def split_requirements_to_checks(dataset: Dataset) -> Dataset:
    expanded = []
    for row in dataset:
        requirements_str: str = row.get("requirements", "")
        if not requirements_str:
            # Fallback: if no structured requirements, keep a single "check" equal to prompt
            expanded.append({**row, "check": row["prompt"], "importance": None})
            continue

        counter = 1
        parts: List[str] = []
        remaining = requirements_str
        while len(remaining) > 0:
            assert remaining.startswith(f"{counter})"), f"Malformed requirements at counter {counter}: {remaining[:80]}"
            needle = f"/100)\n{counter+1})"
            pos = remaining.find(needle)
            if pos > 0:
                curr = remaining[len(f"{counter})"):pos + len("/100)\n")]
            else:
                curr = remaining[len(f"{counter})"):]
            parts.append(curr)
            remaining = remaining[len(curr) + len(f"{counter})"):]
            counter += 1

        parts = list(map(lambda s: s.strip(), parts))
        for chunk in parts:
            new_row = dict(row)
            if "(importance:" in chunk:
                new_row["check"] = chunk.split("(importance:")[0].strip()
                try:
                    new_row["importance"] = int(chunk.split("(importance:")[1].split("/")[0].strip())
                except Exception:
                    new_row["importance"] = None
            else:
                new_row["check"] = chunk
                new_row["importance"] = None
            expanded.append(new_row)

    return Dataset.from_list(expanded)


def load_preference_5score_template() -> str:
    # Prefer template colocated in ultrafeedback_judge
    candidate = Path(__file__).parent.parent / "ultrafeedback_judge" / "prompt_preference_5score.txt"
    if candidate.exists():
        return candidate.read_text()
    # Fallback to any copy in repo
    alt = Path(__file__).parent.parent / "checklist_judge_data_parallel" / "prompt_preference_5score_explanation.txt"
    if alt.exists():
        return alt.read_text()
    raise FileNotFoundError("prompt_preference_5score*.txt not found")


def get_numeric_mode(values: List[int], score_range: Tuple[int, int]) -> int:
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


async def make_api_call_async(client: AsyncOpenAI, prompt: List[dict], judge_model: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            response = await client.chat.completions.create(model=judge_model, messages=prompt)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in API call: {e}")
            return None


async def batch_api_calls_async(client: AsyncOpenAI, prompts: List[List[dict]], judge_model: str, n_samples: int, max_concurrent: int) -> List[List[str]]:
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    for prompt in prompts:
        for _ in range(n_samples):
            tasks.append(make_api_call_async(client, prompt, judge_model, semaphore))
    print(f"Making {len(tasks)} concurrent API calls...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    organized: List[List[str]] = [[] for _ in range(len(prompts))]
    idx = 0
    for i in range(len(prompts)):
        for _ in range(n_samples):
            res = results[idx]
            organized[i].append(None if isinstance(res, Exception) else res)
            idx += 1
    return organized


def generate_n_responses(model_id: str, prompts: List[str], world_size: int, maxlen: int, n_response: int, temperature: float, top_p: float) -> List[List[str]]:
    print(f"Generating {n_response} responses per prompt with model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=world_size,
        max_model_len=maxlen,
        gpu_memory_utilization=0.85,
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
    return per_prompt


def build_pair_prompts(template: str, expanded_rows: List[dict], prompt_to_idx: Dict[str, int], base_resps: List[List[str]], model_resps: List[List[str]]) -> Tuple[List[List[dict]], List[Tuple[int, int, int]]]:
    prompts: List[List[dict]] = []
    pair_map: List[Tuple[int, int, int]] = []  # (row_idx, ai, bj)
    for row_idx, row in enumerate(expanded_rows):
        p = row["prompt"]
        check = row.get("check", p)
        pid = prompt_to_idx[p]
        for ai, a_text in enumerate(model_resps[pid]):
            for bj, b_text in enumerate(base_resps[pid]):
                instruction = template.format(prompt=p, response_a=a_text, response_b=b_text, check=check)
                prompts.append(get_message(instruction))
                pair_map.append((row_idx, ai, bj))
    return prompts, pair_map


def aggregate_pair_scores(raw_samples: List[str]) -> Tuple[float, int]:
    # Parse -> ints; valid range includes -1..4; filter for mean and mode accordingly
    parsed: List[int] = []
    for s in raw_samples:
        if s is None:
            continue
        ss = s.strip()
        try:
            val = int(ss)
        except Exception:
            # Try to extract trailing number
            tail = ss.split()[-1]
            try:
                val = int(tail)
            except Exception:
                continue
        if -1 <= val <= 4:
            parsed.append(val)

    if not parsed:
        return None, None

    # mean excluding -1
    valid_for_mean = [x for x in parsed if x != -1]
    mean_val = (sum(valid_for_mean) / len(valid_for_mean)) if valid_for_mean else None

    # majority over 0..4 only
    majority_val = get_numeric_mode(parsed, (0, 4))
    return mean_val, majority_val


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


def score(repo_id: str, beta: float) -> float:
    ds = load_dataset(repo_id, split='train')

    # Determine counts
    base_cols = sorted([c for c in ds.column_names if c.startswith("base_response_")], key=lambda x: int(x.split("_")[-1]))
    model_cols = sorted([c for c in ds.column_names if c.startswith("model_response_")], key=lambda x: int(x.split("_")[-1]))
    n_base = len(base_cols)
    n_model = len(model_cols)

    # Group rows by prompt
    grouped: Dict[str, List[dict]] = {}
    for row in ds:
        p = row["prompt"]
        grouped.setdefault(p, []).append(row)

    prompt_vals: List[float] = []
    for p, rows in grouped.items():
        # For each base response y' index b, compute P(x,y')
        exp_terms: List[float] = []
        for b in range(n_base):
            # For each check row: average over model responses a of judge_{a}_{b}
            per_check_avgs: List[float] = []
            for row in rows:
                vals: List[float] = []
                # Prefer mean; fallback to majority
                for a in range(n_model):
                    mean_key = f"judge_{a}_{b}_mean"
                    maj_key = f"judge_{a}_{b}_majority"
                    v = row.get(mean_key, None)
                    if v is None:
                        v = row.get(maj_key, None)
                    if v is not None:
                        try:
                            vals.append(float(v))
                        except Exception:
                            continue
                if vals:
                    per_check_avgs.append(sum(vals) / len(vals))
            if per_check_avgs:
                P_xy = min(per_check_avgs)
                exp_terms.append(math.exp(-P_xy / beta))
        if exp_terms:
            prompt_vals.append(sum(exp_terms) / len(exp_terms))

    if not prompt_vals:
        return float('nan')

    return -beta * (sum(prompt_vals) / len(prompt_vals))


async def main():
    st = time.time()
    args = parse_arguments()

    # Load dataset and expand to (prompt, check)
    raw = load_dataset(args.dataset_repo, split='train')
    if args.end_idx != -1:
        raw = raw.select(range(args.start_idx, min(args.end_idx, len(raw))))
    expanded = split_requirements_to_checks(raw)

    # Prepare unique prompts
    unique_prompts: List[str] = list(dict.fromkeys([row["prompt"] for row in expanded]))
    if args.max_prompts is not None:
        unique_prompts = unique_prompts[: args.max_prompts]
        expanded = Dataset.from_list([row for row in expanded if row["prompt"] in set(unique_prompts)])
    prompt_to_idx: Dict[str, int] = {p: i for i, p in enumerate(unique_prompts)}

    print(f"Num expanded rows (prompt, check): {len(expanded)}")
    print(f"Num unique prompts: {len(unique_prompts)}")

    # Generate base responses once
    base_resps = generate_n_responses(
        args.base_model,
        unique_prompts,
        args.world_size,
        args.maxlen,
        args.n_response,
        args.temperature,
        args.top_p,
    )

    # Load judge template
    template = load_preference_5score_template()

    # Prepare API client
    api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set --openrouter_api_key or environment OPENROUTER_API_KEY/OPENAI_API_KEY")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Iterate over candidate models
    expanded_rows = [row for row in expanded]
    for model_id in args.check_points:
        model_name = sanitize_model_name(model_id.split(":")[0])
        print(f"\n=== Evaluating {model_name} ===")

        # Generate candidate model responses
        model_resps = generate_n_responses(
            model_id,
            unique_prompts,
            args.world_size,
            args.maxlen,
            args.n_response,
            args.temperature,
            args.top_p,
        )

        # Build pair prompts (A = model, B = base); n_response*n_response per (prompt, check)
        prompts, pair_map = build_pair_prompts(template, expanded_rows, prompt_to_idx, base_resps, model_resps)

        # Score via async API
        raw_lists = await batch_api_calls_async(client, prompts, args.judge_model, args.n_judge_samples, args.max_concurrent)

        # Aggregate and collect per (row, ai, bj)
        num_rows = len(expanded_rows)
        mean_cols: Dict[Tuple[int, int], List[float]] = {}
        maj_cols: Dict[Tuple[int, int], List[int]] = {}
        for ai in range(args.n_response):
            for bj in range(args.n_response):
                mean_cols[(ai, bj)] = [None] * num_rows
                maj_cols[(ai, bj)] = [None] * num_rows

        for idx, samples in enumerate(raw_lists):
            row_idx, ai, bj = pair_map[idx]
            mean_val, maj_val = aggregate_pair_scores(samples)
            mean_cols[(ai, bj)][row_idx] = mean_val
            maj_cols[(ai, bj)][row_idx] = maj_val

        # Build dataset to push: start from expanded
        ds = expanded
        ds = add_static_response_columns(ds, expanded_rows, prompt_to_idx, base_resps, model_resps)
        for (ai, bj), col in maj_cols.items():
            ds = ds.add_column(f"judge_{ai}_{bj}_majority", col)
        for (ai, bj), col in mean_cols.items():
            ds = ds.add_column(f"judge_{ai}_{bj}_mean", col)

        # Push to hub
        repo_id = f"{args.output_repo_prefix}_{model_name}"
        print(f"Pushing results to {repo_id} ...")
        ds.push_to_hub(repo_id)

        # Score (placeholder)
        result = score(repo_id, args.beta)
        print(f"Score for {model_name}: {result}")

    print(f"Done. Total time: {time.time() - st:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())


