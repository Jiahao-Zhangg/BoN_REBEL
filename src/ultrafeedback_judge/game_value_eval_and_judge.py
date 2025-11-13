#!/usr/bin/env python3
import os
import argparse
import json
import time
import random
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from datasets import Dataset
from pydantic import BaseModel, Field
from vllm.sampling_params import GuidedDecodingParams

def set_seed(seed: int = 5775709):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Inputs
    parser.add_argument("--base_repo", type=str, required=True, help="HF repo id containing base responses (from step 1)")
    parser.add_argument("--output_repo_prefix", type=str, required=True, help="Prefix for output repo; final repo is {prefix}_{model_name}")
    parser.add_argument("--check_points", type=str, nargs='+', required=True, help="Model ids/paths to evaluate")

    # Generation
    parser.add_argument("--n_response", type=int, default=2)
    parser.add_argument("--maxlen", type=int, default=8192)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--model_temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Judge
    parser.add_argument("--judge_model", type=str, required=True)
    parser.add_argument("--n_judge_samples", type=int, default=5)
    parser.add_argument("--judge_temperature", type=float, default=0.6)
    parser.add_argument("--judge_top_p", type=float, default=0.95)
    parser.add_argument("--judge_top_k", type=int, default=20)
    parser.add_argument("--judge_max_tokens", type=int, default=256)
    parser.add_argument("--switch_position", action="store_true", default=False)
    parser.add_argument("--full_check", action="store_true", default=False)

    # Scoring
    parser.add_argument("--beta", type=float, default=1.0)

    # Misc
    parser.add_argument("--score_json_path", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=None, help="Start index (inclusive) into unique prompts")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (exclusive) into unique prompts")

    return parser.parse_args()


def sanitize_model_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__").replace(" ", "_")


def split_requirements_to_checks(dataset) -> "Dataset":
    from datasets import Dataset

    expanded = []
    for row in dataset:
        requirements_str: str = row.get("requirements", "")
        if not requirements_str:
            expanded.append({
                "prompt": row["prompt"],
                "requirements": requirements_str,
                "check": row["prompt"],
                "importance": None,
            })
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
            base_fields = {
                "prompt": row["prompt"],
                "requirements": requirements_str,
            }
            if "(importance:" in chunk:
                check_text = chunk.split("(importance:")[0].strip()
                try:
                    importance_val = int(chunk.split("(importance:")[1].split("/")[0].strip())
                except Exception:
                    importance_val = None
                expanded.append({**base_fields, "check": check_text, "importance": importance_val})
            else:
                expanded.append({**base_fields, "check": chunk, "importance": None})

    return Dataset.from_list(expanded)


def add_static_response_columns(dataset, rows: List[dict], prompt_to_idx: Dict[str, int], base_resps: List[List[str]], model_resps: List[List[str]]):
    num_rows = len(rows)
    n_base = len(base_resps[0]) if base_resps else 0
    n_model = len(model_resps[0]) if model_resps else 0
    for i in range(n_base):
        col = [base_resps[prompt_to_idx[rows[r]["prompt"]]][i] for r in range(num_rows)]
        dataset = dataset.add_column(f"base_response_{i}", col)
    for j in range(n_model):
        col = [model_resps[prompt_to_idx[rows[r]["prompt"]]][j] for r in range(num_rows)]
        dataset = dataset.add_column(f"model_response_{j}", col)
    return dataset


def load_preference_5score_template(full_check: bool = False) -> str:
    filename = "prompt_preference_5score_fullchecks.txt" if full_check else "prompt_preference_5score_explanation.txt"
    template_path = Path(__file__).parent / filename
    with open(template_path, 'r') as f:
        return f.read()


def get_numeric_mode(values: List[int], score_range: Tuple[int, int]) -> Optional[int]:
    if not values:
        return None
    mn, mx = score_range
    filtered = [int(v) for v in values if mn <= int(v) <= mx]
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
    if score == -1:
        return -1
    return 4 - int(score)


def parse_verdict_safely(text: str) -> Optional[int]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            value = obj.get("verdict")
        else:
            value = obj
        if is_valid_5score(value):
            return int(value)
    except Exception:
        pass

    import re as _re
    match = _re.search(r"verdict\s*[:=]\s*([-]?\d)", text, flags=_re.IGNORECASE)
    if match:
        try:
            cand = int(match.group(1))
            return cand if is_valid_5score(cand) else None
        except Exception:
            pass

    match = _re.search(r"\b([-]?\d)\b", text)
    if match:
        try:
            cand = int(match.group(1))
            return cand if is_valid_5score(cand) else None
        except Exception:
            pass
    return None


def aggregate_numeric_samples(samples: List[int]) -> Tuple[Optional[float], Optional[int]]:
    parsed = [int(s) for s in samples if is_valid_5score(s)]
    if not parsed:
        return None, None
    valid_for_mean = [x for x in parsed if x != -1]
    mean_val = (sum(valid_for_mean) / len(valid_for_mean)) if valid_for_mean else None
    majority_val = get_numeric_mode(parsed, (0, 4))
    return mean_val, majority_val


def _collect_normalized_scores(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(_collect_normalized_scores(item))
        return out
    try:
        return [float(value) / 4]
    except Exception:
        return []


def generate_n_responses(model_id: str, prompts: List[str], world_size: int, maxlen: int, n_response: int, temperature: float, top_p: float) -> List[List[str]]:
    print(f"Generating {n_response} responses per prompt with model: {model_id}")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        tensor_parallel_size=world_size,
        max_model_len=maxlen,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    chat_prompts = [tokenizer.apply_chat_template([{ "role": "user", "content": p }], tokenize=False, add_generation_prompt=True) for p in tqdm(prompts)]

    all_pass_outputs: List[List[str]] = []
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

    per_prompt: List[List[str]] = []
    for idx in range(len(prompts)):
        per_prompt.append([all_pass_outputs[j][idx] for j in range(n_response)])
    del llm
    del tokenizer
    torch.cuda.empty_cache()
    return per_prompt


def merge_dataset_results(dataset) -> "Dataset":
    from datasets import Dataset

    merged: Dict[str, dict] = {}
    for row in dataset:
        prompt = row["prompt"]
        if prompt not in merged:
            base_fields = {}
            for key, value in row.items():
                if key == "prompt" or key == "requirements" or key.startswith("base_response_") or key.startswith("model_response_"):
                    base_fields[key] = value
            for key in row.keys():
                if key.startswith("judge_") and (key.endswith("_majority") or key.endswith("_mean")):
                    base_fields[key] = []
            merged[prompt] = base_fields

        for key, value in row.items():
            if key.startswith("judge_") and (key.endswith("_majority") or key.endswith("_mean")):
                if key not in merged[prompt]:
                    merged[prompt][key] = []
                if value is not None:
                    merged[prompt][key].append(value)

    return Dataset.from_list(list(merged.values()))


def score_dataset(ds, beta: float) -> float:
    base_cols = sorted([c for c in ds.column_names if c.startswith("base_response_")], key=lambda x: int(x.split("_")[-1]))
    model_cols = sorted([c for c in ds.column_names if c.startswith("model_response_")], key=lambda x: int(x.split("_")[-1]))
    n_base = len(base_cols)
    n_model = len(model_cols)

    grouped: Dict[str, List[dict]] = {}
    for row in ds:
        p = row["prompt"]
        grouped.setdefault(p, []).append(row)

    prompt_vals: List[float] = []
    for p, rows in grouped.items():
        check_objectives: List[float] = []
        for row in rows:
            base_exp_terms: List[float] = []
            for b in range(n_base):
                vals: List[float] = []
                for a in range(n_model):
                    mean_key = f"judge_{a}_{b}_mean"
                    maj_key = f"judge_{a}_{b}_majority"
                    values = _collect_normalized_scores(row.get(mean_key, None))
                    if not values:
                        values = _collect_normalized_scores(row.get(maj_key, None))
                    if values:
                        vals.extend(values)
                if vals:
                    expected_score = sum(vals) / len(vals)
                    base_exp_terms.append(math.exp(-expected_score / beta))
            if not base_exp_terms:
                continue
            avg_exp = sum(base_exp_terms) / len(base_exp_terms)
            if avg_exp <= 0:
                continue
            check_objectives.append(-beta * math.log(avg_exp))
        if check_objectives:
            prompt_vals.append(min(check_objectives))

    if not prompt_vals:
        return float('nan')

    return sum(prompt_vals) / len(prompt_vals)


def main():
    st = time.time()
    args = parse_arguments()

    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    # Load base responses dataset (pushed in stage 1)
    try:
        base_ds = load_dataset(args.base_repo, split='test')
    except Exception:
        base_ds = load_dataset(args.base_repo, split='train')

    # Expand to rows for judging
    if args.full_check:
        # Do NOT split the requirements. Keep one row per unique prompt and
        # pass the full requirements into the scoring template as a single check.
        all_prompts = [base_ds[i]["prompt"] for i in range(len(base_ds))]
        unique_prompts: List[str] = list(dict.fromkeys(all_prompts))
        # Apply start/end slicing first
        if args.start_idx is not None or args.end_idx is not None:
            s = 0 if args.start_idx is None else max(0, int(args.start_idx))
            e = None if args.end_idx is None else int(args.end_idx)
            unique_prompts = unique_prompts[s:e]
        # Map prompt -> first index to retrieve requirements text
        base_prompt_to_first_idx: Dict[str, int] = {}
        for i in range(len(base_ds)):
            p = base_ds[i]["prompt"]
            if p not in base_prompt_to_first_idx:
                base_prompt_to_first_idx[p] = i
        expanded_rows = []
        for p in unique_prompts:
            bi = base_prompt_to_first_idx[p]
            requirements_text = base_ds[bi].get("requirements", "")
            expanded_rows.append({
                "prompt": p,
                "requirements": requirements_text,
                "check": requirements_text,
            })
        expanded = Dataset.from_list(expanded_rows)
    else:
        # Split the requirements into multiple checks; one row per check
        expanded = split_requirements_to_checks(base_ds)
        expanded_rows = [row for row in expanded]
        # Prepare unique prompts and mappings
        unique_prompts: List[str] = list(dict.fromkeys([row["prompt"] for row in expanded_rows]))
        # Apply start/end slicing first
        if args.start_idx is not None or args.end_idx is not None:
            s = 0 if args.start_idx is None else max(0, int(args.start_idx))
            e = None if args.end_idx is None else int(args.end_idx)
            unique_prompts = unique_prompts[s:e]
            expanded = Dataset.from_list([row for row in expanded_rows if row["prompt"] in set(unique_prompts)])
            expanded_rows = [row for row in expanded]
    prompt_to_idx: Dict[str, int] = {p: i for i, p in enumerate(unique_prompts)}

    # Extract base responses matrix [num_prompts][n_base]
    n_base = len([c for c in base_ds.column_names if c.startswith("base_response_")])
    base_prompt_to_idx: Dict[str, int] = {}
    for i in range(len(base_ds)):
        p = base_ds[i]["prompt"]
        if p not in base_prompt_to_idx:
            base_prompt_to_idx[p] = i
    base_resps: List[List[str]] = []
    for p in unique_prompts:
        bi = base_prompt_to_idx[p]
        base_resps.append([base_ds[bi][f"base_response_{j}"] for j in range(n_base)])

    template = load_preference_5score_template(args.full_check)

    scores_accumulator: List[float] = []

    for model_id in args.check_points:
        model_name = sanitize_model_name(model_id.split(":")[0])

        # Generate model responses per unique prompt
        model_resps = generate_n_responses(
            model_id,
            unique_prompts,
            args.world_size,
            args.maxlen,
            args.n_response,
            args.model_temperature if args.model_temperature is not None else args.temperature,
            args.top_p,
        )

        # Judge
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
        judge_llm = LLM(
            model=args.judge_model,
            tensor_parallel_size=args.world_size,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
            top_k=args.judge_top_k,
            n=args.n_judge_samples,
            max_tokens=args.judge_max_tokens,
            guided_decoding=PREFERENCE_5SCORE_GUIDED_DECODING,
        )

        num_rows = len(expanded_rows)
        mean_cols: Dict[Tuple[int, int], List[Optional[float]]] = {}
        maj_cols: Dict[Tuple[int, int], List[Optional[int]]] = {}
        for ai in range(len(model_resps[0])):
            for bj in range(n_base):
                mean_cols[(ai, bj)] = [None] * num_rows
                maj_cols[(ai, bj)] = [None] * num_rows

        for ai in range(len(model_resps[0])):
            for bj in range(n_base):
                rendered_prompts = []
                for row in expanded_rows:
                    p = row["prompt"]
                    check = row.get("check", p)
                    pid = prompt_to_idx[p]
                    a_text = model_resps[pid][ai]
                    b_text = base_resps[pid][bj]
                    instruction = template.format(prompt=p, response_a=a_text, response_b=b_text, check=check)
                    chat = [{"role": "user", "content": instruction}]
                    rendered_prompts.append(
                        judge_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    )

                response = judge_llm.generate(rendered_prompts, sampling_params)
                all_samples: List[List[int]] = []
                for result in response:
                    verdicts = [parse_verdict_safely(candidate.text) for candidate in result.outputs]
                    filtered = [int(v) for v in verdicts if is_valid_5score(v)]
                    all_samples.append(filtered)

                if args.switch_position:
                    rendered_switched = []
                    for row in expanded_rows:
                        p = row["prompt"]
                        check = row.get("check", p)
                        pid = prompt_to_idx[p]
                        a_text = base_resps[pid][bj]
                        b_text = model_resps[pid][ai]
                        instruction = template.format(prompt=p, response_a=a_text, response_b=b_text, check=check)
                        chat = [{"role": "user", "content": instruction}]
                        rendered_switched.append(
                            judge_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                        )

                    response_switched = judge_llm.generate(rendered_switched, sampling_params)
                    for idx_row, result_switched in enumerate(response_switched):
                        verdicts = [parse_verdict_safely(candidate.text) for candidate in result_switched.outputs]
                        filtered = [int(v) for v in verdicts if is_valid_5score(v)]
                        reversed_scores = [reverse_score_5score(v) for v in filtered]
                        if idx_row < len(all_samples):
                            all_samples[idx_row].extend(reversed_scores)
                        else:
                            all_samples.append(reversed_scores)
                    del response_switched
                    torch.cuda.empty_cache()

                for row_idx, scores in enumerate(all_samples):
                    mean_val, maj_val = aggregate_numeric_samples(scores)
                    mean_cols[(ai, bj)][row_idx] = mean_val
                    maj_cols[(ai, bj)][row_idx] = maj_val

                del response
                torch.cuda.empty_cache()

        try:
            del judge_llm
        except Exception:
            pass
        try:
            del judge_tokenizer
        except Exception:
            pass
        torch.cuda.empty_cache()

        # Build expanded dataset and add static/judge columns
        ds = expanded
        ds = add_static_response_columns(ds, expanded_rows, prompt_to_idx, base_resps, model_resps)
        for (ai, bj), col in maj_cols.items():
            ds = ds.add_column(f"judge_{ai}_{bj}_majority", col)
        for (ai, bj), col in mean_cols.items():
            ds = ds.add_column(f"judge_{ai}_{bj}_mean", col)

        result = score_dataset(ds, args.beta)

        # Push merged-by-prompt dataset
        repo_id = f"{args.output_repo_prefix}_{model_name}"
        ds_merged = merge_dataset_results(ds)
        print(f"Pushing merged results to {repo_id} ...")
        ds_merged.push_to_hub(repo_id)

        print(f"Score for {model_name}: {result}")
        scores_accumulator.append(result)
    if args.score_json_path:
        summary_path = Path(args.score_json_path) if args.score_json_path else Path(f"{sanitize_model_name(args.output_repo_prefix)}_scores.json")
        summary_data = {"checkpoints": args.check_points, "scores": scores_accumulator}
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved score summary to {summary_path}")


if __name__ == "__main__":
    main()


