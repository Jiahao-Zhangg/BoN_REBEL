import os
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def set_seed(seed=5775709):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_tag(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def get_message(instruction: str):
    return [{"role": "user", "content": instruction}]


def parse_models_arg(models: str, models_file: Optional[str]) -> List[str]:
    out: List[str] = []
    if models:
        out.extend([m.strip() for m in models.split(",") if m.strip()])
    if models_file:
        with open(models_file, "r") as f:
            for line in f:
                val = line.strip()
                if val:
                    out.append(val)
    if not out:
        raise ValueError("No models provided. Use --models or --models_file.")
    seen = set()
    ordered: List[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def parse_aliases(models: List[str], aliases: str) -> List[str]:
    if not aliases:
        return [safe_tag(m) for m in models]
    values = [a.strip() for a in aliases.split(",") if a.strip()]
    if len(values) != len(models):
        raise ValueError(f"--aliases length {len(values)} must match --models length {len(models)}")
    return values


def load_input_dataset(input_repo: str, split: str) -> Dataset:
    if os.path.isdir(input_repo):
        ds = load_from_disk(input_repo)
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            raise ValueError(f"Split '{split}' not found in local dataset: {list(ds.keys())}")
        return ds
    try:
        return load_dataset(input_repo, split=split)
    except Exception:
        ds_dict = load_dataset(input_repo)
        if split in ds_dict:
            return ds_dict[split]
        raise


def load_existing_responses(path: str, n_rows: int) -> List[Optional[Tuple[str, str]]]:
    responses: List[Optional[Tuple[str, str]]] = [None] * n_rows
    if not os.path.exists(path):
        return responses
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            idx = row.get("idx")
            if idx is None or not isinstance(idx, int):
                continue
            if idx < 0 or idx >= n_rows:
                continue
            r1 = row.get("response_1")
            r2 = row.get("response_2")
            if r1 is None or r2 is None:
                continue
            responses[idx] = (r1, r2)
    return responses


def write_response_rows(path: str, rows: List[dict]):
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def resolve_response_repo(template: str, alias: str) -> str:
    tag = safe_tag(alias)
    if "{model" in template or "{model_tag" in template:
        return template.format(model=alias, model_tag=tag)
    return f"{template}_{tag}"


def apply_postfix(repo_id: str, postfix: str) -> str:
    if not postfix:
        return repo_id
    tag = safe_tag(postfix).lstrip("_")
    if not tag:
        return repo_id
    return f"{repo_id}_{tag}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate two responses per prompt for multiple models (robust/resumable)."
    )
    parser.add_argument("--input_repo", type=str, required=True, help="HF repo or local dataset path")
    parser.add_argument("--test_split", type=str, default="test", help="Split to load from input repo")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index into the split")
    parser.add_argument("--num_prompts", type=int, default=-1,
                        help="Number of prompts to use from start_idx (<=0 means all remaining)")

    parser.add_argument("--models", type=str, default="", help="Comma-separated list of model ids")
    parser.add_argument("--models_file", type=str, default=None, help="Optional file with one model id per line")
    parser.add_argument("--aliases", type=str, default="",
                        help="Comma-separated list of aliases aligned with --models")

    parser.add_argument("--n_responses", type=int, default=2, help="Number of responses per prompt per model")
    parser.add_argument("--output_dir", type=str, default="./game_matrix_generation")
    parser.add_argument("--responses_repo_template", type=str, default="zjhhhh/{model_tag}")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hf_postfix", type=str, default="",
                        help="Optional suffix to append to the hub repo id (e.g., shard id)")

    parser.add_argument("--response_max_tokens", type=int, default=2048)
    parser.add_argument("--response_temperature", type=float, default=0.1)
    parser.add_argument("--response_top_p", type=float, default=0.9)
    parser.add_argument("--response_top_k", type=int, default=20)
    parser.add_argument("--response_world_size", type=int, default=1)
    parser.add_argument("--response_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--response_max_model_len", type=int, default=None)
    parser.add_argument("--response_seed", type=int, default=42)
    parser.add_argument("--response_trust_remote_code", action="store_true", default=False)
    return parser.parse_args()


def main():
    st = time.time()
    args = parse_args()

    if args.n_responses != 2:
        raise ValueError("This script expects --n_responses 2 for pairwise scoring.")

    models = parse_models_arg(args.models, args.models_file)
    aliases = parse_aliases(models, args.aliases)
    alias_by_model = {m: a for m, a in zip(models, aliases)}
    ds = load_input_dataset(args.input_repo, args.test_split)
    if args.start_idx < 0 or args.start_idx >= len(ds):
        raise ValueError(f"--start_idx {args.start_idx} is out of range for split size {len(ds)}")
    if args.num_prompts > 0:
        end_idx = min(args.start_idx + args.num_prompts, len(ds))
    else:
        end_idx = len(ds)
    ds = ds.select(range(args.start_idx, end_idx))
    if args.prompt_column not in ds.column_names:
        raise ValueError(
            f"Missing prompt column '{args.prompt_column}' in dataset. Columns: {ds.column_names}"
        )

    prompts = [row[args.prompt_column] for row in ds]
    n_rows = len(prompts)
    print(f"Loaded {n_rows} prompts from {args.input_repo} ({args.test_split})")

    responses_root = os.path.join(args.output_dir, "responses")
    os.makedirs(responses_root, exist_ok=True)

    responses_by_model: Dict[str, List[Optional[Tuple[str, str]]]] = {}

    for model_id in models:
        alias_tag = safe_tag(alias_by_model[model_id])
        out_path = os.path.join(responses_root, f"{alias_tag}_generattion.jsonl")
        existing = load_existing_responses(out_path, n_rows)
        missing = [i for i, v in enumerate(existing) if v is None]
        if not missing:
            print(f"[{model_id}] responses already complete -> {out_path}")
            responses_by_model[model_id] = existing
            continue

        print(f"[{model_id}] generating {len(missing)} missing rows -> {out_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.response_trust_remote_code)
        chat_prompts = [
            tokenizer.apply_chat_template(get_message(p), tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        llm_kwargs = {
            "model": model_id,
            "tensor_parallel_size": args.response_world_size,
            "gpu_memory_utilization": args.response_gpu_memory_utilization,
            "trust_remote_code": args.response_trust_remote_code,
        }
        if args.response_max_model_len is not None:
            llm_kwargs["max_model_len"] = args.response_max_model_len
        llm = LLM(**llm_kwargs)

        responses: List[List[Optional[str]]] = [[None, None] for _ in range(n_rows)]
        written = set()
        for idx, pair in enumerate(existing):
            if pair is not None:
                responses[idx][0] = pair[0]
                responses[idx][1] = pair[1]
                written.add(idx)

        for p in range(args.n_responses):
            missing_idx = [i for i in range(n_rows) if responses[i][p] is None]
            if not missing_idx:
                continue
            curr_seed = args.response_seed + p * 50
            set_seed(curr_seed)
            sampling_params = SamplingParams(
                temperature=args.response_temperature,
                top_p=args.response_top_p,
                top_k=args.response_top_k,
                max_tokens=args.response_max_tokens,
                seed=curr_seed,
            )
            batch_prompts = [chat_prompts[i] for i in missing_idx]
            outputs = llm.generate(batch_prompts, sampling_params)
            for idx, out in zip(missing_idx, outputs):
                responses[idx][p] = out.outputs[0].text

            rows = []
            for idx in missing_idx:
                if idx in written:
                    continue
                if responses[idx][0] is None or responses[idx][1] is None:
                    continue
                rows.append({
                    "idx": idx,
                    "prompt": prompts[idx],
                    "response_1": responses[idx][0],
                    "response_2": responses[idx][1],
                    "model": model_id,
                })
                written.add(idx)
            if rows:
                write_response_rows(out_path, rows)

        del llm
        del tokenizer
        torch.cuda.empty_cache()

        model_responses: List[Optional[Tuple[str, str]]] = []
        for idx in range(n_rows):
            r1 = responses[idx][0]
            r2 = responses[idx][1]
            if r1 is None or r2 is None:
                model_responses.append(None)
            else:
                model_responses.append((r1, r2))
        responses_by_model[model_id] = model_responses

    if args.push_to_hub:
        for model_id in models:
            responses = responses_by_model[model_id]
            if any(v is None for v in responses):
                print(f"[{model_id}] responses incomplete, skipping push")
                continue
            repo_id = resolve_response_repo(args.responses_repo_template, alias_by_model[model_id])
            repo_id = apply_postfix(repo_id, args.hf_postfix)
            rows = []
            for idx, (r1, r2) in enumerate(responses):
                rows.append({"idx": idx, "prompt": prompts[idx], "response_1": r1, "response_2": r2, "model": model_id})
            try:
                Dataset.from_list(rows).push_to_hub(repo_id)
                print(f"Pushed responses -> {repo_id}")
            except Exception as e:
                print(f"Failed to push responses for {model_id}: {e}")

    print(f"Time taken: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
