#!/usr/bin/env python3
import os
import argparse
import time
import random
from pathlib import Path
from typing import Dict, List


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
    # Data
    parser.add_argument("--dataset_repo", type=str, required=True, help="Hugging Face dataset repo id (split=train/test)")
    parser.add_argument("--base_output_repo", type=str, required=True, help="HF repo id to push base responses (e.g., org/name)")

    # Generation
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_response", type=int, default=2)
    parser.add_argument("--maxlen", type=int, default=8192)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--base_gpus", type=str, default=None, help="Comma-separated GPU IDs for base generation")

    return parser.parse_args()


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

    per_prompt: List[List[str]] = []
    for idx in range(len(prompts)):
        per_prompt.append([all_pass_outputs[j][idx] for j in range(n_response)])
    del llm
    del tokenizer
    torch.cuda.empty_cache()
    return per_prompt


def main():
    st = time.time()
    args = parse_arguments()

    if args.base_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([g.strip() for g in args.base_gpus.split(",") if g.strip() != ""])
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    from datasets import load_dataset, Dataset

    try:
        raw = load_dataset(args.dataset_repo, split='test')
    except Exception:
        raw = load_dataset(args.dataset_repo, split='train')
    if args.end_idx != -1:
        raw = raw.select(range(args.start_idx, min(args.end_idx, len(raw))))

    # Build unique prompts and remember requirements (first occurrence)
    unique_prompts: List[str] = []
    prompt_to_req: Dict[str, str] = {}
    for row in raw:
        p = row["prompt"]
        if p not in prompt_to_req:
            prompt_to_req[p] = row.get("requirements", "")
            unique_prompts.append(p)

    if args.max_prompts is not None:
        unique_prompts = unique_prompts[: args.max_prompts]

    base_resps = generate_n_responses(
        args.base_model,
        unique_prompts,
        args.world_size,
        args.maxlen,
        args.n_response,
        args.temperature,
        args.top_p,
    )

    # Construct dataset to push: one row per prompt
    rows = []
    for i, p in enumerate(unique_prompts):
        item = {
            "prompt": p,
            "requirements": prompt_to_req.get(p, ""),
        }
        for j, resp in enumerate(base_resps[i]):
            item[f"base_response_{j}"] = resp
        rows.append(item)
    ds = Dataset.from_list(rows)

    print(f"Pushing base responses to {args.base_output_repo} ...")
    ds.push_to_hub(args.base_output_repo)
    print(f"Done. Time: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()


