from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import torch
import random
import numpy as np
import os
import logging
import sys

# # Configure logging to see VLLM's built-in progress information
# logging.basicConfig(
#     level=logging.INFO,  # global default
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
#     stream=sys.stdout,
# )

# # Make sure vLLM's loggers are on INFO (covers engine & workers)
# logging.getLogger("vllm").setLevel(logging.INFO)
# logging.getLogger("vllm.engine").setLevel(logging.INFO)
# logging.getLogger("vllm.worker").setLevel(logging.INFO)


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompts", type=str, default="viswavi/wildchecklists")
    parser.add_argument("--output_repo", type=str, required=True, help="output repo for the generated reponses")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--selection_pairs", type=int, default=1, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=1, help="number of pairs to use for gradient estimation")

    # Parallelism flags
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="number of GPUs per replica (TP)")
    parser.add_argument("--data_parallel_size", type=int, default=2, help="number of replicas (DP)")

    # Optional: control GPU mem use if desired
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    return parser.parse_args()


def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def infer_visible_gpu_count():
    # Respect CUDA_VISIBLE_DEVICES if set, else use torch.cuda.device_count()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return len([x for x in cvd.split(",") if x != ""])
    return torch.cuda.device_count()


def main():
    # init
    args = parse_arguments()

    # Basic sanity checks for DP × TP
    world_gpus = infer_visible_gpu_count()
    required_gpus = args.data_parallel_size * args.tensor_parallel_size
    if required_gpus < 1:
        raise ValueError("data_parallel_size × tensor_parallel_size must be >= 1")
    if world_gpus and required_gpus > world_gpus:
        raise RuntimeError(
            f"Requested DP({args.data_parallel_size}) × TP({args.tensor_parallel_size}) = {required_gpus} "
            f"GPUs, but only {world_gpus} visible. Adjust CUDA_VISIBLE_DEVICES or reduce DP/TP."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # VLLM with both tensor and data parallelism
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        # data_parallel_size=args.data_parallel_size,
        # gpu_memory_utilization=args.gpu_memory_utilization,
        # enforce_eager=True,  # sometimes helpful for multi-rank stability
    )

    # dataset
    dataset = load_dataset(args.prompts, split='train')
    if args.end_idx != -1:
        indices = random.sample(range(len(dataset)), 100)
        dataset = dataset.select(indices)
        # dataset = dataset.select(range(args.start_idx, args.end_idx))

    # prompts for llm
    prompts = [
        tokenizer.apply_chat_template(get_message(row["prompt"]), tokenize=False, add_generation_prompt=True)
        for row in tqdm(dataset)
    ]

    # start generate
    total_pairs = args.selection_pairs + args.gradient_pairs
    for p in range(total_pairs):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        # VLLM will internally distribute generation requests across DP replicas
        response = llm.generate(prompts, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        
        # Check for truncation
        truncated_count = sum(1 for r in response if r.outputs[0].finish_reason == "length")
        if truncated_count > 0:
            print(f"⚠️  Warning: {truncated_count}/{len(response)} responses were truncated (hit max_tokens limit)")
            print(f"   Consider increasing --maxlen (current: {args.maxlen})")
        
        dataset = dataset.add_column(f"response_{p}", output)

    # clean and push
    columns = ["prompt", "requirements"] + [f"response_{i}" for i in range(total_pairs)]
    dataset = dataset.select_columns(columns)
    
    # Try to push to hub first
    try:
        dataset.push_to_hub(args.output_repo)
        print(f"Successfully pushed to {args.output_repo}")
    except Exception as e:
        print(f"Failed to push to hub: {e}")
        # Only save locally if upload fails
        dataset.save_to_disk("./generated_dataset_backup")
        print("Dataset saved locally to ./generated_dataset_backup")


if __name__ == "__main__":
    main()
