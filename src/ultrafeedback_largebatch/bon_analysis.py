"""Evaluate reward as best-of-n increases."""

import os
import argparse
import random
from dataclasses import dataclass, field
from typing import List
import gc
import time
from contextlib import contextmanager

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Dict, List
import tyro
from datasets import load_dataset


@dataclass
class Args:
    models: List[str] = field(default_factory=lambda: [
        "meta-llama/Llama-3.2-3B-Instruct",
        "MisDrifter/3b",
        "MisDrifter/3b_bon",
    ])
    """the model to evaluate"""
    input_dataset: str = "MisDrifter/eval_3B_whole_data_armo_bon_REBEL_tokenized_logprob"
    """the input dataset"""
    split: str = "test"
    """the split of the dataset"""
    end_idx: int = -1
    """the end index of the dataset"""
    n: int = 5
    """best-of-n"""
    reward_models: List[str] = field(default_factory=lambda: [
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        # "OpenAssistant/reward-model-deberta-v3-large-v2",  # These don't work yet
        # "sfairXC/FsfairX-LLaMA3-RM-v0.1",                  # These don't work yet
    ])
    """reward models to evaluate responses on"""
    maxlen: int = 1024
    """the maximum length of the responses"""
    world_size: int = 2
    """the number of GPUs to use"""
    rm_batch_size: int = 2
    """the batch size for the reward model"""


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def cuda_memory_cleanup():
    """Context manager to ensure proper cleanup of CUDA memory and resources."""
    try:
        yield
    finally:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Small delay to ensure cleanup
        time.sleep(1)


def generate_responses(model_path, dataset, maxlen, world_size, n):
    """Generate n responses per prompt using the given model"""
    print(f"Generating {n} responses per prompt with model: {model_path}")

    def get_message(instruction):
        message = [
            {"role": "user", "content": instruction},
        ]
        return message
    
    with cuda_memory_cleanup():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create LLM with explicit configuration to handle CUDA context
            llm = LLM(
                model=model_path,
                tensor_parallel_size=world_size,
                # max_model_len=maxlen,
                gpu_memory_utilization=0.7,  # Reduced to leave more room for cleanup
                # disable_custom_all_reduce=True,  # Help with multi-GPU issues
                # enforce_eager=False,
                # trust_remote_code=True,
                # enable_prefix_caching=False,  # Disable to reduce memory usage
                # use_v2_block_manager=False,   # Use stable block manager
            )
            
            # Prepare prompts
            prompts = [tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=False, add_generation_prompt=True) for row in tqdm(dataset)]

            for p in range(n):
                set_seed(p * 50)
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=maxlen,
                    seed=p * 50,
                )
                response = llm.generate(prompts, sampling_params)
                output = list(map(lambda x: x.outputs[0].text, response))
                dataset = dataset.add_column(f"response_{p}_test", output)
            
            # Explicitly delete the LLM instance to free resources
            del llm
            del tokenizer
            
            return dataset
            
        except Exception as e:
            print(f"Error during generation: {e}")
            # Clean up on error
            if 'llm' in locals():
                del llm
            if 'tokenizer' in locals():
                del tokenizer
            raise


class RewardModelPipeline:
    def __init__(self, model_id, device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def tokenized_len(self, message: Dict[str, str]) -> int:
        return len(self.tokenizer.apply_chat_template(
            message,
            truncation=self.truncation,
            max_length=self.max_length,
        ))

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            scores = output.score.float().detach().cpu()
        return scores


def score_responses(reward_model_name, dataset, n, batch_size=16):
    def get_message(instruction, response):
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]

    with cuda_memory_cleanup():
        rm = RewardModelPipeline(reward_model_name, trust_remote_code=True)

        rewards = {}
        for i in range(n):
            # Create messages
            messages = [(msg_i, get_message(row['prompt'], row[f'response_{i}_test'])) for msg_i, row in enumerate(dataset)]
            # Sort messages by tokenized length, but store original order
            messages.sort(key=lambda x: rm.tokenized_len(x[1]), reverse=True)
            # Generate rewards in batches
            _rewards, _indices = [], []
            for batch_idx in tqdm(range(0, len(messages), batch_size)):
                batch = messages[batch_idx:batch_idx+batch_size]
                batch_messages = [x[1] for x in batch]
                batch_indices = [x[0] for x in batch]
                batch_rewards = rm(batch_messages)
                _rewards.append(batch_rewards)
                _indices.extend(batch_indices)
            _rewards = torch.cat(_rewards)
            _indices = torch.tensor(_indices)
            # Revert rewards back to original order
            sorted_rewards = torch.zeros_like(_rewards)
            sorted_rewards[_indices] = _rewards
            rewards[f"response_{i}_reward_{reward_model_name}"] = sorted_rewards.tolist()

        for k, v in rewards.items():
            dataset = dataset.add_column(k, v)
        
        del rm

        return dataset


def get_bon_data(reward_model_name, dataset, n):
    rewards = np.zeros((len(dataset), n))
    for i in range(n):
        rewards[:, i] = dataset[f"response_{i}_reward_{reward_model_name}"]

    bon_rewards = np.zeros((len(dataset), n))
    for i in range(n):
        bon_rewards[:, i] = np.max(rewards[:, :i+1], axis=1)

    return bon_rewards.mean(axis=0)


def main(args: Args):
    results = {}
    for model in args.models:
        # Setup
        print(f"Loading dataset {args.input_dataset}...")
        dataset = load_dataset(args.input_dataset, split=args.split)
        if args.end_idx > 0:
            dataset = dataset.select(range(args.end_idx))

        # Generate n responses per prompt
        print(f"Generating {args.n} responses per prompt with model: {model}")
        dataset = generate_responses(model, dataset, args.maxlen, args.world_size, args.n)

        # Score each response according to each reward model
        reward_results = {}
        for reward_model_name in args.reward_models:
            print(f"Scoring responses with reward model: {reward_model_name}")
            dataset = score_responses(reward_model_name, dataset, args.n, args.rm_batch_size)
            reward_results[reward_model_name] = get_bon_data(reward_model_name, dataset, args.n)
        
        results[model] = reward_results

    # Plot the reward as n increases
    for reward_model_name in args.reward_models:
        print(f"Plotting rewards with reward model: {reward_model_name}")

        for model in args.models:
            plt.plot(range(1, args.n+1), results[model][reward_model_name], label=model)
        
        plt.xlabel("Best-of-N")
        plt.ylabel("Average Reward")
        plt.legend()

        filename = f"bon_rewards_{reward_model_name.replace('/', '_')}"
        if os.path.exists(filename + ".png"):
            filename += f"_{time.strftime('%Y%m%d_%H%M%S')}"
        plt.savefig(filename + ".png")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
