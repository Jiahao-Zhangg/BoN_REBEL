#!/usr/bin/env python3
"""
Enhanced evaluation script that supports multiple models with pairwise comparisons
and best-of-n sampling without intermediate HuggingFace uploads/downloads
"""

import argparse
import time
import torch
import random
import numpy as np
import gc
import os
import multiprocessing
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from typing import Dict, List
from itertools import combinations
from contextlib import contextmanager


# Set environment variables to help with CUDA context issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True, 
                       help="List of model paths to compare (e.g., --models model1 model2 model3)")
    parser.add_argument("--model_names", type=str, nargs='+', default=None,
                       help="Optional custom names for models (e.g., --model_names base my_model other_model)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset for evaluation")
    parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--n", type=int, default=1, help="Number of samples per prompt (best-of-n)")
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use for testing (default: use all samples)")
    return parser.parse_args()


def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def get_message_with_response(instruction, response):
    return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]


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


def cleanup_vllm_resources():
    """Cleanup VLLM and CUDA resources between model iterations."""
    print("Cleaning up VLLM resources...")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Small delay for cleanup
    time.sleep(5)


class ArmoRMPipeline:
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

    def __call__(self, messages: List[Dict[str, str]]) -> float:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return score


def generate_responses(model_path, dataset, maxlen, world_size, n):
    """Generate n responses per prompt using the given model"""
    print(f"Generating {n} responses per prompt with model: {model_path}")
    
    with cuda_memory_cleanup():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create LLM with explicit configuration to handle CUDA context
            llm = LLM(
                model=model_path,
                tensor_parallel_size=world_size,
                max_model_len=maxlen,
                gpu_memory_utilization=0.85,  # Reduced to leave more room for cleanup
                disable_custom_all_reduce=True,  # Help with multi-GPU issues
                enforce_eager=False,
                trust_remote_code=True,
                enable_prefix_caching=False,  # Disable to reduce memory usage
                use_v2_block_manager=False,   # Use stable block manager
            )
            
            # Prepare prompts
            prompts = [tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=False, add_generation_prompt=True) for row in tqdm(dataset)]
            
            # Generate n responses for each prompt
            all_responses = []
            for i in range(n):
                set_seed(i * 50)
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=maxlen,
                    seed=i * 50,
                )
                response = llm.generate(prompts, sampling_params)
                output = list(map(lambda x: x.outputs[0].text, response))
                all_responses.append(output)
            
            # Transpose to get [prompt_idx][response_idx] structure
            responses_per_prompt = []
            for prompt_idx in range(len(prompts)):
                prompt_responses = [all_responses[response_idx][prompt_idx] for response_idx in range(n)]
                responses_per_prompt.append(prompt_responses)
            
            # Explicitly delete the LLM instance to free resources
            del llm
            del tokenizer
            
            return responses_per_prompt
            
        except Exception as e:
            print(f"Error during generation: {e}")
            # Clean up on error
            if 'llm' in locals():
                del llm
            if 'tokenizer' in locals():
                del tokenizer
            raise


def get_best_response_per_prompt(dataset, responses_per_prompt, reward_model_name, n):
    """Get the best response per prompt based on reward scores"""
    print(f"Selecting best response from {n} candidates per prompt...")
    
    rm = ArmoRMPipeline(reward_model_name, trust_remote_code=True)
    
    best_responses = []
    best_rewards = []
    
    for idx, row in enumerate(tqdm(dataset)):
        prompt_responses = responses_per_prompt[idx]
        
        # Get rewards for all n responses for this prompt
        rewards = []
        for response in prompt_responses:
            reward = rm(get_message_with_response(row['prompt'], response))
            rewards.append(reward)
        
        # Select the best response and its reward
        best_idx = np.argmax(rewards)
        best_responses.append(prompt_responses[best_idx])
        best_rewards.append(rewards[best_idx])
    
    return best_responses, best_rewards


def calculate_pairwise_stats(model_rewards, model_names):
    """Calculate pairwise win rates between all models"""
    print("Calculating pairwise win rates...")
    
    results = {}
    
    # Get all pairs of models
    model_pairs = list(combinations(range(len(model_names)), 2))
    
    for i, j in model_pairs:
        model_i_name = model_names[i]
        model_j_name = model_names[j]
        
        rewards_i = np.array(model_rewards[i])
        rewards_j = np.array(model_rewards[j])
        
        # Calculate wins for model i vs model j
        wins_i = np.sum(rewards_i > rewards_j)
        wins_j = np.sum(rewards_j > rewards_i)
        ties = np.sum(rewards_i == rewards_j)
        total = len(rewards_i)
        
        win_rate_i = wins_i / total
        win_rate_j = wins_j / total
        tie_rate = ties / total
        
        pair_name = f"{model_i_name} vs {model_j_name}"
        results[pair_name] = {
            f"{model_i_name}_wins": wins_i,
            f"{model_j_name}_wins": wins_j,
            "ties": ties,
            "total": total,
            f"{model_i_name}_win_rate": win_rate_i,
            f"{model_j_name}_win_rate": win_rate_j,
            "tie_rate": tie_rate
        }
        
        print(f"\n{pair_name}:")
        print(f"  {model_i_name}: {wins_i}/{total} wins ({win_rate_i:.4f})")
        print(f"  {model_j_name}: {wins_j}/{total} wins ({win_rate_j:.4f})")
        print(f"  Ties: {ties}/{total} ({tie_rate:.4f})")
    
    return results


def calculate_overall_stats(model_rewards, model_names):
    """Calculate overall statistics for all models"""
    print("\nOverall Model Statistics:")
    print("=" * 50)
    
    overall_stats = {}
    
    for i, model_name in enumerate(model_names):
        rewards = np.array(model_rewards[i])
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        overall_stats[model_name] = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "total_samples": len(rewards)
        }
        
        print(f"{model_name}:")
        print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"  Total samples: {len(rewards)}")
    
    return overall_stats


def calculate_overall_win_rates(model_rewards, model_names):
    """Calculate overall win rate for each model by averaging pairwise win rates"""
    print("\nOverall Win Rates (Average of Pairwise Win Rates):")
    print("=" * 60)
    
    overall_win_rates = {}
    
    for i, model_name in enumerate(model_names):
        win_rates = []
        total_wins = 0
        total_games = 0
        
        # Calculate win rate against each other model
        for j, opponent_name in enumerate(model_names):
            if i != j:  # Don't compare model with itself
                rewards_i = np.array(model_rewards[i])
                rewards_j = np.array(model_rewards[j])
                
                wins = np.sum(rewards_i > rewards_j)
                total = len(rewards_i)
                win_rate = wins / total
                
                win_rates.append(win_rate)
                total_wins += wins
                total_games += total
        
        # Calculate average win rate and overall win rate
        avg_win_rate = np.mean(win_rates) if win_rates else 0.0
        overall_win_rate = total_wins / total_games if total_games > 0 else 0.0
        
        overall_win_rates[model_name] = {
            "avg_pairwise_win_rate": avg_win_rate,
            "overall_win_rate": overall_win_rate,
            "total_wins": total_wins,
            "total_games": total_games,
        }
    
    # Sort models by average win rate for ranking
    sorted_models = sorted(overall_win_rates.items(), 
                          key=lambda x: x[1]["avg_pairwise_win_rate"], 
                          reverse=True)
    
    print(f"Model Ranking by Average Pairwise Win Rate:")
    print("-" * 50)
    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}: {stats['avg_pairwise_win_rate']:.4f}")
    
    return overall_win_rates


def main():
    # Set multiprocessing start method to 'spawn' to avoid CUDA context issues
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass
    st = time.time()
    args = parse_arguments()
    
    # Validate arguments
    if len(args.models) < 2:
        raise ValueError("At least 2 models are required for comparison")
    
    # Set model names
    if args.model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(args.models))]
    else:
        if len(args.model_names) != len(args.models):
            raise ValueError("Number of model names must match number of models")
        model_names = args.model_names
    
    print(f"Loading evaluation dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split='test')
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Using first {len(dataset)} samples for testing")
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Models to compare: {len(args.models)}")
    print(f"Best-of-{args.n} sampling enabled")
    print(f"Total pairwise comparisons: {len(list(combinations(range(len(args.models)), 2)))}")
    print("")
    
    # Generate responses for all models
    all_model_responses = []
    all_model_rewards = []
    
    for i, (model_path, model_name) in enumerate(zip(args.models, model_names)):
        print(f"\n=== Processing {model_name} ({i+1}/{len(args.models)}) ===")
        
        # Clean up resources before processing each model (except the first one)
        if i > 0:
            cleanup_vllm_resources()
        
        try:
            # Generate responses
            responses_per_prompt = generate_responses(
                model_path, dataset, args.maxlen, args.world_size, args.n
            )
            
            # Get best response per prompt (works for both n=1 and n>1)
            best_responses, best_rewards = get_best_response_per_prompt(
                dataset, responses_per_prompt, args.reward_model, args.n
            )
            
            all_model_rewards.append(best_rewards)
            print(f"✓ {model_name} completed")
            
        except Exception as e:
            print(f"✗ Error processing {model_name}: {e}")
            print("Attempting cleanup and continuing...")
            cleanup_vllm_resources()
            raise
    
    # Final cleanup
    cleanup_vllm_resources()
    
    # Calculate statistics
    print("\n=== Final Results ===")
    
    # Overall statistics
    overall_stats = calculate_overall_stats(all_model_rewards, model_names)
    
    # Pairwise comparisons
    print(f"\nPairwise Win Rates:")
    print("=" * 50)
    pairwise_stats = calculate_pairwise_stats(all_model_rewards, model_names)
    
    # Overall win rates
    overall_win_rates = calculate_overall_win_rates(all_model_rewards, model_names)
    
    print(f"\nTotal evaluation time: {time.time() - st:.2f} seconds")
    
    return {
        "overall_stats": overall_stats,
        "pairwise_stats": pairwise_stats,
        "overall_win_rates": overall_win_rates,
        "model_names": model_names,
        "best_of_n": args.n
    }


if __name__ == "__main__":
    main() 