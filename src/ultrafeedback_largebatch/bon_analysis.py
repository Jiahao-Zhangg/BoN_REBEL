"""Evaluate reward as best-of-n increases."""

import os
import argparse
import random
from dataclasses import dataclass, field
from typing import List
import gc
import time
from contextlib import contextmanager
import subprocess
import sys
import tempfile
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm
from typing import Dict, List
import tyro
from datasets import load_dataset


@dataclass
class Args:
    models: List[str] = field(default_factory=lambda: [
        "meta-llama/Llama-3.2-3B-Instruct",
        # "/work2/lujingz/model3b",
        # "/work2/lujingz/model3b_bon_rerun",
        "/work2/lujingz/model_3b_multi",
    ])
    """the model to evaluate"""
    input_dataset: str = "zjhhhh/Whole-Data-Llama-3.2-3B-Instruct-20_armo_tokenized_Llama-3.2-3B-Instruct_slice30"
    """the input dataset"""
    split: str = "test"
    """the split of the dataset"""
    end_idx: int = -1
    """the end index of the dataset"""
    n: int = 5
    """best-of-n"""
    reward_models: List[str] = field(default_factory=lambda: [
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        "sfairXC/FsfairX-LLaMA3-RM-v0.1",
    ])
    """reward models to evaluate responses on"""
    maxlen: int = 1024
    """the maximum length of the responses"""
    world_size: int = 1
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


def clear_transformers_cache():
    """Aggressively clear all transformers caches"""
    import transformers
    
    # Clear model cache
    if hasattr(transformers, '_model_cache'):
        transformers._model_cache.clear()
    
    # Clear tokenizer cache  
    if hasattr(transformers, '_tokenizer_cache'):
        transformers._tokenizer_cache.clear()
    
    # Clear config cache
    if hasattr(transformers, '_config_cache'):
        transformers._config_cache.clear()
    
    # Clear dynamic module imports
    import sys
    modules_to_remove = []
    for name in sys.modules.keys():
        if ('transformers_modules' in name or 
            'modeling_custom' in name or
            'configuration_custom' in name):
            modules_to_remove.append(name)
    
    for name in modules_to_remove:
        if name in sys.modules:
            del sys.modules[name]
    
    # Force garbage collection
    gc.collect()


def score_responses_isolated(reward_model_name, messages_data, batch_size=16):
    """Score responses in an isolated subprocess to avoid model conflicts"""
    
    # Create a temporary script for isolated scoring
    script_content = f'''
import torch
import pickle
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List

def load_and_score(model_id, messages_data, batch_size):
    try:
        # Load model fresh in this process
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        
        device = model.device
        max_length = 4096
        
        all_scores = []
        
        for batch_start in range(0, len(messages_data), batch_size):
            batch_messages = messages_data[batch_start:batch_start + batch_size]
            
            input_ids = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            
            with torch.no_grad():
                output = model(input_ids)
                
                # Handle different output formats
                if hasattr(output, "score"):
                    score_tensor = output.score
                elif hasattr(output, "rewards"):
                    score_tensor = output.rewards
                else:
                    logits = output.logits
                    
                    # Apply different processing based on model type
                    if "ArmoRM" in model_id or "RLHFlow" in model_id:
                        # ArmoRM models might have different output format
                        if logits.ndim == 2 and logits.size(-1) > 1:
                            score_tensor = logits[:, -1]  # Take last logit
                        else:
                            score_tensor = logits.squeeze(-1)
                    else:
                        # Standard reward model processing
                        logits = logits / 30 + 0.1  # Normalize logits
                        if logits.ndim == 2 and logits.size(-1) > 1:
                            score_tensor = logits[:, 1]  # Take positive class
                        else:
                            score_tensor = logits.squeeze(-1)
                
                scores = score_tensor.float().detach().cpu().tolist()
                all_scores.extend(scores)
        
        return all_scores
        
    except Exception as e:
        print(f"Error in isolated scoring: {{e}}")
        return [0.0] * len(messages_data)

if __name__ == "__main__":
    model_id = sys.argv[1]
    input_file = sys.argv[2] 
    output_file = sys.argv[3]
    batch_size = int(sys.argv[4])
    
    with open(input_file, 'rb') as f:
        messages_data = pickle.load(f)
    
    scores = load_and_score(model_id, messages_data, batch_size)
    
    with open(output_file, 'wb') as f:
        pickle.dump(scores, f)
'''
    
    # Write the script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    # Write messages data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(messages_data, f)
        input_file = f.name
    
    # Create output file path
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        output_file = f.name
    
    try:
        # Run the scoring in a subprocess
        result = subprocess.run([
            sys.executable, script_path, 
            reward_model_name, input_file, output_file, str(batch_size)
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            print(f"Subprocess failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return [0.0] * len(messages_data)
        
        # Load the results
        with open(output_file, 'rb') as f:
            scores = pickle.load(f)
        
        return scores
        
    except subprocess.TimeoutExpired:
        print(f"Scoring timed out for {reward_model_name}")
        return [0.0] * len(messages_data)
    except Exception as e:
        print(f"Error running isolated scoring: {e}")
        return [0.0] * len(messages_data)
    finally:
        # Clean up temporary files
        for temp_file in [script_path, input_file, output_file]:
            try:
                os.unlink(temp_file)
            except:
                pass


class SimpleRewardModelPipeline:
    """Simplified pipeline that uses isolated scoring"""
    
    def __init__(self, model_id):
        self.model_id = model_id
        # Load tokenizer only for length calculation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                use_fast=True,
                trust_remote_code=True
            )
        except:
            # Fallback tokenizer if model-specific fails
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct"
            )
    
    def tokenized_len(self, message: Dict[str, str]) -> int:
        try:
            return len(self.tokenizer.apply_chat_template(
                message,
                truncation=True,
                max_length=4096,
            ))
        except:
            # Fallback length estimation
            return len(str(message)) // 4


def score_responses(reward_model_name, dataset, n, batch_size=16):
    def get_message(instruction, response):
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]

    print(f"Scoring with isolated subprocess for {reward_model_name}")
    
    # Clear any existing transformers cache before starting
    clear_transformers_cache()
    
    try:
        # Use simplified pipeline for tokenization only
        rm = SimpleRewardModelPipeline(reward_model_name)

        rewards = {}
        for i in range(n):
            print(f"Scoring response set {i+1}/{n}")
            
            # Create messages
            messages = [(msg_i, get_message(row['prompt'], row[f'response_{i}_test'])) 
                       for msg_i, row in enumerate(dataset)]
            
            # Sort messages by tokenized length, but store original order
            messages.sort(key=lambda x: rm.tokenized_len(x[1]), reverse=True)
            
            # Extract just the messages for scoring
            sorted_messages = [x[1] for x in messages]
            original_indices = [x[0] for x in messages]
            
            # Score using isolated subprocess
            scores = score_responses_isolated(reward_model_name, sorted_messages, batch_size)
            
            # Revert scores back to original order
            scores_tensor = torch.tensor(scores)
            indices_tensor = torch.tensor(original_indices)
            sorted_scores = torch.zeros_like(scores_tensor)
            sorted_scores[indices_tensor] = scores_tensor
            
            rewards[f"response_{i}_reward_{reward_model_name}"] = sorted_scores.tolist()

        # Add all reward columns to dataset
        for k, v in rewards.items():
            dataset = dataset.add_column(k, v)

        return dataset
        
    except Exception as e:
        print(f"Failed to score with {reward_model_name}: {e}")
        print(f"Skipping this reward model...")
        return dataset


def get_bon_data(reward_model_name, dataset, n):
    rewards = np.zeros((len(dataset), n))
    for i in range(n):
        column_name = f"response_{i}_reward_{reward_model_name}"
        if column_name in dataset.column_names:
            rewards[:, i] = dataset[column_name]
        else:
            print(f"Warning: Column {column_name} not found, using zeros")

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
            
            # Clear cache before each reward model
            clear_transformers_cache()
            
            dataset = score_responses(reward_model_name, dataset, args.n, args.rm_batch_size)
            reward_results[reward_model_name] = get_bon_data(reward_model_name, dataset, args.n)

        results[model] = reward_results

    # Plot the reward as n increases
    for reward_model_name in args.reward_models:
        if reward_model_name in results[args.models[0]]:  # Only plot if we have results
            print(f"Plotting rewards with reward model: {reward_model_name}")

            plt.figure(figsize=(10, 6))
            
            for model in args.models:
                if reward_model_name in results[model]:
                    plt.plot(range(1, args.n+1), results[model][reward_model_name], 
                            label=model, marker='o')

            plt.xlabel("Best-of-N")
            plt.ylabel("Average Reward")
            plt.title(f"Best-of-N Performance - {reward_model_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            filename = f"bon_rewards_{reward_model_name.replace('/', '_')}"
            if os.path.exists(filename + ".png"):
                filename += f"_{time.strftime('%Y%m%d_%H%M%S')}"
            plt.savefig(filename + ".png", dpi=150, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)