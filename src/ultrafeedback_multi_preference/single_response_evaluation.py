"""Evaluate models by generating single responses and plotting reward dimensions."""

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
        "zjhhhh/REBEL_1e4_ver2",
        "zjhhhh/Multi_Preference_REBEL_1e4",
        "zjhhhh/BoN_REBEL_1e4",
    ])
    """the models to evaluate"""
    input_dataset: str = "zjhhhh/Whole-Data-Llama-3.2-3B-Instruct-20_armo_tokenized_Llama-3.2-3B-Instruct_slice30"
    """the input dataset"""
    split: str = "test"
    """the split of the dataset"""
    end_idx: int = -1
    """the end index of the dataset"""
    reward_model: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    """reward model to evaluate responses on"""
    maxlen: int = 1024
    """the maximum length of the responses"""
    world_size: int = 1
    """the number of GPUs to use"""
    rm_batch_size: int = 2
    """the batch size for the reward model"""
    output_dir: str = "./evaluation_results"
    """directory to save results"""


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


def generate_single_response(model_path, dataset, maxlen, world_size):
    """Generate single response per prompt using the given model"""
    print(f"Generating single response per prompt with model: {model_path}")

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
                gpu_memory_utilization=0.7,
            )

            # Prepare prompts
            prompts = [tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=False, add_generation_prompt=True) for row in tqdm(dataset)]

            # Generate single response with fixed seed for reproducibility
            set_seed(42)
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.9,
                max_tokens=maxlen,
                seed=42,
            )
            response = llm.generate(prompts, sampling_params)
            output = list(map(lambda x: x.outputs[0].text, response))
            dataset = dataset.add_column("response", output)

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
    
    # Create a temporary script for isolated scoring that returns first two reward dimensions
    script_content = f'''
import torch
import pickle
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List

def load_and_score_dual_rewards(model_id, messages_data, batch_size):
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
        
        all_scores_dim1 = []
        all_scores_dim2 = []
        
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
                
                # For ArmoRM, we want the first two reward dimensions
                if hasattr(output, "rewards"):
                    rewards = output.rewards
                    # Take first two dimensions (helpsteer-helpfulness, helpsteer-correctness)
                    if rewards.shape[-1] >= 2:
                        scores_dim1 = rewards[:, 0].float().detach().cpu().tolist()
                        scores_dim2 = rewards[:, 1].float().detach().cpu().tolist()
                    else:
                        # Fallback if not enough dimensions
                        scores_dim1 = rewards[:, 0].float().detach().cpu().tolist()
                        scores_dim2 = [0.0] * len(batch_messages)
                else:
                    # Fallback for other models
                    if hasattr(output, "score"):
                        score_tensor = output.score
                    else:
                        logits = output.logits
                        if logits.ndim == 2 and logits.size(-1) > 1:
                            score_tensor = logits[:, -1]
                        else:
                            score_tensor = logits.squeeze(-1)
                    
                    scores_dim1 = score_tensor.float().detach().cpu().tolist()
                    scores_dim2 = [0.0] * len(batch_messages)  # No second dimension
                
                all_scores_dim1.extend(scores_dim1)
                all_scores_dim2.extend(scores_dim2)
        
        return all_scores_dim1, all_scores_dim2
        
    except Exception as e:
        print(f"Error in isolated scoring: {{e}}")
        return [0.0] * len(messages_data), [0.0] * len(messages_data)

if __name__ == "__main__":
    model_id = sys.argv[1]
    input_file = sys.argv[2] 
    output_file = sys.argv[3]
    batch_size = int(sys.argv[4])
    
    with open(input_file, 'rb') as f:
        messages_data = pickle.load(f)
    
    scores_dim1, scores_dim2 = load_and_score_dual_rewards(model_id, messages_data, batch_size)
    
    with open(output_file, 'wb') as f:
        pickle.dump((scores_dim1, scores_dim2), f)
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
            return [0.0] * len(messages_data), [0.0] * len(messages_data)
        
        # Load the results
        with open(output_file, 'rb') as f:
            scores_dim1, scores_dim2 = pickle.load(f)
        
        return scores_dim1, scores_dim2
        
    except subprocess.TimeoutExpired:
        print(f"Scoring timed out for {reward_model_name}")
        return [0.0] * len(messages_data), [0.0] * len(messages_data)
    except Exception as e:
        print(f"Error running isolated scoring: {e}")
        return [0.0] * len(messages_data), [0.0] * len(messages_data)
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


def score_responses(reward_model_name, dataset, batch_size=16):
    def get_message(instruction, response):
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]

    print(f"Scoring with isolated subprocess for {reward_model_name}")
    
    # Clear any existing transformers cache before starting
    clear_transformers_cache()
    
    try:
        # Use simplified pipeline for tokenization only
        rm = SimpleRewardModelPipeline(reward_model_name)

        print(f"Scoring responses...")
        
        # Create messages
        messages = [(msg_i, get_message(row['prompt'], row['response'])) 
                   for msg_i, row in enumerate(dataset)]
        
        # Sort messages by tokenized length, but store original order
        messages.sort(key=lambda x: rm.tokenized_len(x[1]), reverse=True)
        
        # Extract just the messages for scoring
        sorted_messages = [x[1] for x in messages]
        original_indices = [x[0] for x in messages]
        
        # Score using isolated subprocess - returns two reward dimensions
        scores_dim1, scores_dim2 = score_responses_isolated(reward_model_name, sorted_messages, batch_size)
        
        # Revert scores back to original order
        scores_tensor_dim1 = torch.tensor(scores_dim1)
        scores_tensor_dim2 = torch.tensor(scores_dim2)
        indices_tensor = torch.tensor(original_indices)
        sorted_scores_dim1 = torch.zeros_like(scores_tensor_dim1)
        sorted_scores_dim2 = torch.zeros_like(scores_tensor_dim2)
        sorted_scores_dim1[indices_tensor] = scores_tensor_dim1
        sorted_scores_dim2[indices_tensor] = scores_tensor_dim2
        
        # Add reward columns to dataset
        dataset = dataset.add_column("reward_helpfulness", sorted_scores_dim1.tolist())
        dataset = dataset.add_column("reward_correctness", sorted_scores_dim2.tolist())

        return dataset
        
    except Exception as e:
        print(f"Failed to score with {reward_model_name}: {e}")
        print(f"Skipping this reward model...")
        return dataset


def compute_average_rewards(dataset):
    """Compute average rewards for both dimensions"""
    helpfulness_scores = dataset["reward_helpfulness"]
    correctness_scores = dataset["reward_correctness"]
    
    avg_helpfulness = np.mean(helpfulness_scores)
    avg_correctness = np.mean(correctness_scores)
    
    return avg_helpfulness, avg_correctness


def main(args: Args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    for model in args.models:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model}")
        print(f"{'='*50}")
        
        # Setup
        print(f"Loading dataset {args.input_dataset}...")
        dataset = load_dataset(args.input_dataset, split=args.split)
        if args.end_idx > 0:
            dataset = dataset.select(range(args.end_idx))

        # Generate single response per prompt
        print(f"Generating single response per prompt with model: {model}")
        dataset = generate_single_response(model, dataset, args.maxlen, args.world_size)

        # Score responses with reward model for both dimensions
        print(f"Scoring responses with reward model: {args.reward_model}")
        
        # Clear cache before scoring
        clear_transformers_cache()
        
        dataset = score_responses(args.reward_model, dataset, args.rm_batch_size)
        avg_helpfulness, avg_correctness = compute_average_rewards(dataset)
        
        results[model] = {
            'helpfulness': avg_helpfulness,
            'correctness': avg_correctness
        }
        
        print(f"Results for {model}:")
        print(f"  Average Helpfulness: {avg_helpfulness:.4f}")
        print(f"  Average Correctness: {avg_correctness:.4f}")

    # Create scatter plot
    print(f"\nCreating scatter plot...")
    
    plt.figure(figsize=(10, 8))
    
    model_names = []
    helpfulness_scores = []
    correctness_scores = []
    
    for model_name, scores in results.items():
        model_names.append(model_name.split('/')[-1])  # Use just the model name without path
        helpfulness_scores.append(scores['helpfulness'])
        correctness_scores.append(scores['correctness'])
    
    # Create scatter plot
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
    
    for i, (name, help_score, corr_score) in enumerate(zip(model_names, helpfulness_scores, correctness_scores)):
        plt.scatter(help_score, corr_score, c=[colors[i]], s=100, alpha=0.7, label=name)
        
        # Add model name as text annotation
        plt.annotate(name, (help_score, corr_score), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.xlabel("Helpfulness Score")
    plt.ylabel("Correctness Score")
    plt.title("Model Performance: Helpfulness vs Correctness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(args.output_dir, f"helpfulness_vs_correctness_{timestamp}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    results_filename = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.txt")
    with open(results_filename, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        for model_name, scores in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  Helpfulness: {scores['helpfulness']:.4f}\n")
            f.write(f"  Correctness: {scores['correctness']:.4f}\n\n")
    
    print(f"Results saved to:")
    print(f"  Plot: {plot_filename}")
    print(f"  Text results: {results_filename}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
