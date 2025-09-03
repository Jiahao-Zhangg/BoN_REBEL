"""Evaluate models using 3x3 table comparison with pairwise reward comparisons."""

import os
import argparse
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import gc
import time
from contextlib import contextmanager
import subprocess
import sys
import tempfile
import pickle

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm
import tyro
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Args:
    models: List[str] = field(default_factory=lambda: [
        "meta-llama/Llama-3.2-3B-Instruct",
        "zjhhhh/Helpfulness_REBEL_1e4", 
        "zjhhhh/Multi_Preference_REBEL_1e4",
    ])
    """the models to evaluate (Base, REBEL, Ours)"""
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
    output_dir: str = "./table_evaluation_results"
    """directory to save results"""
    output_dataset_name: str = "zjhhhh/table_comparison_evaluation_data"
    """name for the output dataset to push to hub"""


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

            # Explicitly delete the LLM instance to free resources
            del llm
            del tokenizer

            return output

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


def score_all_responses(reward_model_name, all_responses_dict, batch_size=16):
    """Score all responses from all models for all prompts"""
    def get_message(instruction, response):
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]

    print(f"Scoring all responses with {reward_model_name}")
    
    # Clear any existing transformers cache before starting
    clear_transformers_cache()
    
    try:
        # Use simplified pipeline for tokenization only
        rm = SimpleRewardModelPipeline(reward_model_name)

        # Create all messages for all model responses
        all_messages = []
        message_indices = []  # Track which (prompt_idx, model_idx) each message corresponds to
        
        for prompt_idx, prompt in enumerate(all_responses_dict['prompts']):
            for model_idx, model_name in enumerate(all_responses_dict['model_names']):
                response = all_responses_dict['responses'][model_name][prompt_idx]
                message = get_message(prompt, response)
                all_messages.append(message)
                message_indices.append((prompt_idx, model_idx))
        
        # Sort messages by tokenized length, but store original order
        indexed_messages = [(i, msg, rm.tokenized_len(msg)) for i, msg in enumerate(all_messages)]
        indexed_messages.sort(key=lambda x: x[2], reverse=True)
        
        # Extract sorted messages and original indices
        sorted_messages = [x[1] for x in indexed_messages]
        original_order = [x[0] for x in indexed_messages]
        
        # Score using isolated subprocess - returns two reward dimensions
        scores_dim1, scores_dim2 = score_responses_isolated(reward_model_name, sorted_messages, batch_size)
        
        # Revert scores back to original order
        scores_tensor_dim1 = torch.tensor(scores_dim1)
        scores_tensor_dim2 = torch.tensor(scores_dim2)
        indices_tensor = torch.tensor(original_order)
        sorted_scores_dim1 = torch.zeros_like(scores_tensor_dim1)
        sorted_scores_dim2 = torch.zeros_like(scores_tensor_dim2)
        sorted_scores_dim1[indices_tensor] = scores_tensor_dim1
        sorted_scores_dim2[indices_tensor] = scores_tensor_dim2
        
        # Organize scores back by prompt and model
        scores_by_prompt_model = {}
        for i, (prompt_idx, model_idx) in enumerate(message_indices):
            if prompt_idx not in scores_by_prompt_model:
                scores_by_prompt_model[prompt_idx] = {}
            scores_by_prompt_model[prompt_idx][model_idx] = {
                'helpfulness': sorted_scores_dim1[i].item(),
                'correctness': sorted_scores_dim2[i].item()
            }

        return scores_by_prompt_model
        
    except Exception as e:
        print(f"Failed to score with {reward_model_name}: {e}")
        print(f"Skipping this reward model...")
        return {}


def compute_pairwise_tables(scores_by_prompt_model, num_models):
    """Compute the three 3x3 tables as specified"""
    num_prompts = len(scores_by_prompt_model)
    
    # Initialize tables
    table1 = np.zeros((num_models, num_models))  # helpfulness
    table2 = np.zeros((num_models, num_models))  # correctness  
    table3 = np.zeros((num_models, num_models))  # min of both
    
    for row_model in range(num_models):
        for col_model in range(num_models):
            if row_model == col_model:
                # Diagonal entries should be 0.5 (equal comparison)
                table1[row_model, col_model] = 0.5
                table2[row_model, col_model] = 0.5
                table3[row_model, col_model] = 0.5
            else:
                table1_values = []
                table2_values = []
                table3_values = []
                
                for prompt_idx in range(num_prompts):
                    if prompt_idx in scores_by_prompt_model:
                        row_scores = scores_by_prompt_model[prompt_idx][row_model]
                        col_scores = scores_by_prompt_model[prompt_idx][col_model]
                        
                        # Table 1: 1[r_1(x,y) > r_1(x,y')] (0.5 for ties)
                        r1_y = row_scores['helpfulness']
                        r1_y_prime = col_scores['helpfulness']
                        if r1_y > r1_y_prime:
                            table1_val = 1.0
                        elif r1_y == r1_y_prime:
                            table1_val = 0.5
                        else:
                            table1_val = 0.0
                        table1_values.append(table1_val)
                        
                        # Table 2: 1[r_2(x,y) > r_2(x,y')] (0.5 for ties)
                        r2_y = row_scores['correctness']
                        r2_y_prime = col_scores['correctness']
                        if r2_y > r2_y_prime:
                            table2_val = 1.0
                        elif r2_y == r2_y_prime:
                            table2_val = 0.5
                        else:
                            table2_val = 0.0
                        table2_values.append(table2_val)
                        
                        # Table 3: min_i 1[r_i(x,y) > r_i(x,y')]
                        table3_val = min(table1_val, table2_val)
                        table3_values.append(table3_val)
                
                # Average over all prompts
                table1[row_model, col_model] = np.mean(table1_values) if table1_values else 0.0
                table2[row_model, col_model] = np.mean(table2_values) if table2_values else 0.0
                table3[row_model, col_model] = np.mean(table3_values) if table3_values else 0.0
    
    return table1, table2, table3


def create_table_visualizations(table1, table2, table3, model_names, output_dir):
    """Create LaTeX-ready heatmap visualizations for the three tables"""
    
    # Set up the plot with LaTeX-style formatting
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # Model names
    display_names = ['Base', 'REBEL', 'Ours']
    
    # Create the figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Table titles as requested
    titles = [
        'Win Rate on Helpfulness',
        'Win Rate on Correctness', 
        'Win Rate on Worst Dimension'
    ]
    
    # Table data
    tables = [table1, table2, table3]
    
    # Create heatmaps
    for i, (table, title) in enumerate(zip(tables, titles)):
        # Create heatmap without individual colorbars
        im = axes[i].imshow(table, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
        
        # Set title
        axes[i].set_title(title, fontsize=18, pad=20)
        
        # Set tick labels
        axes[i].set_xticks(range(3))
        axes[i].set_yticks(range(3))
        axes[i].set_xticklabels(display_names, fontsize=14)
        axes[i].set_yticklabels(display_names, fontsize=14)
        
        # Add value annotations
        for j in range(3):
            for k in range(3):
                text = axes[i].text(k, j, f'{table[j, k]:.3f}', 
                                  ha="center", va="center", 
                                  color="black", fontsize=12, fontweight='bold')
        
        # Clean up the appearance - no axis labels as requested
        axes[i].tick_params(axis='both', which='major', labelsize=14)
    
    # Add a single colorbar for all plots
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    
    # Save plot
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(output_dir, f"pairwise_comparison_tables_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return plot_filename


def prepare_dataset_for_hub(all_responses_dict, scores_by_prompt_model):
    """Prepare dataset with responses and rewards for hub upload"""
    
    data_rows = []
    
    for prompt_idx, prompt in enumerate(all_responses_dict['prompts']):
        row = {'prompt': prompt}
        
        # Add responses from each model
        for model_idx, model_name in enumerate(all_responses_dict['model_names']):
            model_key = model_name.split('/')[-1]  # Use just the model name
            row[f'response_{model_key}'] = all_responses_dict['responses'][model_name][prompt_idx]
            
            # Add rewards for this model's response
            if prompt_idx in scores_by_prompt_model and model_idx in scores_by_prompt_model[prompt_idx]:
                scores = scores_by_prompt_model[prompt_idx][model_idx]
                row[f'reward_helpfulness_{model_key}'] = scores['helpfulness']
                row[f'reward_correctness_{model_key}'] = scores['correctness']
            else:
                row[f'reward_helpfulness_{model_key}'] = 0.0
                row[f'reward_correctness_{model_key}'] = 0.0
        
        data_rows.append(row)
    
    return Dataset.from_list(data_rows)


def main(args: Args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset {args.input_dataset}...")
    dataset = load_dataset(args.input_dataset, split=args.split)
    if args.end_idx > 0:
        dataset = dataset.select(range(args.end_idx))
    
    prompts = [row['prompt'] for row in dataset]
    
    # Generate responses from all models
    all_responses = {}
    
    for model in args.models:
        print(f"\n{'='*50}")
        print(f"Generating responses with model: {model}")
        print(f"{'='*50}")
        
        responses = generate_single_response(model, dataset, args.maxlen, args.world_size)
        all_responses[model] = responses
        
        # Clear memory after each model
        with cuda_memory_cleanup():
            pass
    
    # Prepare data structure for scoring
    all_responses_dict = {
        'prompts': prompts,
        'model_names': args.models,
        'responses': all_responses
    }
    
    # Score all responses
    print(f"\n{'='*50}")
    print(f"Scoring all responses with reward model: {args.reward_model}")
    print(f"{'='*50}")
    
    clear_transformers_cache()
    scores_by_prompt_model = score_all_responses(args.reward_model, all_responses_dict, args.rm_batch_size)
    
    # Compute the three tables
    print(f"\nComputing pairwise comparison tables...")
    table1, table2, table3 = compute_pairwise_tables(scores_by_prompt_model, len(args.models))
    
    # Create visualizations
    plot_filename = create_table_visualizations(table1, table2, table3, args.models, args.output_dir)
    
    # Save tables as text
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_filename = os.path.join(args.output_dir, f"pairwise_tables_{timestamp}.txt")
    
    model_display_names = ['Base', 'REBEL', 'Ours']
    
    with open(results_filename, 'w') as f:
        f.write("Pairwise Model Comparison Tables\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Table 1: Helpfulness Comparison\n")
        f.write("1[r_1(x,y) > r_1(x,y')]\n")
        f.write("Rows: Row model (y), Columns: Column model (y')\n\n")
        for i, row_name in enumerate(model_display_names):
            f.write(f"{row_name:8s}")
            for j, col_name in enumerate(model_display_names):
                f.write(f"{table1[i,j]:8.3f}")
            f.write("\n")
        f.write("\n")
        
        f.write("Table 2: Correctness Comparison\n")
        f.write("1[r_2(x,y) > r_2(x,y')]\n")
        f.write("Rows: Row model (y), Columns: Column model (y')\n\n")
        for i, row_name in enumerate(model_display_names):
            f.write(f"{row_name:8s}")
            for j, col_name in enumerate(model_display_names):
                f.write(f"{table2[i,j]:8.3f}")
            f.write("\n")
        f.write("\n")
        
        f.write("Table 3: Min(Helpfulness, Correctness)\n") 
        f.write("min_i 1[r_i(x,y) > r_i(x,y')]\n")
        f.write("Rows: Row model (y), Columns: Column model (y')\n\n")
        for i, row_name in enumerate(model_display_names):
            f.write(f"{row_name:8s}")
            for j, col_name in enumerate(model_display_names):
                f.write(f"{table3[i,j]:8.3f}")
            f.write("\n")
    
    # Prepare dataset for hub upload
    print(f"\nPreparing dataset for hub upload...")
    hub_dataset = prepare_dataset_for_hub(all_responses_dict, scores_by_prompt_model)
    
    # Save dataset locally first
    dataset_filename = os.path.join(args.output_dir, f"comparison_dataset_{timestamp}")
    hub_dataset.save_to_disk(dataset_filename)
    
    # Push to hub
    print(f"Pushing dataset to hub: {args.output_dataset_name}")
    try:
        hub_dataset.push_to_hub(args.output_dataset_name)
        print(f"Successfully uploaded to https://huggingface.co/datasets/{args.output_dataset_name}")
    except Exception as e:
        print(f"Failed to push to hub: {e}")
        print(f"Dataset saved locally at: {dataset_filename}")
        print(f"You can manually upload later using:")
        print(f"  from datasets import Dataset")
        print(f"  dataset = Dataset.load_from_disk('{dataset_filename}')")
        print(f"  dataset.push_to_hub('{args.output_dataset_name}')")
    
    print(f"\nResults saved to:")
    print(f"  Tables plot: {plot_filename}")
    print(f"  Tables text: {results_filename}")
    print(f"  Dataset: {dataset_filename}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
