import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List
import multiprocessing as mp
import os

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_models", type=str, nargs='+', default=["sfairXC/FsfairX-LLaMA3-RM-v0.1"], help="List of reward models to use")
    # "RLHFlow/ArmoRM-Llama3-8B-v0.1", 
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from generate.py")
    parser.add_argument("--selection_pairs", type=int, default=5, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=5, help="number of pairs to use for gradient estimation")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="Specific GPU IDs to use (e.g., --gpu_ids 0 2 3)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    return parser.parse_args()


def get_message(instruction, response):
    return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]


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

    def __call__(self, messages_batch: List[List[Dict[str, str]]]) -> List[float]:
        """Process a batch of messages and return a list of scores."""
        if not isinstance(messages_batch[0], list):
            # Handle single message case for backward compatibility
            messages_batch = [messages_batch]
            single_message = True
        else:
            single_message = False
            
        # Tokenize all messages in the batch
        all_input_ids = []
        for messages in messages_batch:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                truncation=self.truncation,
                max_length=self.max_length,
            )
            all_input_ids.append(input_ids.squeeze(0))  # Remove batch dimension
        
        # Pad sequences to the same length
        input_ids_batch = self.tokenizer.pad(
            {"input_ids": all_input_ids},
            return_tensors="pt",
            padding=True
        )["input_ids"].to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids_batch)
            if hasattr(output, "score"):
                score_tensor = output.score
            else:
                logits = output.logits/30+0.1  # shape [batch_size, …]

                if logits.ndim == 2 and logits.size(-1) > 1:
                    score_tensor = logits[:, 1]
                else:
                    score_tensor = logits.squeeze(-1)
            scores = score_tensor.float().tolist()
        
        if single_message:
            return scores[0]
        return scores


def process_model_on_gpu(args_tuple):
    """Process a single reward model on a specific GPU."""
    model_id, gpu_id, dataset_dict, selection_indices, gradient_indices, model_idx, batch_size = args_tuple
    
    # Set the GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # Since we set CUDA_VISIBLE_DEVICES, use device 0
    
    # Initialize the reward model
    rm = ArmoRMPipeline(model_id, device_map="cuda:0", trust_remote_code=True)
    
    # Process all responses for this model
    results = {}
    
    # Process selection pairs
    for i in selection_indices:
        rewards = []
        print(f'GPU {gpu_id}: Processing model {model_idx+1}, selection response {i+1}')
        
        # Process in batches
        for batch_start in tqdm(range(0, len(dataset_dict), batch_size), desc=f"GPU {gpu_id}, Model {model_idx+1}, Selection Response {i+1}"):
            batch_end = min(batch_start + batch_size, len(dataset_dict))
            batch_messages = []
            
            for row_idx in range(batch_start, batch_end):
                row = dataset_dict[row_idx]
                prompt = row['prompt']
                response = row[f'response_{i}']
                batch_messages.append(get_message(prompt, response))
            
            # Process the batch
            batch_rewards = rm(batch_messages)
            rewards.extend(batch_rewards)
        
        results[f"reward_{model_idx}_response_{i}"] = rewards
    
    # Process gradient pairs
    for i in gradient_indices:
        rewards = []
        print(f'GPU {gpu_id}: Processing model {model_idx+1}, gradient response {i+1}')
        
        # Process in batches
        for batch_start in tqdm(range(0, len(dataset_dict), batch_size), desc=f"GPU {gpu_id}, Model {model_idx+1}, Gradient Response {i+1}"):
            batch_end = min(batch_start + batch_size, len(dataset_dict))
            batch_messages = []
            
            for row_idx in range(batch_start, batch_end):
                row = dataset_dict[row_idx]
                prompt = row['prompt']
                response = row[f'response_{i}']
                batch_messages.append(get_message(prompt, response))
            
            # Process the batch
            batch_rewards = rm(batch_messages)
            rewards.extend(batch_rewards)
        
        results[f"reward_{model_idx}_response_{i}_for_gradient"] = rewards
    
    return results


def main():
    # init
    args = parse_arguments()
    dataset = load_dataset(args.input_repo, split='train')
    # Remove existing response columns to avoid conflicts
    response_columns = [col for col in dataset.column_names if col.endswith('reward')]
    print(response_columns)
    if response_columns:
        print(f"Renaming existing reward columns: {response_columns}")
        names = [f"reward_1_response_{i}" for i in range(5)] + [f"reward_1_response_{i}_for_gradient" for i in range(5,10)]
        rename_map = dict(zip(response_columns, names))
        dataset = dataset.rename_columns(rename_map)
    print(dataset.column_names)

    
    # Determine which GPUs to use
    if args.gpu_ids is not None:
        # Use specified GPU IDs
        available_gpus = args.gpu_ids
        args.num_gpus = len(available_gpus)
        print(f"Using specified GPUs: {available_gpus}")
    else:
        # Auto-detect or use num_gpus
        total_gpus = torch.cuda.device_count()
        if args.num_gpus is None:
            args.num_gpus = total_gpus
        available_gpus = list(range(min(args.num_gpus, total_gpus)))
        print(f"Using auto-detected GPUs: {available_gpus}")
    
    print(f"Total available GPUs: {torch.cuda.device_count()}")
    print(f"Using {len(available_gpus)} GPUs for parallel processing")
    
    # Convert dataset to list of dictionaries for multiprocessing
    dataset_dict = [dict(row) for row in dataset]
    
    # Calculate indices for selection and gradient pairs separately
    selection_indices = list(range(args.selection_pairs))
    gradient_indices = list(range(args.selection_pairs, args.selection_pairs + args.gradient_pairs))
    
    print(f"Selection pairs: responses {selection_indices[0]}-{selection_indices[-1]} (total: {len(selection_indices)})")
    print(f"Gradient pairs: responses {gradient_indices[0]}-{gradient_indices[-1]} (total: {len(gradient_indices)})")
    print(f"Batch size: {args.batch_size}")
    
    # Prepare arguments for multiprocessing
    process_args = []
    for model_idx, model_id in enumerate(args.reward_models):
        gpu_id = available_gpus[model_idx % len(available_gpus)]  # Distribute models across specified GPUs
        process_args.append((model_id, gpu_id, dataset_dict, selection_indices, gradient_indices, model_idx, args.batch_size))
    
    # Use multiprocessing to process models in parallel
    print(f"Starting parallel processing with {len(process_args)} processes...")
    print("GPU assignments:")
    for model_idx, (model_id, gpu_id, _, _, _, _, _) in enumerate(process_args):
        print(f"  Model {model_idx+1} ({model_id}) -> GPU {gpu_id}")
    
    with mp.Pool(processes=min(len(args.reward_models), len(available_gpus))) as pool:
        results_list = pool.map(process_model_on_gpu, process_args)
    
    # Combine results from all processes
    all_rewards = {}
    for results in results_list:
        all_rewards.update(results)
    
    # Add all reward columns to dataset
    for k, v in all_rewards.items():
        if k in dataset.column_names:
            dataset = dataset.remove_columns([k])
        dataset = dataset.add_column(k, v)

    dataset.push_to_hub(args.input_repo+'_multi_armo')


if __name__ == "__main__":
    main()