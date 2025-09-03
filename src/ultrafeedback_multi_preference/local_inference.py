import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
import os
from multiprocessing import Process
from time import sleep


def get_message(instruction):
    """Format instruction as a chat message."""
    message = [
        {"role": "user", "content": instruction},
    ]
    return message








def set_seed(seed=5775709):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate responses using vLLM with data parallelism")
    
    # Model and data arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name or path")
    parser.add_argument("--prompts", type=str, default="viswavi/wildchecklists", help="Dataset to load prompts from")
    parser.add_argument("--output_repo", type=str, help="Output repo for the generated responses")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for dataset")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index for dataset")
    parser.add_argument("--selection_pairs", type=int, default=1, help="Number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=1, help="Number of pairs to use for gradient estimation")
    
    # Data parallelism arguments
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--node-size", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of the current node")
    parser.add_argument("--master-addr", type=str, default="", help="Master node IP address")
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    
    # vLLM engine arguments
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager mode execution")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Maximum number of sequences to be processed in a single iteration")
    parser.add_argument("--max-model-len", type=int, help="Maximum number of tokens to be processed in a single iteration")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="Fraction of GPU memory vLLM is allowed to allocate")
    parser.add_argument("--timeout", type=int, default=300, help="Number of seconds before unresponsive process is killed")
    
    return parser.parse_args()


def worker_main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    tp_size,
    args
):
    """Main function for each data parallel worker."""
    # Set environment variables for data parallelism
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Initialize tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load dataset
    print(f"DP rank {global_dp_rank}: Loading dataset: {args.prompts}")
    dataset = load_dataset(args.prompts, split='train')
    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))
    
    # Prepare all prompts using chat template
    all_prompts = [tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=False, add_generation_prompt=True) for row in dataset]

    # Distribute prompts across DP ranks
    floor = len(all_prompts) // dp_size
    remainder = len(all_prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    # Each rank processes different part of the dataset
    rank_prompts = all_prompts[start(global_dp_rank):start(global_dp_rank + 1)]
    if len(rank_prompts) == 0:
        rank_prompts = ["Placeholder"]  # Avoid empty prompts
    
    print(f"DP rank {global_dp_rank}: Processing {len(rank_prompts)} prompts")

    # Create LLM instance
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Generate responses for multiple iterations
    total_pairs = args.selection_pairs + args.gradient_pairs
    all_rank_responses = {}
    
    for p in range(total_pairs):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=p * 50,
        )
        
        # Generate responses for this rank's prompts
        outputs = llm.generate(rank_prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        # Store responses for this rank and pair
        all_rank_responses[f"response_{p}"] = responses
        print(f"DP rank {global_dp_rank}: Generated {len(responses)} responses for pair {p}")

    # Save responses to file for this rank
    import pickle
    rank_output_file = f"rank_{global_dp_rank}_responses.pkl"
    with open(rank_output_file, 'wb') as f:
        pickle.dump({
            'rank': global_dp_rank,
            'start_idx': start(global_dp_rank),
            'end_idx': start(global_dp_rank + 1),
            'responses': all_rank_responses,
            'original_dataset_slice': dataset.select(range(start(global_dp_rank), start(global_dp_rank + 1))) if start(global_dp_rank + 1) <= len(dataset) else dataset.select([])
        }, f)
    
    print(f"DP rank {global_dp_rank}: Saved responses to {rank_output_file}")
    sleep(1)  # Give engines time to pause


def combine_rank_results(args, dp_size):
    """Combine results from all data parallel ranks."""
    import pickle
    
    # Load original dataset to get the full structure
    dataset = load_dataset(args.prompts, split='train')
    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))
    
    total_pairs = args.selection_pairs + args.gradient_pairs
    
    # Initialize combined responses
    combined_responses = {}
    for p in range(total_pairs):
        combined_responses[f"response_{p}"] = [""] * len(dataset)
    
    # Load and combine results from all ranks
    for rank in range(dp_size):
        rank_file = f"rank_{rank}_responses.pkl"
        try:
            with open(rank_file, 'rb') as f:
                rank_data = pickle.load(f)
            
            start_idx = rank_data['start_idx']
            end_idx = rank_data['end_idx']
            rank_responses = rank_data['responses']
            
            # Place rank responses in correct positions
            for response_key, responses in rank_responses.items():
                if end_idx <= len(dataset):
                    for i, response in enumerate(responses):
                        if start_idx + i < len(combined_responses[response_key]):
                            combined_responses[response_key][start_idx + i] = response
            
            # Clean up rank file
            os.remove(rank_file)
            print(f"Processed and cleaned up {rank_file}")
            
        except Exception as e:
            print(f"Warning: Failed to load results from rank {rank}: {e}")
    
    # Add response columns to dataset
    for response_key, responses in combined_responses.items():
        dataset = dataset.add_column(response_key, responses)
    
    # Clean and prepare for upload
    columns = ["prompt", "requirements"] + [f"response_{i}" for i in range(total_pairs)]
    dataset = dataset.select_columns(columns)
    
    # Try to upload to Hugging Face first
    if args.output_repo:
        try:
            dataset.push_to_hub(args.output_repo)
            print(f"✓ Successfully pushed to {args.output_repo}")
        except Exception as e:
            print(f"✗ Failed to push to hub: {e}")
            print("Saving locally as fallback...")
            dataset.save_to_disk("./generated_dataset_backup")
            print("Dataset saved locally to ./generated_dataset_backup")
    else:
        print("No output_repo specified, saving locally...")
        dataset.save_to_disk("./generated_dataset_backup")
        print("Dataset saved locally to ./generated_dataset_backup")


def main():
    args = parse_arguments()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    # Set up master IP and port for data parallelism coordination
    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    print(f"Starting data parallel inference with {dp_size} DP ranks, {tp_size} TP size")
    print(f"Master: {dp_master_ip}:{dp_master_port}")

    # Start data parallel workers
    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=worker_main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args
            ),
        )
        proc.start()
        procs.append(proc)
        print(f"Started DP worker {global_dp_rank} (local rank {local_dp_rank})")

    # Wait for all workers to complete
    exit_code = 0
    for proc in procs:
        proc.join(timeout=args.timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within timeout.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    # If all workers completed successfully, combine results
    if exit_code == 0:
        print("All workers completed successfully. Combining results...")
        combine_rank_results(args, dp_size)
        print("✓ Dataset generation completed!")
    else:
        print(f"Some workers failed with exit code {exit_code}")

    exit(exit_code)


if __name__ == "__main__":
    main()
