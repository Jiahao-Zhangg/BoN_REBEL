import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List
import multiprocessing as mp
import os

from reward_model import ArmoRMPipeline
from utils import get_message

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards", type=str, nargs='+', help="List of rewards to use",
                        default=["helpsteer-helpfulness", "helpsteer-correctness"])
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from generate.py")
    parser.add_argument("--selection_pairs", type=int, default=3, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=3, help="number of pairs to use for gradient estimation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    return parser.parse_args()


def main():

    # init
    args = parse_arguments()
    dataset = load_dataset(args.input_repo, split='train')

    # setup reward model
    rm = ArmoRMPipeline(reward_names=args.rewards, trust_remote_code=True)

    # gather reward
    total_pairs = args.selection_pairs + args.gradient_pairs
    for i in range(total_pairs):
        print(f'gathering reward for {i+1}th response')

        # Create messages
        messages = [(msg_i, get_message(row['prompt'], row[f'response_{i}'])) for msg_i, row in enumerate(dataset)]
        
        # Sort messages by tokenized length, but store original order
        messages.sort(key=lambda x: rm.tokenized_len(x[1]), reverse=True)
        
        # Generate rewards in batches
        _rewards, _indices = [], []
        for batch_idx in tqdm(range(0, len(messages), args.batch_size)):
            batch = messages[batch_idx:batch_idx+args.batch_size]
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

        for reward_idx in range(len(args.rewards)):
            if i < args.selection_pairs:
                dataset = dataset.add_column(
                    f"reward_{reward_idx}_response_{i}", 
                    sorted_rewards[:, reward_idx].tolist()
                )
            else:
                dataset = dataset.add_column(
                    f"reward_{reward_idx}_response_{i-args.selection_pairs}_for_gradient", 
                    sorted_rewards[:, reward_idx].tolist()
                )

    dataset.push_to_hub(args.input_repo+'_multi_armo')


if __name__ == "__main__":
    main()
