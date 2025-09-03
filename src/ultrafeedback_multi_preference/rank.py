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
    parser.add_argument("--rewards", type=str, nargs='+', help="List of rewards to use", default=ArmoRMPipeline.all_reward_names)
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from generate.py")
    parser.add_argument("--num_responses", type=int, default=20, help="Number of responses to generate rewards for")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    return parser.parse_args()


def main():

    # init
    args = parse_arguments()
    dataset = load_dataset(args.input_repo, split='train')

    # setup reward model
    rm = ArmoRMPipeline(reward_names=args.rewards, trust_remote_code=True)

    # gather reward for responses
    for i in range(args.num_responses):
        print(f'gathering reward for {i+1}th response')

        # Create messages
        messages = [get_message(row['prompt'], row[f'response_{i}']) for row in dataset]
        
        # Generate rewards in batches
        all_rewards = []
        for batch_idx in tqdm(range(0, len(messages), args.batch_size)):
            batch_messages = messages[batch_idx:batch_idx+args.batch_size]
            batch_rewards = rm(batch_messages)
            all_rewards.append(batch_rewards)
        
        rewards = torch.cat(all_rewards)

        for reward_idx in range(len(args.rewards)):
            dataset = dataset.add_column(
                f"reward_{reward_idx}_response_{i}", 
                rewards[:, reward_idx].tolist()
            )

    dataset.push_to_hub(args.input_repo+'_multi_armo')


if __name__ == "__main__":
    main()
