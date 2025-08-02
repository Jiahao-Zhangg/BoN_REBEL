import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

from utils import get_message

torch.set_printoptions(threshold=10_000)


# WARNING: Magic number, make sure it works for your model
SYS_PROMPT_LEN = 30


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from rank.py")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--selection_pairs", type=int, default=3, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=3, help="number of pairs to use for gradient estimation")
    parser.add_argument("--num_rewards", type=int, required=True, help="number of rewards used in rank.py")
    return parser.parse_args()


def filter_same_responses(row):
    return row['chosen'] != row['reject']


def main():

    # init
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})

    dataset = load_dataset(args.input_repo, split='train')
    
    # process dataset
    print('initial length:', len(dataset))

    # filter dataset with long prompt or response
    dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    print('filtered long prompts:', len(dataset))
    for i in range(args.selection_pairs):
        dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(response=row[f'response_{i}']), tokenize=True, add_generation_prompt=False, return_tensors='pt')[:, SYS_PROMPT_LEN:].shape[-1] <= args.maxlen)
        print(f'filtered response_{i}:', len(dataset))

    # add prompt tokens
    def tokenize_prompt(row):
        llama_prompt_token = tokenizer_left.apply_chat_template(
                get_message(row['prompt']), 
                add_generation_prompt=True,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen_prompt,
        )
        llama_prompt = tokenizer_left.decode(llama_prompt_token, skip_special_tokens=False)
        assert len(llama_prompt_token) == args.maxlen_prompt
        assert (llama_prompt_token[0] == 128000 or llama_prompt_token[0] == 128256) and llama_prompt_token[-1] == 271
        row['llama_prompt'] = llama_prompt
        row['llama_prompt_tokens'] = llama_prompt_token
        return row
    dataset = dataset.map(tokenize_prompt, num_proc=4)

    # select chosen and reject    
    def select_chosen_and_reject(row):

        # 1. Use reward_0 for choosing chosen/reject responses (for consistency)
        all_rewards_selection = [row[f"reward_0_response_{i}"] for i in range(args.selection_pairs)]
        chosen_idx, reject_idx = np.argmax(all_rewards_selection), np.argmin(all_rewards_selection)

        row["chosen"] = row[f"response_{chosen_idx}"]
        row["reject"] = row[f"response_{reject_idx}"]

        llama_chosen_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{chosen_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+SYS_PROMPT_LEN,
        )[SYS_PROMPT_LEN:]
        llama_chosen_tokens = llama_chosen_token
        llama_chosen = tokenizer.decode(llama_chosen_token, skip_special_tokens=False)
        assert len(llama_chosen_token) == args.maxlen
        assert llama_chosen_token[-1] == 128009 or llama_chosen_token[-1] == 128256
        row["llama_chosen"] = llama_chosen
        row["llama_chosen_tokens"] = llama_chosen_tokens

        llama_reject_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{reject_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+SYS_PROMPT_LEN,
        )[SYS_PROMPT_LEN:]
        llama_reject_tokens = llama_reject_token
        llama_reject = tokenizer.decode(llama_reject_token, skip_special_tokens=False)
        assert len(llama_reject_token) == args.maxlen
        assert llama_reject_token[-1] == 128009 or llama_reject_token[-1] == 128256
        row["llama_reject"] = llama_reject
        row["llama_reject_tokens"] = llama_reject_tokens

        # 2. Collect rewards for chosen/reject from all reward models and compute ranks
        for reward_model_idx in range(args.num_rewards):
            # Collect all rewards for ranking
            all_rewards = []
            for i in range(args.selection_pairs):
                all_rewards.append(row[f"reward_{reward_model_idx}_response_{i}"])
            for i in range(args.gradient_pairs):
                all_rewards.append(row[f"reward_{reward_model_idx}_response_{i}_for_gradient"])
            
            # Compute ranks (0-based, highest reward gets highest rank)
            ranks = np.argsort(np.argsort(all_rewards))  # This gives ranks 0 to (n-1)
            normalized_ranks = ranks / (args.selection_pairs + args.gradient_pairs - 1)  # Normalize to 0-1 range
            
            # Get chosen and reject rewards and ranks
            row[f"chosen_reward_{reward_model_idx}"] = row[f"reward_{reward_model_idx}_response_{chosen_idx}"]
            row[f"reject_reward_{reward_model_idx}"] = row[f"reward_{reward_model_idx}_response_{reject_idx}"]
            row[f"chosen_rank_{reward_model_idx}"] = normalized_ranks[chosen_idx]
            row[f"reject_rank_{reward_model_idx}"] = normalized_ranks[reject_idx]

        # 3. Compute g_chosen, g_reject, g_winrate_chosen, and g_winrate_reject for each reward model
        M = args.gradient_pairs
        n = 2
        tau = 1e3
        constant = n / (M * tau)
        
        for reward_model_idx in range(args.num_rewards):
            # Get gradient rewards and ranks for this reward model
            gradient_rewards = [row[f"reward_{reward_model_idx}_response_{i}_for_gradient"] for i in range(args.gradient_pairs)]
            
            # Get gradient ranks (from the normalized_ranks computed above)
            all_rewards = []
            for i in range(args.selection_pairs):
                all_rewards.append(row[f"reward_{reward_model_idx}_response_{i}"])
            for i in range(args.gradient_pairs):
                all_rewards.append(row[f"reward_{reward_model_idx}_response_{i}_for_gradient"])
            ranks = np.argsort(np.argsort(all_rewards))
            normalized_ranks = ranks / (args.selection_pairs + args.gradient_pairs - 1)
            gradient_ranks = normalized_ranks[args.selection_pairs:args.selection_pairs + args.gradient_pairs]
            
            # Compute g_chosen and g_reject for this reward model (original)
            _chosen_reward = row[f"reward_{reward_model_idx}_response_{chosen_idx}"]
            _reject_reward = row[f"reward_{reward_model_idx}_response_{reject_idx}"]
            
            g_chosen = constant * sum([np.logaddexp(tau*_chosen_reward, tau*r) for r in gradient_rewards])
            g_reject = constant * sum([np.logaddexp(tau*_reject_reward, tau*r) for r in gradient_rewards])
            
            row[f"g_chosen_{reward_model_idx}"] = g_chosen
            row[f"g_reject_{reward_model_idx}"] = g_reject
            
            # Compute g_winrate_chosen and g_winrate_reject using ranks instead of rewards
            _chosen_rank = normalized_ranks[chosen_idx]
            _reject_rank = normalized_ranks[reject_idx]
            
            g_winrate_chosen = constant * sum([np.logaddexp(tau*_chosen_rank, tau*r) for r in gradient_ranks])
            g_winrate_reject = constant * sum([np.logaddexp(tau*_reject_rank, tau*r) for r in gradient_ranks])
            row[f"g_winrate_chosen_{reward_model_idx}"] = g_winrate_chosen
            row[f"g_winrate_reject_{reward_model_idx}"] = g_winrate_reject

        return row

    dataset = dataset.map(select_chosen_and_reject, num_proc=4)

    # filter prompts with exactly same responses
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_tokenized')


if __name__ == "__main__":
    main()