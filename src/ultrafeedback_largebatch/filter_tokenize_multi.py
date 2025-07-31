import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from rank.py")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--selection_pairs", type=int, default=5, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=5, help="number of pairs to use for gradient estimation")
    parser.add_argument("--num_reward_models", type=int, required=True, help="number of reward models used in rank_multi.py")
    return parser.parse_args()


def get_message(instruction=None, response=None):

    assert instruction != None or response != None

    if response == None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction == None:
        message = [
            {"role": "assistant", "content": response}
        ]
    else:
        message = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

    return message


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
    for i in range(args.selection_pairs): # magic number 30. Pay attention.
        dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(response=row[f'response_{i}']), tokenize=True, add_generation_prompt=False, return_tensors='pt')[:, 30:].shape[-1] <= args.maxlen)
        print(f'filtered response_{i}:', len(dataset))

    # add prompt tokens
    llama_prompts = []
    llama_prompt_tokens = []
    for row in tqdm(dataset):
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
        llama_prompts.append(llama_prompt)
        llama_prompt_tokens.append(llama_prompt_token)
    dataset = dataset.add_column("llama_prompt", llama_prompts)
    dataset = dataset.add_column("llama_prompt_tokens", llama_prompt_tokens)

    # select chosen and reject
    chosen, reject, llama_chosen, llama_reject, llama_chosen_tokens, llama_reject_tokens = [], [], [], [], [], []
    chosen_reward_lists = [[] for _ in range(args.num_reward_models)]  # Each reward model gets its own list
    reject_reward_lists = [[] for _ in range(args.num_reward_models)]
    g_chosen_lists = [[] for _ in range(args.num_reward_models)]
    g_reject_lists = [[] for _ in range(args.num_reward_models)]
    chosen_rank_lists = [[] for _ in range(args.num_reward_models)]  # New: chosen response ranks
    reject_rank_lists = [[] for _ in range(args.num_reward_models)]   # New: reject response ranks
    g_winrate_chosen_lists = [[] for _ in range(args.num_reward_models)]  # New: g_winrate_chosen
    g_winrate_reject_lists = [[] for _ in range(args.num_reward_models)]  # New: g_winrate_reject
    
    for row in tqdm(dataset):

        # 1. Use reward_1 for choosing chosen/reject responses (for consistency)
        all_rewards_selection = [row[f"reward_1_response_{i}"] for i in range(args.selection_pairs)]
        chosen_idx, reject_idx = np.argmax(all_rewards_selection), np.argmin(all_rewards_selection)

        chosen.append(row[f"response_{chosen_idx}"])
        reject.append(row[f"response_{reject_idx}"])

        llama_chosen_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{chosen_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+30,
        )[30:]
        llama_chosen_tokens.append(llama_chosen_token)
        llama_chosen.append(tokenizer.decode(llama_chosen_token, skip_special_tokens=False))
        assert len(llama_chosen_token) == args.maxlen
        assert llama_chosen_token[-1] == 128009 or llama_chosen_token[-1] == 128256

        llama_reject_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{reject_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+30,
        )[30:]
        llama_reject_tokens.append(llama_reject_token)
        llama_reject.append(tokenizer.decode(llama_reject_token, skip_special_tokens=False))
        assert len(llama_reject_token) == args.maxlen
        assert llama_reject_token[-1] == 128009 or llama_reject_token[-1] == 128256

        # 2. Collect rewards for chosen/reject from all reward models and compute ranks
        for reward_model_idx in range(args.num_reward_models):
            # Collect all 10 rewards for ranking (5 selection + 5 gradient)
            all_10_rewards = []
            for i in range(args.selection_pairs):
                all_10_rewards.append(row[f"reward_{reward_model_idx}_response_{i}"])
            for i in range(args.selection_pairs, args.selection_pairs + args.gradient_pairs):
                all_10_rewards.append(row[f"reward_{reward_model_idx}_response_{i}_for_gradient"])
            
            # Compute ranks (0-based, highest reward gets highest rank)
            ranks = np.argsort(np.argsort(all_10_rewards))  # This gives ranks 0-9
            normalized_ranks = ranks / 9.0  # Normalize to 0-1 range
            
            # Get chosen and reject rewards and ranks
            chosen_reward_lists[reward_model_idx].append(row[f"reward_{reward_model_idx}_response_{chosen_idx}"])
            reject_reward_lists[reward_model_idx].append(row[f"reward_{reward_model_idx}_response_{reject_idx}"])
            chosen_rank_lists[reward_model_idx].append(normalized_ranks[chosen_idx])
            reject_rank_lists[reward_model_idx].append(normalized_ranks[reject_idx])

        # 3. Compute g_chosen, g_reject, g_winrate_chosen, and g_winrate_reject for each reward model
        M = args.gradient_pairs
        n = 2
        tau = 1e3
        constant = n / (M * tau)
        
        for reward_model_idx in range(args.num_reward_models):
            # Get gradient rewards and ranks for this reward model
            gradient_rewards = [row[f"reward_{reward_model_idx}_response_{i}_for_gradient"] for i in range(args.selection_pairs, args.selection_pairs + args.gradient_pairs)]
            
            # Get gradient ranks (from the normalized_ranks computed above)
            all_10_rewards = []
            for i in range(args.selection_pairs):
                all_10_rewards.append(row[f"reward_{reward_model_idx}_response_{i}"])
            for i in range(args.selection_pairs, args.selection_pairs + args.gradient_pairs):
                all_10_rewards.append(row[f"reward_{reward_model_idx}_response_{i}_for_gradient"])
            ranks = np.argsort(np.argsort(all_10_rewards))
            normalized_ranks = ranks / 9.0
            gradient_ranks = normalized_ranks[args.selection_pairs:args.selection_pairs + args.gradient_pairs]
            
            # Compute g_chosen and g_reject for this reward model (original)
            _chosen_reward = row[f"reward_{reward_model_idx}_response_{chosen_idx}"]
            _reject_reward = row[f"reward_{reward_model_idx}_response_{reject_idx}"]
            
            g_chosen = constant * sum([np.log((np.exp(tau*_chosen_reward) + np.exp(tau*r))) for r in gradient_rewards])
            g_reject = constant * sum([np.log((np.exp(tau*_reject_reward) + np.exp(tau*r))) for r in gradient_rewards])
            
            g_chosen_lists[reward_model_idx].append(g_chosen)
            g_reject_lists[reward_model_idx].append(g_reject)
            
            # Compute g_winrate_chosen and g_winrate_reject using ranks instead of rewards
            _chosen_rank = normalized_ranks[chosen_idx]
            _reject_rank = normalized_ranks[reject_idx]
            
            tau_ = 100
            g_winrate_chosen = constant * sum([np.log(np.exp(tau_*_chosen_rank) + np.exp(tau_*r)) for r in gradient_ranks])
            g_winrate_reject = constant * sum([np.log(np.exp(tau_*_reject_rank) + np.exp(tau_*r)) for r in gradient_ranks])
            g_winrate_chosen_lists[reward_model_idx].append(g_winrate_chosen)
            g_winrate_reject_lists[reward_model_idx].append(g_winrate_reject)

    dataset = dataset.add_column("chosen", chosen)
    dataset = dataset.add_column("llama_chosen", llama_chosen)
    dataset = dataset.add_column("llama_chosen_tokens", llama_chosen_tokens)
    dataset = dataset.add_column("reject", reject)
    dataset = dataset.add_column("llama_reject", llama_reject)
    dataset = dataset.add_column("llama_reject_tokens", llama_reject_tokens)
    
    # Add separate columns for each reward model
    for reward_model_idx in range(args.num_reward_models):
        dataset = dataset.add_column(f"chosen_reward_{reward_model_idx}", chosen_reward_lists[reward_model_idx])
        dataset = dataset.add_column(f"reject_reward_{reward_model_idx}", reject_reward_lists[reward_model_idx])
        dataset = dataset.add_column(f"g_chosen_{reward_model_idx}", g_chosen_lists[reward_model_idx])
        dataset = dataset.add_column(f"g_reject_{reward_model_idx}", g_reject_lists[reward_model_idx])
        dataset = dataset.add_column(f"chosen_rank_{reward_model_idx}", chosen_rank_lists[reward_model_idx])
        dataset = dataset.add_column(f"reject_rank_{reward_model_idx}", reject_rank_lists[reward_model_idx])
        dataset = dataset.add_column(f"g_winrate_chosen_{reward_model_idx}", g_winrate_chosen_lists[reward_model_idx])
        dataset = dataset.add_column(f"g_winrate_reject_{reward_model_idx}", g_winrate_reject_lists[reward_model_idx])

    # filter prompts with exactly same responses
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_tokenized')


if __name__ == "__main__":
    main()