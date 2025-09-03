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
    parser.add_argument("--total_responses", type=int, default=20, help="total number of responses per prompt")
    parser.add_argument("--num_reward_models", type=int, default=2, help="number of reward models")
    parser.add_argument("--M", type=int, default=8, help="number of responses for y' and y_k each")
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for BTL model")
    parser.add_argument("--slicing_idx", type=int, default=30)
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


def btl_probability(reward_y, reward_y_prime):
    """Compute BTL probability: P_j(x,y,y') = exp(r_j(x,y)) / (exp(r_j(x,y)) + exp(r_j(x,y')))"""
    exp_y = np.exp(reward_y)
    exp_y_prime = np.exp(reward_y_prime)
    return exp_y / (exp_y + exp_y_prime)


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
    for i in range(args.total_responses):
        dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(response=row[f'response_{i}']), tokenize=True, add_generation_prompt=False, return_tensors='pt')[:, 5:].shape[-1] <= args.maxlen)
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
    chosen, reject, llama_chosen, llama_reject, llama_chosen_tokens, llama_reject_tokens, chosen_reward, reject_reward = [], [], [], [], [], [], [], []
    g_chosen_list, g_reject_list = [], []
    j_star_list = []
    
    for row in tqdm(dataset):
        M = args.M
        beta = args.beta
        
        # Split responses: y'_i (0-7), y_k (8-15), z (16-19)
        y_prime_indices = np.arange(0, M)  # 0-7
        y_k_indices = np.arange(M, 2*M)    # 8-15
        z_indices = np.arange(2*M, args.total_responses)  # 16-19
        
        # Step 1: Compute A_j(x) for each reward model j using numpy
        A_j = []
        cached_computations = {}  # Cache computations for reuse
        
        for j in range(args.num_reward_models):
            # Get all rewards for this reward model
            y_prime_rewards = np.array([row[f"reward_{j}_response_{i}"] for i in y_prime_indices])  # shape: (8,)
            y_k_rewards = np.array([row[f"reward_{j}_response_{k}"] for k in y_k_indices])  # shape: (8,)
            
            # Compute all P_j(x, y_k, y'_i) at once
            # y_k_rewards[:, None] broadcasts to (8, 1), y_prime_rewards broadcasts to (8,)
            # Result is (8, 8) matrix where [k, i] = P_j(x, y_k, y'_i)
            exp_y_k = np.exp(y_k_rewards[:, None])  # (8, 1)
            exp_y_prime = np.exp(y_prime_rewards)   # (8,)
            p_matrix = exp_y_k / (exp_y_k + exp_y_prime)  # (8, 8)
            
            # Sum over k for each i, then compute exponential and sum over i
            inner_sums = np.sum(p_matrix / beta, axis=0) / M  # shape: (8,)
            exp_terms = np.exp(-inner_sums)  # shape: (8,)
            A_j_val = np.sum(exp_terms)
            A_j.append(A_j_val)
            
            # Cache computations for potential reuse in B_j* calculation
            cached_computations[j] = {
                'y_prime_rewards': y_prime_rewards,
                'exp_y_prime': exp_y_prime,
                'exp_terms': exp_terms
            }
        
        # Find j* = argmax A_j(x)
        j_star = np.argmax(A_j)
        j_star_list.append(j_star)
        
        # Step 2: Use argmax/argmin of rewards for z responses selection, but still compute g(x,z) for storage
        z_rewards = np.array([row[f"reward_{j_star}_response_{z_idx}"] for z_idx in z_indices])
        
        # Find chosen (argmax reward) and rejected (argmin reward)
        chosen_idx_in_z = np.argmax(z_rewards)
        reject_idx_in_z = np.argmin(z_rewards)
        
        chosen_idx = z_indices[chosen_idx_in_z]
        reject_idx = z_indices[reject_idx_in_z]
        
        # Compute g(x,z) values for storage using cached computations from A_j* calculation
        cached_j_star = cached_computations[j_star]
        exp_y_prime_j_star = cached_j_star['exp_y_prime']  # shape: (8,)
        exp_terms = cached_j_star['exp_terms']  # shape: (8,)
        A_j_star = A_j[j_star]  # Use the already computed A_j* value
        
        g_values = []
        for z_idx in z_indices:
            # Compute B_j*(x,z) using vectorized operations and cached values
            z_reward = row[f"reward_{j_star}_response_{z_idx}"]
            exp_z = np.exp(z_reward)
            p_z_y_prime = exp_z / (exp_z + exp_y_prime_j_star)  # shape: (8,)
            B_j_star = np.sum(p_z_y_prime * exp_terms)
            
            # Compute g(x,z) = B_j*(x,z) / A_j*(x) using pre-computed A_j*
            g_z = B_j_star / A_j_star
            g_values.append(g_z)
        
        chosen.append(row[f"response_{chosen_idx}"])
        reject.append(row[f"response_{reject_idx}"])
        
        # Process chosen response
        llama_chosen_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{chosen_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+args.slicing_idx,
        )[args.slicing_idx:]
        llama_chosen_tokens.append(llama_chosen_token)
        llama_chosen.append(tokenizer.decode(llama_chosen_token, skip_special_tokens=False))
        _chosen_reward = row[f"reward_{j_star}_response_{chosen_idx}"]
        chosen_reward.append(_chosen_reward)
        assert len(llama_chosen_token) == args.maxlen
        assert llama_chosen_token[-1] == 128009 or llama_chosen_token[-1] == 128256
        
        # Process rejected response
        llama_reject_token = tokenizer.apply_chat_template(
                get_message(response=row[f"response_{reject_idx}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+args.slicing_idx,
        )[args.slicing_idx:]
        llama_reject_tokens.append(llama_reject_token)
        llama_reject.append(tokenizer.decode(llama_reject_token, skip_special_tokens=False))
        _reject_reward = row[f"reward_{j_star}_response_{reject_idx}"]
        reject_reward.append(_reject_reward)
        assert len(llama_reject_token) == args.maxlen
        assert llama_reject_token[-1] == 128009 or llama_reject_token[-1] == 128256
        
        # Store g(x,z) values
        g_chosen_list.append(g_values[chosen_idx_in_z])
        g_reject_list.append(g_values[reject_idx_in_z])

    dataset = dataset.add_column("chosen", chosen)
    dataset = dataset.add_column("chosen_reward", chosen_reward)
    dataset = dataset.add_column("llama_chosen", llama_chosen)
    dataset = dataset.add_column("llama_chosen_tokens", llama_chosen_tokens)
    dataset = dataset.add_column("reject", reject)
    dataset = dataset.add_column("reject_reward", reject_reward)
    dataset = dataset.add_column("llama_reject", llama_reject)
    dataset = dataset.add_column("llama_reject_tokens", llama_reject_tokens)
    dataset = dataset.add_column("g_chosen", g_chosen_list)
    dataset = dataset.add_column("g_reject", g_reject_list)
    dataset = dataset.add_column("j_star", j_star_list)

    # filter prompts with exactly same responses
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    model_name = args.model.split('/')[-1]
    dataset.push_to_hub(args.input_repo + '_tokenized_multi_preference')


if __name__ == "__main__":
    main()