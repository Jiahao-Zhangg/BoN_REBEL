import argparse
import numpy as np
import re
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

torch.set_printoptions(threshold=10_000)

SYS_PROMPT_LEN = 24

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    # Default to MisDrifter/test_dataset which contains selection/current/base score vectors
    parser.add_argument("--input_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_rescale",
                        help="HF dataset repo to load (expects selection/current/base responses and score vectors)")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for A/B/g computation")
    parser.add_argument("--slicing_idx", type=int, default=24)
    parser.add_argument("--score_type", type=str, default="mean", choices=["mean", "majority"],
                        help="Use all mean or all majority score vectors when computing preferences")
    parser.add_argument("--output_repo_prefix", type=str, default=None,
                        help="If set, use this as the repo prefix for push_to_hub instead of input_repo")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of examples for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--limit_rows", type=int, default=0, help="If >0, use only first N rows for debugging")
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


# BTL no longer used; probabilities are provided by dataset


def main():

    # init
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    # Prefer explicit [PAD] token for Qwen so it shows in decoded strings
    if "Qwen" in args.model:
        if tokenizer.pad_token != "[PAD]":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
        if tokenizer_left.pad_token != "[PAD]":
            tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer_left.pad_token = "[PAD]"
    else:
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if tokenizer_left.pad_token is None:
            if tokenizer_left.eos_token is not None:
                tokenizer_left.pad_token = tokenizer_left.eos_token
            else:
                tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})

    dataset = load_dataset(args.input_repo, split='train')
    
    # process dataset
    print('initial length:', len(dataset))

    if args.limit_rows and args.limit_rows > 0:
        n = min(args.limit_rows, len(dataset))
        dataset = dataset.select(range(n))
        print(f'limited to first {n} rows')

    if "Qwen" in args.model:
        slicing_idx_used = SYS_PROMPT_LEN
        print(f'slicing index used (fixed): {slicing_idx_used}')
    else:
        slicing_idx_used = args.slicing_idx

    # Filter overly long prompts
    dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(
        get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    print('filtered long prompts:', len(dataset))

    response_pattern = re.compile(r'^(selection|current|base)_response_\d+$')
    response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
    if not response_columns:
        raise ValueError("Dataset is missing response columns required for length filtering.")
    print(f'response columns length-filtered: {response_columns}')

    def responses_within_limit(row):
        for col in response_columns:
            resp = row[col]
            if not isinstance(resp, str):
                return False
            tokens = tokenizer.apply_chat_template(
                get_message(response=resp),
                tokenize=True,
                add_generation_prompt=False,
                return_tensors='pt',
            )[:, slicing_idx_used:]
            if tokens.shape[-1] > args.maxlen:
                return False
        return True

    dataset = dataset.filter(responses_within_limit)
    print('filtered responses by length:', len(dataset))

    def responses_end_properly(row):
        for col in response_columns:
            resp = row[col]
            if not isinstance(resp, str):
                return False
            response_token = tokenizer.apply_chat_template(
                get_message(response=resp),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen + slicing_idx_used,
            )[slicing_idx_used:]

            if len(response_token) != args.maxlen:
                return False

            pad_id = tokenizer.pad_token_id
            eos_id = tokenizer.eos_token_id
            if not response_token:
                return False
            last_id = int(response_token[-1])
            if pad_id is None and eos_id is None:
                continue
            pad_ok = pad_id is not None and last_id == pad_id
            eos_ok = eos_id is not None and last_id == eos_id
            if not (pad_ok or eos_ok):
                return False
        return True

    dataset = dataset.filter(responses_end_properly)
    print('filtered responses not ending with PAD/EOS:', len(dataset))

    # add prompt tokens
    qwen_prompts = []
    qwen_prompt_tokens = []
    for row in tqdm(dataset):
        qwen_prompt_token = tokenizer_left.apply_chat_template(
                get_message(row['prompt']), 
                add_generation_prompt=True,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen_prompt,
        )
        qwen_prompt = tokenizer_left.decode(qwen_prompt_token, skip_special_tokens=False)
        assert len(qwen_prompt_token) == args.maxlen_prompt
        if "Qwen" in args.model:
            # Expect Qwen chat markers present
            assert ("<|start_header_id|>" in qwen_prompt or "<|im_start|>" in qwen_prompt), "Qwen prompt missing chat header markers"
        qwen_prompts.append(qwen_prompt)
        qwen_prompt_tokens.append(qwen_prompt_token)
    dataset = dataset.add_column("qwen_prompt", qwen_prompts)
    dataset = dataset.add_column("qwen_prompt_tokens", qwen_prompt_tokens)

    # select chosen and reject
    expanded_data = {col: [] for col in dataset.column_names}
    expanded_data.update({
        "chosen": [],
        "chosen_reward": [],
        "qwen_chosen": [],
        "qwen_chosen_tokens": [],
        "reject": [],
        "reject_reward": [],
        "qwen_reject": [],
        "qwen_reject_tokens": [],
        "g_chosen": [],
        "g_reject": [],
        "j_star": [],
    })
    
    for row in tqdm(dataset):
        beta = args.beta
        # We have 2 base responses (y'), 2 current responses (y), and 3 selection responses (z).
        # Columns provide per-check probability vectors P_m(x, y/current or z/selection, y'/base).
        # Build A_m(x) for each check m using current vs base scores, then pick m* = argmax_m A_m(x).

        # Collect P matrices for current vs base for both bases
        # Each list entry is an array of shape (K=2, L) where L = number of checks (vector length)
        current_base_scores = []  # length 2 (base index), each is (K, L)
        for base_j in range(1, 3):
            rows = []
            for cur_k in range(1, 3):
                key = f"current_{cur_k}_base_{base_j}_{args.score_type}"
                p_vec = np.array(row[key], dtype=float)
                rows.append(p_vec)
            current_base_scores.append(np.stack(rows, axis=0))  # (2, L)

        # Compute exp_terms per base_j: exp(- (1/K) * sum_k P_m / beta) for each m
        exp_terms_list = []  # length 2, each is vector (L,)
        for cb in current_base_scores:
            # cb shape: (K=2, L)
            inner_sums = np.sum(cb, axis=0) / (beta * cb.shape[0])  # (L,)
            exp_terms_list.append(np.exp(-inner_sums))

        # A_m(x) = sum_i exp_terms[i][m] over base i in {1,2}
        A_vec = exp_terms_list[0] + exp_terms_list[1]  # (L,)
        j_star = int(np.argmax(A_vec))  # m* index

        # For each selection z, compute B_m*(x,z) using selection vs base probabilities
        # B_m*(x,z) = sum_i P_m(x, z, y'_i) * exp_terms_i[m]
        g_values = []
        for sel_i in range(1, 4):
            p_z_vs_bases = []  # [P_m(x, z, y'_1), P_m(x, z, y'_2)] for chosen m*
            for base_j in range(1, 3):
                key = f"selection_{sel_i}_base_{base_j}_{args.score_type}"
                p_vec = np.array(row[key], dtype=float)
                p_z_vs_bases.append(p_vec[j_star])
            # Compute B and g for this z
            B_star = p_z_vs_bases[0] * exp_terms_list[0][j_star] + p_z_vs_bases[1] * exp_terms_list[1][j_star]
            g_z = B_star / A_vec[j_star]
            g_values.append(float(g_z))

        selection_tokens = []
        selection_texts = []
        for sel_i in range(1, 4):
            sel_token = tokenizer.apply_chat_template(
                    get_message(response=row[f"selection_response_{sel_i}"]),
                    add_generation_prompt=False,
                    tokenize=True,
                    padding='max_length',
                    max_length=args.maxlen+slicing_idx_used,
            )[slicing_idx_used:]
            selection_tokens.append(list(sel_token))
            sel_text = tokenizer.decode(sel_token, skip_special_tokens=False)
            selection_texts.append(sel_text)
            assert len(sel_token) == args.maxlen
            if "Qwen" in args.model:
                assert not sel_text.lstrip().startswith("<|im_start|>assistant"), "Qwen selection should not include assistant header"
                assert ("<|eot_id|>" in sel_text) or ("<|im_end|>" in sel_text), "Qwen selection text should include end-of-turn marker"
                last_id = int(sel_token[-1])
                pid = tokenizer.pad_token_id
                eid = tokenizer.eos_token_id
                assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen selection last token should be PAD or EOS"

        for pair in ((0, 1), (1, 2), (0, 2)):
            higher_idx, lower_idx = pair
            if g_values[higher_idx] < g_values[lower_idx]:
                higher_idx, lower_idx = lower_idx, higher_idx

            for col in dataset.column_names:
                expanded_data[col].append(row[col])

            expanded_data["chosen"].append(row[f"selection_response_{higher_idx+1}"])
            expanded_data["reject"].append(row[f"selection_response_{lower_idx+1}"])

            expanded_data["qwen_chosen_tokens"].append(selection_tokens[higher_idx])
            expanded_data["qwen_reject_tokens"].append(selection_tokens[lower_idx])

            expanded_data["qwen_chosen"].append(selection_texts[higher_idx])
            expanded_data["qwen_reject"].append(selection_texts[lower_idx])

            expanded_data["chosen_reward"].append(float(g_values[higher_idx]))
            expanded_data["reject_reward"].append(float(g_values[lower_idx]))

            expanded_data["g_chosen"].append(float(g_values[higher_idx]))
            expanded_data["g_reject"].append(float(g_values[lower_idx]))

            expanded_data["j_star"].append(j_star)

    dataset = Dataset.from_dict(expanded_data)

    # filter prompts with exactly same responses
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=args.test_size, shuffle=True, seed=args.seed)
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    dataset.push_to_hub(repo_prefix + '_' + args.score_type +'_beta_'+str(args.beta) + '_multi_expand_tokenized')



if __name__ == "__main__":
    main()
