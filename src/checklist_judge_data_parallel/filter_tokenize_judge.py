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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    # Default to MisDrifter/test_dataset which contains selection/current/base score vectors
    parser.add_argument("--input_repo", type=str, default="MisDrifter/test_dataset",
                        help="HF dataset repo to load (expects selection/current/base responses and score vectors)")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=8192)
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for A/B/g computation")
    parser.add_argument("--slicing_idx", type=int, default=30)
    parser.add_argument("--score_type", type=str, default="mean", choices=["mean", "majority"],
                        help="Use all mean or all majority score vectors when computing preferences")
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

    def compute_qwen_slicing_idx(tok):
        """Find the start-of-content offset by aligning raw content tokens inside the templated assistant message."""
        sample = "__QWEN_SLICE_PROBE__"
        full = tok.apply_chat_template(get_message(response=sample), tokenize=True, add_generation_prompt=False)
        if isinstance(full, dict):
            full_ids = full["input_ids"]
        else:
            full_ids = full
        content_ids = tok(sample, add_special_tokens=False)["input_ids"]

        # Simple subsequence search
        def find_subseq(haystack, needle):
            n, m = len(haystack), len(needle)
            for i in range(0, n - m + 1):
                if haystack[i:i+m] == needle:
                    return i
            return -1

        pos = find_subseq(full_ids, content_ids)
        if pos < 0:
            raise RuntimeError("Failed to locate content within Qwen assistant template; cannot compute slicing index.")
        return pos

    dataset = load_dataset(args.input_repo, split='train')
    
    # process dataset
    print('initial length:', len(dataset))

    # filter dataset with long prompt or response (only selection responses are tokenized/used downstream)
    dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    print('filtered long prompts:', len(dataset))
    for i in range(1, 4):
        key = f'selection_response_{i}'
        dataset = dataset.filter(lambda row, _key=key: tokenizer.apply_chat_template(get_message(response=row[_key]), tokenize=True, add_generation_prompt=False, return_tensors='pt')[:, 5:].shape[-1] <= args.maxlen)
        print(f'filtered {key}:', len(dataset))

    # add prompt tokens
    qwen_prompts = []
    qwen_prompt_tokens = []
    # Compute slicing index for Qwen if applicable
    if "Qwen" in args.model:
        slicing_idx_used = compute_qwen_slicing_idx(tokenizer)
        print(f'slicing index used: {slicing_idx_used}')
    else:
        slicing_idx_used = args.slicing_idx
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
    chosen, reject, qwen_chosen, qwen_reject, qwen_chosen_tokens, qwen_reject_tokens, chosen_reward, reject_reward = [], [], [], [], [], [], [], []
    g_chosen_list, g_reject_list = [], []
    j_star_list = []  # here j_star indexes the m* (check) maximizing A
    
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
        j_star_list.append(j_star)

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

        # Choose z that maximizes/minimizes g
        chosen_idx_in_z = int(np.argmax(g_values))  # 0..2
        reject_idx_in_z = int(np.argmin(g_values))  # 0..2

        chosen.append(row[f"selection_response_{chosen_idx_in_z+1}"])
        reject.append(row[f"selection_response_{reject_idx_in_z+1}"])
        
        # Process chosen response
        qwen_chosen_token = tokenizer.apply_chat_template(
                get_message(response=row[f"selection_response_{chosen_idx_in_z+1}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+slicing_idx_used,
        )[slicing_idx_used:]
        qwen_chosen_tokens.append(qwen_chosen_token)
        chosen_text = tokenizer.decode(qwen_chosen_token, skip_special_tokens=False)
        qwen_chosen.append(chosen_text)
        # Store g-value as the chosen 'reward' for traceability
        chosen_reward.append(g_values[chosen_idx_in_z])
        assert len(qwen_chosen_token) == args.maxlen
        if "Qwen" in args.model:
            # After slicing, ensure no assistant header at start and end-of-turn marker present
            assert not chosen_text.lstrip().startswith("<|im_start|>assistant"), "Qwen chosen should not include assistant header"
            assert ("<|eot_id|>" in chosen_text) or ("<|im_end|>" in chosen_text), "Qwen chosen text should include end-of-turn marker"
            last_id = int(qwen_chosen_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen chosen last token should be PAD or EOS"
        
        # Process rejected response
        qwen_reject_token = tokenizer.apply_chat_template(
                get_message(response=row[f"selection_response_{reject_idx_in_z+1}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+slicing_idx_used,
        )[slicing_idx_used:]
        qwen_reject_tokens.append(qwen_reject_token)
        reject_text = tokenizer.decode(qwen_reject_token, skip_special_tokens=False)
        qwen_reject.append(reject_text)
        reject_reward.append(g_values[reject_idx_in_z])
        assert len(qwen_reject_token) == args.maxlen
        if "Qwen" in args.model:
            assert not reject_text.lstrip().startswith("<|im_start|>assistant"), "Qwen reject should not include assistant header"
            assert ("<|eot_id|>" in reject_text) or ("<|im_end|>" in reject_text), "Qwen reject text should include end-of-turn marker"
            last_id = int(qwen_reject_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen reject last token should be PAD or EOS"
        
        # Store g(x,z) values
        g_chosen_list.append(g_values[chosen_idx_in_z])
        g_reject_list.append(g_values[reject_idx_in_z])

    dataset = dataset.add_column("chosen", chosen)
    dataset = dataset.add_column("chosen_reward", chosen_reward)
    dataset = dataset.add_column("qwen_chosen", qwen_chosen)
    dataset = dataset.add_column("qwen_chosen_tokens", qwen_chosen_tokens)
    dataset = dataset.add_column("reject", reject)
    dataset = dataset.add_column("reject_reward", reject_reward)
    dataset = dataset.add_column("qwen_reject", qwen_reject)
    dataset = dataset.add_column("qwen_reject_tokens", qwen_reject_tokens)
    dataset = dataset.add_column("g_chosen", g_chosen_list)
    dataset = dataset.add_column("g_reject", g_reject_list)
    dataset = dataset.add_column("j_star", j_star_list)

    # filter prompts with exactly same responses
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    model_name = args.model.split('/')[-1]
    dataset.push_to_hub('zjhhhh/gangdu_tokenized')


if __name__ == "__main__":
    main()
