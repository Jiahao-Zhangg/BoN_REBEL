import argparse
import numpy as np
import re
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, DatasetDict
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

    # Load both splits and process similarly to the expand script
    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    if "Qwen" in args.model:
        slicing_idx_used = SYS_PROMPT_LEN
        print(f'slicing index used (fixed): {slicing_idx_used}')
    else:
        slicing_idx_used = args.slicing_idx

    def process_split(dataset):
        print('split length:', len(dataset))

        # Filter overly long prompts
        dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(
            get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
        print('filtered long prompts:', len(dataset))

        # Identify all response columns
        response_pattern = re.compile(r'^(selection|current|base)_response_\d+$')
        response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
        if not response_columns:
            raise ValueError("Dataset is missing response columns required for length filtering.")
        print(f'response columns length-filtered: {response_columns}')

        # Filter by max response length
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

        # Ensure responses end with PAD/EOS
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

        # Ensure prompt columns exist; compute only if missing
        has_qp = "qwen_prompt" in dataset.column_names
        has_qpt = "qwen_prompt_tokens" in dataset.column_names
        if not (has_qp and has_qpt):
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
                    assert ("<|start_header_id|>" in qwen_prompt or "<|im_start|>" in qwen_prompt), "Qwen prompt missing chat header markers"
                qwen_prompts.append(qwen_prompt)
                qwen_prompt_tokens.append(qwen_prompt_token)
            if not has_qp:
                dataset = dataset.add_column("qwen_prompt", qwen_prompts)
            if not has_qpt:
                dataset = dataset.add_column("qwen_prompt_tokens", qwen_prompt_tokens)

        # Compute chosen/reject once per example (no expand)
        chosen, reject, qwen_chosen, qwen_reject, qwen_chosen_tokens, qwen_reject_tokens, chosen_reward, reject_reward = [], [], [], [], [], [], [], []
        g_chosen_list, g_reject_list = [], []
        j_star_list = []

        for row in tqdm(dataset):
            beta = args.beta
            # Build current vs base matrices for two bases and two currents
            current_base_scores = []
            for base_j in range(1, 3):
                rows = []
                for cur_k in range(1, 3):
                    key = f"current_{cur_k}_base_{base_j}_{args.score_type}"
                    p_vec = np.array(row[key], dtype=float)
                    rows.append(p_vec)
                current_base_scores.append(np.stack(rows, axis=0))  # (2, L)

            exp_terms_list = []
            for cb in current_base_scores:
                inner_sums = np.sum(cb, axis=0) / (beta * cb.shape[0])
                exp_terms_list.append(np.exp(-inner_sums))

            A_vec = exp_terms_list[0] + exp_terms_list[1]
            j_star = int(np.argmax(A_vec))
            j_star_list.append(j_star)

            g_values = []
            for sel_i in range(1, 4):
                p_z_vs_bases = []
                for base_j in range(1, 3):
                    key = f"selection_{sel_i}_base_{base_j}_{args.score_type}"
                    p_vec = np.array(row[key], dtype=float)
                    p_z_vs_bases.append(p_vec[j_star])
                B_star = p_z_vs_bases[0] * exp_terms_list[0][j_star] + p_z_vs_bases[1] * exp_terms_list[1][j_star]
                g_z = B_star / A_vec[j_star]
                g_values.append(float(g_z))

            chosen_idx_in_z = int(np.argmax(g_values))
            reject_idx_in_z = int(np.argmin(g_values))

            chosen.append(row[f"selection_response_{chosen_idx_in_z+1}"])
            reject.append(row[f"selection_response_{reject_idx_in_z+1}"])

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
            chosen_reward.append(g_values[chosen_idx_in_z])
            assert len(qwen_chosen_token) == args.maxlen
            if "Qwen" in args.model:
                assert not chosen_text.lstrip().startswith("<|im_start|>assistant"), "Qwen chosen should not include assistant header"
                assert ("<|eot_id|>" in chosen_text) or ("<|im_end|>" in chosen_text), "Qwen chosen text should include end-of-turn marker"
                last_id = int(qwen_chosen_token[-1])
                pid = tokenizer.pad_token_id
                eid = tokenizer.eos_token_id
                assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen chosen last token should be PAD or EOS"

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

        dataset = dataset.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(dataset))
        return dataset

    train_processed = process_split(ds_dict['train'])
    test_processed = process_split(ds_dict['test'])

    out = DatasetDict({"train": train_processed, "test": test_processed})
    out.push_to_hub(args.input_repo + '_beta_' + str(args.beta) + '_multi_noexpand_tokenized')



if __name__ == "__main__":
    main()
