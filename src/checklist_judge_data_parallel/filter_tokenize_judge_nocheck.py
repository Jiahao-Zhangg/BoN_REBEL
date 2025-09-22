import argparse
import numpy as np
import re
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

torch.set_printoptions(threshold=10_000)


# WARNING: Magic number, make sure it works for your model
SYS_PROMPT_LEN = 24


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter + tokenize using scalar judge scores (no fixed check).",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_nocheck_rescale",
                        help="HF dataset repo to load (expects selection/current/base responses and scalar scores + requirements)")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for A/B/g computation")
    parser.add_argument("--slicing_idx", type=int, default=24,
                        help="Fallback slicing index if model-specific detection not used")
    parser.add_argument("--score_type", type=str, default="mean", choices=["mean", "majority"],
                        help="Use all mean or all majority scores when computing preferences")
    return parser.parse_args()


def get_message(instruction=None, response=None):
    assert instruction is not None or response is not None

    if response is None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction is None:
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
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')

    # Ensure PAD handling sensible
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

    if "Qwen" in args.model:
        slicing_idx_used = SYS_PROMPT_LEN
    else:
        slicing_idx_used = args.slicing_idx

    dataset = load_dataset(args.input_repo, split='train')
    print('initial length:', len(dataset))

    # Filter overly long prompts
    dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(
        get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    print('filtered long prompts:', len(dataset))

    # Filter responses by length across all selection/current/base response columns
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
            )[:, SYS_PROMPT_LEN:]
            if tokens.shape[-1] > args.maxlen:
                return False
        return True

    dataset = dataset.filter(responses_within_limit)
    print('filtered responses by length:', len(dataset))

    # filter responses that don't end with PAD or EOS token after tokenization
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
                max_length=args.maxlen + SYS_PROMPT_LEN,
            )[SYS_PROMPT_LEN:]

            if "Qwen" in args.model:
                last_id = int(response_token[-1])
                pid = tokenizer.pad_token_id
                eid = tokenizer.eos_token_id
                if not ((pid is not None and last_id == pid) or (eid is not None and last_id == eid)):
                    return False
        return True

    dataset = dataset.filter(responses_end_properly)
    print('filtered responses not ending with PAD/EOS:', len(dataset))

    # Add prompt tokens
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
    dataset = dataset.add_column("qwen_prompt", qwen_prompts)
    dataset = dataset.add_column("qwen_prompt_tokens", qwen_prompt_tokens)

    # select chosen and reject using scalar judge scores
    chosen, reject = [], []
    qwen_chosen, qwen_reject = [], []
    qwen_chosen_tokens, qwen_reject_tokens = [], []
    chosen_reward, reject_reward = [], []
    g_chosen_list, g_reject_list = [], []

    for row in tqdm(dataset):
        beta = args.beta

        # Collect current vs base scalar scores per base model
        current_base_scores = []
        for base_j in range(1, 3):
            rows = []
            for cur_k in range(1, 3):
                key = f"current_{cur_k}_base_{base_j}_{args.score_type}"
                score = float(row[key])
                rows.append(score)
            current_base_scores.append(np.array(rows, dtype=float))

        # Compute exp_terms using the scalar scores
        exp_terms = []  # length 2
        for cb in current_base_scores:
            inner = np.sum(cb) / (beta * cb.size)
            exp_terms.append(np.exp(-inner))

        # Denominator A for all scores
        A_value = exp_terms[0] + exp_terms[1]

        # For each selection z, compute B and g
        g_values = []
        for sel_i in range(1, 4):
            p_z_vs_bases = []
            for base_j in range(1, 3):
                key = f"selection_{sel_i}_base_{base_j}_{args.score_type}"
                score = float(row[key])
                p_z_vs_bases.append(score)
            B_value = p_z_vs_bases[0] * exp_terms[0] + p_z_vs_bases[1] * exp_terms[1]
            g_z = B_value / A_value
            g_values.append(float(g_z))

        chosen_idx_in_z = int(np.argmax(g_values))  # 0..2
        reject_idx_in_z = int(np.argmin(g_values))  # 0..2

        # Raw strings
        chosen.append(row[f"selection_response_{chosen_idx_in_z+1}"])
        reject.append(row[f"selection_response_{reject_idx_in_z+1}"])

        # Tokenize chosen
        qwen_chosen_token = tokenizer.apply_chat_template(
            get_message(response=row[f"selection_response_{chosen_idx_in_z+1}"]),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen + slicing_idx_used,
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

        # Tokenize reject
        qwen_reject_token = tokenizer.apply_chat_template(
            get_message(response=row[f"selection_response_{reject_idx_in_z+1}"]),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen + slicing_idx_used,
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

        # Store g-values
        g_chosen_list.append(g_values[chosen_idx_in_z])
        g_reject_list.append(g_values[reject_idx_in_z])

    # Attach columns
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

    # Remove rows where chosen == reject
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    # Split and push (keep naming consistent)
    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_' + args.score_type + '_maxlenp_' + str(args.maxlen_prompt)+'_beta_'+str(args.beta) + '_nocheck_tokenized')


if __name__ == "__main__":
    main()
