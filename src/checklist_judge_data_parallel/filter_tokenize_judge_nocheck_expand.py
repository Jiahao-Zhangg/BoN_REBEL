import argparse
import numpy as np
import re
import torch
from datasets import load_dataset, Dataset
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
    })

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

        selection_tokens = []
        selection_texts = []
        for sel_i in range(1, 4):
            sel_token = tokenizer.apply_chat_template(
                get_message(response=row[f"selection_response_{sel_i}"]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen + slicing_idx_used,
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

    dataset = Dataset.from_dict(expanded_data)

    # Remove rows where chosen == reject
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    # Split and push (keep naming consistent)
    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_' + args.score_type + '_maxlenp_' + str(args.maxlen_prompt)+'_beta_'+str(args.beta) + '_nocheck_tokenized')


if __name__ == "__main__":
    main()
