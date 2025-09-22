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
        description="Filter + tokenize while pooling current and selection responses, selecting the best check dynamically.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_rescale",
                        help="HF dataset repo to load (expects selection/current/base responses and score vectors)")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for A/B/g computation")
    parser.add_argument("--slicing_idx", type=int, default=24,
                        help="Fallback slicing index if model-specific detection not used")
    parser.add_argument("--score_type", type=str, default="mean", choices=["mean", "majority"],
                        help="Use all mean or all majority score vectors when computing preferences")
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

    if "Qwen" in args.model:
        if tokenizer.pad_token != "[PAD]":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.pad_token = "[PAD]"
        if tokenizer_left.pad_token != "[PAD]":
            tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer_left.pad_token = "[PAD]"
        slicing_idx_used = SYS_PROMPT_LEN
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
        slicing_idx_used = args.slicing_idx

    dataset = load_dataset(args.input_repo, split='train')
    print('initial length:', len(dataset))

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
            )[:, SYS_PROMPT_LEN:]
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

    escaped_score_type = re.escape(args.score_type)
    selection_score_pattern = re.compile(rf"^selection_(\d+)_base_(\d+)_({escaped_score_type})$")
    current_score_pattern = re.compile(rf"^current_(\d+)_base_(\d+)_({escaped_score_type})$")
    selection_ids = set()
    current_ids = set()
    base_ids = set()
    for name in dataset.column_names:
        sel_match = selection_score_pattern.match(name)
        if sel_match:
            selection_ids.add(int(sel_match.group(1)))
            base_ids.add(int(sel_match.group(2)))
        cur_match = current_score_pattern.match(name)
        if cur_match:
            current_ids.add(int(cur_match.group(1)))
            base_ids.add(int(cur_match.group(2)))

    selection_ids = sorted(selection_ids)
    current_ids = sorted(current_ids)
    base_ids = sorted(base_ids)
    if not selection_ids:
        raise ValueError("Dataset is missing selection-base score columns for the specified score type.")
    if not current_ids:
        raise ValueError("Dataset is missing current-base score columns for the specified score type.")
    if not base_ids:
        raise ValueError("Dataset is missing base indices for the specified score type.")

    chosen, reject = [], []
    qwen_chosen, qwen_reject = [], []
    qwen_chosen_tokens, qwen_reject_tokens = [], []
    chosen_reward, reject_reward = [], []
    chosen_origin, reject_origin = [], []
    g_chosen_list, g_reject_list = [], []
    j_star_list = []

    for row in tqdm(dataset):
        beta = args.beta

        scores_by_candidate = {}
        for cur_id in current_ids:
            scores_by_candidate[("current", cur_id)] = {}
        for sel_id in selection_ids:
            scores_by_candidate[("selection", sel_id)] = {}

        exp_terms = {}
        for base_id in base_ids:
            per_base_vectors = []
            for cur_id in current_ids:
                key = f"current_{cur_id}_base_{base_id}_{args.score_type}"
                p_vec = np.array(row[key], dtype=float)
                scores_by_candidate[("current", cur_id)][base_id] = p_vec
                per_base_vectors.append(p_vec)
            for sel_id in selection_ids:
                key = f"selection_{sel_id}_base_{base_id}_{args.score_type}"
                p_vec = np.array(row[key], dtype=float)
                scores_by_candidate[("selection", sel_id)][base_id] = p_vec
                per_base_vectors.append(p_vec)
            stacked = np.stack(per_base_vectors, axis=0)
            inner = np.sum(stacked, axis=0) / (beta * stacked.shape[0])
            exp_terms[base_id] = np.exp(-inner)

        A_vec = None
        for base_id in base_ids:
            if A_vec is None:
                A_vec = exp_terms[base_id]
            else:
                A_vec = A_vec + exp_terms[base_id]
        j_star = int(np.argmax(A_vec))
        A_star = float(A_vec[j_star])
        j_star_list.append(j_star)

        candidates = []
        for cur_id in current_ids:
            weighted_sum = 0.0
            for base_id in base_ids:
                weighted_sum += scores_by_candidate[("current", cur_id)][base_id][j_star] * exp_terms[base_id][j_star]
            g_val = float(weighted_sum / A_star)
            candidates.append({
                "g": g_val,
                "text": row[f"current_response_{cur_id}"],
                "origin": "current",
                "id": cur_id,
            })

        for sel_id in selection_ids:
            weighted_sum = 0.0
            for base_id in base_ids:
                weighted_sum += scores_by_candidate[("selection", sel_id)][base_id][j_star] * exp_terms[base_id][j_star]
            g_val = float(weighted_sum / A_star)
            candidates.append({
                "g": g_val,
                "text": row[f"selection_response_{sel_id}"],
                "origin": "selection",
                "id": sel_id,
            })

        g_values = [cand["g"] for cand in candidates]
        chosen_idx = int(np.argmax(g_values))
        reject_idx = int(np.argmin(g_values))

        chosen_entry = candidates[chosen_idx]
        reject_entry = candidates[reject_idx]

        chosen.append(chosen_entry["text"])
        reject.append(reject_entry["text"])
        chosen_origin.append(chosen_entry["origin"])
        reject_origin.append(reject_entry["origin"])

        qwen_chosen_token = tokenizer.apply_chat_template(
            get_message(response=chosen_entry["text"]),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen + slicing_idx_used,
        )[slicing_idx_used:]
        qwen_chosen_tokens.append(qwen_chosen_token)
        chosen_text = tokenizer.decode(qwen_chosen_token, skip_special_tokens=False)
        qwen_chosen.append(chosen_text)
        chosen_reward.append(chosen_entry["g"])
        assert len(qwen_chosen_token) == args.maxlen
        if "Qwen" in args.model:
            assert not chosen_text.lstrip().startswith("<|im_start|>assistant"), "Qwen chosen should not include assistant header"
            assert ("<|eot_id|>" in chosen_text) or ("<|im_end|>" in chosen_text), "Qwen chosen text should include end-of-turn marker"
            last_id = int(qwen_chosen_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen chosen last token should be PAD or EOS"

        qwen_reject_token = tokenizer.apply_chat_template(
            get_message(response=reject_entry["text"]),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen + slicing_idx_used,
        )[slicing_idx_used:]
        qwen_reject_tokens.append(qwen_reject_token)
        reject_text = tokenizer.decode(qwen_reject_token, skip_special_tokens=False)
        qwen_reject.append(reject_text)
        reject_reward.append(reject_entry["g"])
        assert len(qwen_reject_token) == args.maxlen
        if "Qwen" in args.model:
            assert not reject_text.lstrip().startswith("<|im_start|>assistant"), "Qwen reject should not include assistant header"
            assert ("<|eot_id|>" in reject_text) or ("<|im_end|>" in reject_text), "Qwen reject text should include end-of-turn marker"
            last_id = int(qwen_reject_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen reject last token should be PAD or EOS"

        g_chosen_list.append(chosen_entry["g"])
        g_reject_list.append(reject_entry["g"])

    dataset = dataset.add_column("chosen", chosen)
    dataset = dataset.add_column("chosen_reward", chosen_reward)
    dataset = dataset.add_column("chosen_origin", chosen_origin)
    dataset = dataset.add_column("qwen_chosen", qwen_chosen)
    dataset = dataset.add_column("qwen_chosen_tokens", qwen_chosen_tokens)
    dataset = dataset.add_column("reject", reject)
    dataset = dataset.add_column("reject_reward", reject_reward)
    dataset = dataset.add_column("reject_origin", reject_origin)
    dataset = dataset.add_column("qwen_reject", qwen_reject)
    dataset = dataset.add_column("qwen_reject_tokens", qwen_reject_tokens)
    dataset = dataset.add_column("g_chosen", g_chosen_list)
    dataset = dataset.add_column("g_reject", g_reject_list)
    dataset = dataset.add_column("j_star", j_star_list)

    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_' + args.score_type +'_beta_'+ str(args.beta) + '_multi_equal_tokenized')


if __name__ == "__main__":
    main()
