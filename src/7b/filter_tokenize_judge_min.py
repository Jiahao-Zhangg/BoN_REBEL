import argparse
import numpy as np
import re
import torch
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

torch.set_printoptions(threshold=10_000)


# WARNING: Magic number, make sure it works for your model
SYS_PROMPT_LEN = 24


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter + tokenize using j* from current-base averages, then g(z)=mean_base P_{j*}(x,z,base).",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_rescale",
                        help="HF dataset repo to load (expects selection/current/base responses and score vectors + requirements)")
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

    # Handle both train and test splits, mirroring non-expand handling
    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    if "Qwen" in args.model:
        slicing_idx_used = SYS_PROMPT_LEN
    else:
        slicing_idx_used = args.slicing_idx

    def process_split(dataset):
        print('split length:', len(dataset))

        # Filter overly long prompts
        dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(
            get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
        print('filtered long prompts:', len(dataset))

        # Identify response columns
        response_pattern = re.compile(r'^(selection|current|base)_response_\d+$')
        response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
        if not response_columns:
            raise ValueError("Dataset is missing response columns required for length filtering.")
        print(f'response columns length-filtered: {response_columns}')

        # Filter responses within token limit
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

                if "Qwen" in args.model:
                    last_id = int(response_token[-1])
                    pid = tokenizer.pad_token_id
                    eid = tokenizer.eos_token_id
                    if not ((pid is not None and last_id == pid) or (eid is not None and last_id == eid)):
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

        # Discover score columns
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

        # Compute chosen/reject per example
        chosen, reject = [], []
        qwen_chosen, qwen_reject = [], []
        qwen_chosen_tokens, qwen_reject_tokens = [], []
        chosen_reward, reject_reward = [], []
        g_chosen_list, g_reject_list = [], []

        for row in tqdm(dataset):
            # Estimate j* from current-base averages
            current_vectors = []
            vector_len = None
            for cur_id in current_ids:
                for base_id in base_ids:
                    key = f"current_{cur_id}_base_{base_id}_{args.score_type}"
                    raw_scores = row.get(key, None)
                    if raw_scores is None:
                        continue
                    p_vec = np.array(raw_scores, dtype=float)
                    p_vec = np.atleast_1d(p_vec)
                    if vector_len is None:
                        vector_len = p_vec.shape[0]
                    elif p_vec.shape[0] != vector_len:
                        raise ValueError(
                            f"Mismatched score vector length for {key}: expected {vector_len}, got {p_vec.shape[0]}"
                        )
                    current_vectors.append(p_vec)
            if not current_vectors:
                raise ValueError("No current-base scores available to estimate P(x, pi, pi').")

            stacked_current = np.stack(current_vectors, axis=0)
            estimated_p = np.mean(stacked_current, axis=0)
            j_star = int(np.argmin(estimated_p))

            # g(z) = mean over bases of selection score at j*
            g_values = []
            for sel_id in selection_ids:
                per_base_scores = []
                for base_id in base_ids:
                    key = f"selection_{sel_id}_base_{base_id}_{args.score_type}"
                    raw_scores = row.get(key, None)
                    if raw_scores is None:
                        continue
                    p_vec = np.array(raw_scores, dtype=float)
                    p_vec = np.atleast_1d(p_vec)
                    if j_star >= p_vec.shape[0]:
                        raise ValueError(
                            f"Score vector for {key} missing coordinate j*={j_star} (length={p_vec.shape[0]})."
                        )
                    per_base_scores.append(float(p_vec[j_star]))
                if not per_base_scores:
                    raise ValueError(
                        f"No selection-base scores available for selection {sel_id} to compute gradient."
                    )
                g_values.append(float(np.mean(per_base_scores)))

            chosen_idx = int(np.argmax(g_values))
            reject_idx = int(np.argmin(g_values))
            chosen_sel_id = selection_ids[chosen_idx]
            reject_sel_id = selection_ids[reject_idx]

            chosen_key = f"selection_response_{chosen_sel_id}"
            reject_key = f"selection_response_{reject_sel_id}"
            chosen.append(row[chosen_key])
            reject.append(row[reject_key])

            qwen_chosen_token = tokenizer.apply_chat_template(
                get_message(response=row[chosen_key]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen + slicing_idx_used,
            )[slicing_idx_used:]
            qwen_chosen_tokens.append(qwen_chosen_token)
            chosen_text = tokenizer.decode(qwen_chosen_token, skip_special_tokens=False)
            qwen_chosen.append(chosen_text)
            chosen_reward.append(g_values[chosen_idx])
            assert len(qwen_chosen_token) == args.maxlen
            if "Qwen" in args.model:
                assert not chosen_text.lstrip().startswith("<|im_start|>assistant"), "Qwen chosen should not include assistant header"
                assert ("<|eot_id|>" in chosen_text) or ("<|im_end|>" in chosen_text), "Qwen chosen text should include end-of-turn marker"
                last_id = int(qwen_chosen_token[-1])
                pid = tokenizer.pad_token_id
                eid = tokenizer.eos_token_id
                assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen chosen last token should be PAD or EOS"

            qwen_reject_token = tokenizer.apply_chat_template(
                get_message(response=row[reject_key]),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen + slicing_idx_used,
            )[slicing_idx_used:]
            qwen_reject_tokens.append(qwen_reject_token)
            reject_text = tokenizer.decode(qwen_reject_token, skip_special_tokens=False)
            qwen_reject.append(reject_text)
            reject_reward.append(g_values[reject_idx])
            assert len(qwen_reject_token) == args.maxlen
            if "Qwen" in args.model:
                assert not reject_text.lstrip().startswith("<|im_start|>assistant"), "Qwen reject should not include assistant header"
                assert ("<|eot_id|>" in reject_text) or ("<|im_end|>" in reject_text), "Qwen reject text should include end-of-turn marker"
                last_id = int(qwen_reject_token[-1])
                pid = tokenizer.pad_token_id
                eid = tokenizer.eos_token_id
                assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen reject last token should be PAD or EOS"

            g_chosen_list.append(g_values[chosen_idx])
            g_reject_list.append(g_values[reject_idx])

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
        return dataset

    train_processed = process_split(ds_dict['train'])
    test_processed = process_split(ds_dict['test'])

    out = DatasetDict({"train": train_processed, "test": test_processed})
    out.push_to_hub(args.input_repo + '_beta_' + str(args.beta) + '_min_noexpand_tokenized')

if __name__ == "__main__":
    main()
