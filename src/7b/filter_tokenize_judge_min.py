import argparse
import numpy as np
import re
import torch
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
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
    parser.add_argument("--output_repo_prefix", type=str, default=None,
                        help="If set, use this as the repo prefix for push_to_hub instead of input_repo")
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split to first N rows for debugging")
    parser.add_argument("--gap_ratio", type=float, default=0.0,
                        help="If >0, filter top ratio by (g_chosen - g_reject) per split and push *_gap")
    parser.add_argument("--gap_shuffle_seed", type=int, default=None,
                        help="Shuffle seed used after gap filtering (None = no fixed seed)")
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

    # Handle both train and test splits, mirroring non-expand handling
    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    # Optionally limit rows for debugging
    if args.limit_rows and args.limit_rows > 0:
        n_train = min(args.limit_rows, len(ds_dict['train']))
        n_test = min(args.limit_rows, len(ds_dict['test']))
        ds_dict = DatasetDict({
            'train': ds_dict['train'].select(range(n_train)),
            'test': ds_dict['test'].select(range(n_test)),
        })

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

        # Stream rows via generator to avoid building a huge dict-of-lists.
        # For the min no-expand variant, we yield at most one (chosen, reject) pair per original row.
        def row_generator():
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

                if not g_values or len(g_values) < 2:
                    continue

                chosen_idx = int(np.argmax(g_values))
                reject_idx = int(np.argmin(g_values))
                if chosen_idx == reject_idx:
                    continue

                chosen_sel_id = selection_ids[chosen_idx]
                reject_sel_id = selection_ids[reject_idx]

                chosen_key = f"selection_response_{chosen_sel_id}"
                reject_key = f"selection_response_{reject_sel_id}"

                qwen_chosen_token = tokenizer.apply_chat_template(
                    get_message(response=row[chosen_key]),
                    add_generation_prompt=False,
                    tokenize=True,
                    padding='max_length',
                    max_length=args.maxlen + slicing_idx_used,
                )[slicing_idx_used:]
                chosen_text = tokenizer.decode(qwen_chosen_token, skip_special_tokens=False)
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
                reject_text = tokenizer.decode(qwen_reject_token, skip_special_tokens=False)
                assert len(qwen_reject_token) == args.maxlen
                if "Qwen" in args.model:
                    assert not reject_text.lstrip().startswith("<|im_start|>assistant"), "Qwen reject should not include assistant header"
                    assert ("<|eot_id|>" in reject_text) or ("<|im_end|>" in reject_text), "Qwen reject text should include end-of-turn marker"
                    last_id = int(qwen_reject_token[-1])
                    pid = tokenizer.pad_token_id
                    eid = tokenizer.eos_token_id
                    assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen reject last token should be PAD or EOS"

                example = {col: row[col] for col in dataset.column_names}
                example.update({
                    "chosen": row[chosen_key],
                    "reject": row[reject_key],
                    "qwen_chosen_tokens": list(qwen_chosen_token),
                    "qwen_reject_tokens": list(qwen_reject_token),
                    "qwen_chosen": chosen_text,
                    "qwen_reject": reject_text,
                    "chosen_reward": float(g_values[chosen_idx]),
                    "reject_reward": float(g_values[reject_idx]),
                    "g_chosen": float(g_values[chosen_idx]),
                    "g_reject": float(g_values[reject_idx]),
                })

                yield example

        # Explicit features for faster Arrow construction; use float64 for rewards/g-values
        features = dataset.features.copy()
        features.update({
            "chosen": Value("string"),
            "reject": Value("string"),
            "qwen_chosen": Value("string"),
            "qwen_reject": Value("string"),
            "qwen_chosen_tokens": Sequence(Value("int64")),
            "qwen_reject_tokens": Sequence(Value("int64")),
            "chosen_reward": Value("float64"),
            "reject_reward": Value("float64"),
            "g_chosen": Value("float64"),
            "g_reject": Value("float64"),
        })

        generated = Dataset.from_generator(row_generator, features=Features(features))
        print('built dataset from generator!')
        generated = generated.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(generated))
        return generated

    train_processed = process_split(ds_dict['train'])
    test_processed = process_split(ds_dict['test'])

    out = DatasetDict({"train": train_processed, "test": test_processed})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    base_repo = repo_prefix + '_min_noexpand_tokenized'
    out.push_to_hub(base_repo)

    # Optional gap filtering and secondary upload (mirroring expand scripts)
    if args.gap_ratio and args.gap_ratio > 0.0:
        def gap_filter(split_ds):
            if "g_chosen" not in split_ds.column_names or "g_reject" not in split_ds.column_names:
                raise ValueError("Missing g_chosen or g_reject for gap filtering")
            with_gap = split_ds.map(lambda row: {"_gap": float(row["g_chosen"]) - float(row["g_reject"])})
            sorted_by_gap = with_gap.sort("_gap", reverse=True)
            keep_count = max(1, int(len(sorted_by_gap) * float(args.gap_ratio)))
            filtered = sorted_by_gap.select(range(keep_count)).remove_columns(["_gap"]).shuffle(seed=args.gap_shuffle_seed)
            return filtered

        gap_out = DatasetDict({
            "train": gap_filter(out["train"]),
            "test": out["test"],
        })
        gap_out.push_to_hub(f"{base_repo}_gap_ratio_{args.gap_ratio}")

if __name__ == "__main__":
    main()
