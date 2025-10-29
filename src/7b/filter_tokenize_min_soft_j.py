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
        description="Filter + tokenize using worst-coordinate gradients from current/base scores.",
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
    parser.add_argument("--softmax_coefficient", type=float, default=1.0,
                        help="Coefficient for computing soft j_star = softmax(-estimated_p * coeff)")
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

    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    if args.limit_rows and args.limit_rows > 0:
        n_train = min(args.limit_rows, len(ds_dict['train']))
        n_test = min(args.limit_rows, len(ds_dict['test']))
        ds_dict = DatasetDict({
            'train': ds_dict['train'].select(range(n_train)),
            'test': ds_dict['test'].select(range(n_test)),
        })

    def process_split(dataset, is_train):
        print('split length:', len(dataset))
        required_cols = ["qwen_prompt", "qwen_prompt_tokens"]
        for c in required_cols:
            if c not in dataset.column_names:
                raise ValueError(f"Expected preprocessed dataset to contain column '{c}'. Please run preprocess_common.py first.")

        response_pattern = re.compile(r'^(selection|current|base)_response_\d+$')
        response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
        if not response_columns:
            raise ValueError("Dataset is missing response columns required for downstream tokenization.")

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
            # Soft j_star emphasizing lower coordinates: j_star = softmax(-estimated_p * coeff)
            j_star = np.exp(-estimated_p * float(args.softmax_coefficient))
            j_star = j_star / np.sum(j_star)

            g_values = []
            selection_tokens = []
            selection_texts = []
            for sel_id in selection_ids:
                per_base_scores = []
                for base_id in base_ids:
                    key = f"selection_{sel_id}_base_{base_id}_{args.score_type}"
                    raw_scores = row.get(key, None)
                    if raw_scores is None:
                        continue
                    p_vec = np.array(raw_scores, dtype=float)
                    p_vec = np.atleast_1d(p_vec)
                    per_base_scores.append(float(p_vec @ j_star))

                if not per_base_scores:
                    raise ValueError(
                        f"No selection-base scores available for selection {sel_id} to compute gradient."
                    )

                g_values.append(float(np.mean(per_base_scores)))

                sel_token = tokenizer.apply_chat_template(
                    get_message(response=row[f"selection_response_{sel_id}"]),
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

            selected_pairs = []
            if is_train:
                for idx_a in range(len(selection_ids)):
                    for idx_b in range(idx_a + 1, len(selection_ids)):
                        higher_idx, lower_idx = idx_a, idx_b
                        if g_values[higher_idx] < g_values[lower_idx]:
                            higher_idx, lower_idx = lower_idx, higher_idx
                        selected_pairs.append((higher_idx, lower_idx))
            else:
                if g_values:
                    g_array = np.array(g_values, dtype=float)
                    if g_array.size >= 2:
                        sorted_indices = np.argsort(g_array)
                        lower_idx = int(sorted_indices[0])
                        higher_idx = int(sorted_indices[-1])
                        if higher_idx != lower_idx:
                            selected_pairs.append((higher_idx, lower_idx))

            for higher_idx, lower_idx in selected_pairs:
                for col in dataset.column_names:
                    expanded_data[col].append(row[col])

                higher_sel_id = selection_ids[higher_idx]
                lower_sel_id = selection_ids[lower_idx]

                expanded_data["chosen"].append(row[f"selection_response_{higher_sel_id}"])
                expanded_data["reject"].append(row[f"selection_response_{lower_sel_id}"])

                expanded_data["qwen_chosen_tokens"].append(selection_tokens[higher_idx])
                expanded_data["qwen_reject_tokens"].append(selection_tokens[lower_idx])

                expanded_data["qwen_chosen"].append(selection_texts[higher_idx])
                expanded_data["qwen_reject"].append(selection_texts[lower_idx])

                expanded_data["chosen_reward"].append(float(g_values[higher_idx]))
                expanded_data["reject_reward"].append(float(g_values[lower_idx]))

                expanded_data["g_chosen"].append(float(g_values[higher_idx]))
                expanded_data["g_reject"].append(float(g_values[lower_idx]))
                expanded_data["j_star"].append(j_star.tolist())

        dataset = Dataset.from_dict(expanded_data)
        dataset = dataset.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(dataset))
        return dataset

    train_processed = process_split(ds_dict['train'], is_train=True)
    test_processed = process_split(ds_dict['test'], is_train=False)

    out = DatasetDict({"train": train_processed, "test": test_processed})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    base_repo = repo_prefix + '_' + args.score_type + '_beta_' + str(args.beta) + '_softmax_coefficient_' + str(args.softmax_coefficient) + '_min_expand_tokenized'
    out.push_to_hub(base_repo)

    # Optional gap filtering and secondary upload
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
