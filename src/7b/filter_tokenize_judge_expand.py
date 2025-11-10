import argparse
import numpy as np
import re
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
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
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split to first N rows for debugging")
    parser.add_argument("--gap_ratio", type=float, default=0.0,
                        help="If >0, filter top ratio by (g_chosen - g_reject) per split and push *_gap")
    parser.add_argument("--gap_shuffle_seed", type=int, default=None,
                        help="Shuffle seed used after gap filtering (None = no fixed seed)")
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

    if "Qwen" in args.model:
        slicing_idx_used = SYS_PROMPT_LEN
        print(f'slicing index used (fixed): {slicing_idx_used}')
    else:
        slicing_idx_used = args.slicing_idx

    # Load both splits from preprocessed dataset
    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    # Optionally limit rows in each split for debugging
    if args.limit_rows and args.limit_rows > 0:
        n_train = min(args.limit_rows, len(ds_dict['train']))
        n_test = min(args.limit_rows, len(ds_dict['test']))
        ds_dict = DatasetDict({
            'train': ds_dict['train'].select(range(n_train)),
            'test': ds_dict['test'].select(range(n_test)),
        })

    # Process a single split
    def process_split(dataset, is_train):
        print('split length:', len(dataset))
        required_cols = ["qwen_prompt", "qwen_prompt_tokens"]
        for c in required_cols:
            if c not in dataset.column_names:
                raise ValueError(f"Expected preprocessed dataset to contain column '{c}'. Please run preprocess_common.py first.")

        # Allow datasets that use either 'base' or 'adversary' terminology
        response_pattern = re.compile(r'^(selection|current|base|adversary)_response_\d+$')
        response_columns = sorted([name for name in dataset.column_names if response_pattern.match(name)])
        if not response_columns:
            raise ValueError("Dataset is missing response columns required for downstream tokenization.")

        escaped_score_type = re.escape(args.score_type)
        # Detect whether score columns use 'base' or 'adversary'
        detected_key = None
        for candidate in ("base", "adversary"):
            sel_pat = re.compile(rf"^selection_(\\d+)_{candidate}_(\\d+)_({escaped_score_type})$")
            cur_pat = re.compile(rf"^current_(\\d+)_{candidate}_(\\d+)_({escaped_score_type})$")
            sel_hits = any(sel_pat.match(n) for n in dataset.column_names)
            cur_hits = any(cur_pat.match(n) for n in dataset.column_names)
            if sel_hits and cur_hits:
                detected_key = candidate
                break
        if detected_key is None:
            raise ValueError("Could not find score columns using either 'base' or 'adversary'.")

        selection_score_pattern = re.compile(rf"^selection_(\\d+)_{detected_key}_(\\d+)_({escaped_score_type})$")
        current_score_pattern = re.compile(rf"^current_(\\d+)_{detected_key}_(\\d+)_({escaped_score_type})$")
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

        # Stream rows via generator to avoid building a huge dict-of-lists
        def row_generator():
            for row in tqdm(dataset):
                beta = args.beta
                # Build current vs base/adversary matrices per base: shape (K=len(current_ids), L)
                current_base_scores = []
                vector_len = None
                for base_id in base_ids:
                    rows = []
                    for cur_id in current_ids:
                        key = f"current_{cur_id}_{detected_key}_{base_id}_{args.score_type}"
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
                        rows.append(p_vec)
                    if not rows:
                        continue
                    current_base_scores.append(np.stack(rows, axis=0))  # (K, L)
                if not current_base_scores:
                    raise ValueError("No current-base/adversary scores available to estimate A_vec and j_star.")

                # Compute exp_terms per base: exp(- (1/K) * sum_k P_m / beta) for each m
                exp_terms_list = []  # list of vectors (L,)
                for cb in current_base_scores:
                    inner_sums = np.sum(cb, axis=0) / (beta * cb.shape[0])  # (L,)
                    exp_terms_list.append(np.exp(-inner_sums))

                # A_m(x) = sum_i exp_terms_i[m] across bases
                A_vec = exp_terms_list[0]
                for i in range(1, len(exp_terms_list)):
                    A_vec = A_vec + exp_terms_list[i]
                j_star = int(np.argmax(A_vec))

                # Compute g(z) using weighted B at m*
                g_values = []
                selection_tokens = []
                selection_texts = []
                for sel_id in selection_ids:
                    # Gather selection vectors per base
                    p_z_vs_bases = []
                    for base_index, base_id in enumerate(base_ids):
                        key = f"selection_{sel_id}_{detected_key}_{base_id}_{args.score_type}"
                        raw_scores = row.get(key, None)
                        if raw_scores is None:
                            continue
                        p_vec = np.array(raw_scores, dtype=float)
                        p_vec = np.atleast_1d(p_vec)
                        if j_star >= p_vec.shape[0]:
                            raise ValueError(
                                f"Score vector for {key} missing coordinate j*={j_star} (length={p_vec.shape[0]})."
                            )
                        # Weighted by exp_terms at coordinate j_star for that base
                        weight = exp_terms_list[base_index][j_star]
                        p_z_vs_bases.append(float(p_vec[j_star]) * float(weight))

                    if not p_z_vs_bases:
                        raise ValueError(
                            f"No selection-{detected_key} scores available for selection {sel_id} to compute g."
                        )

                    B_star = float(np.sum(p_z_vs_bases))
                    g_z = B_star / float(A_vec[j_star])
                    g_values.append(float(g_z))

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
                    higher_sel_id = selection_ids[higher_idx]
                    lower_sel_id = selection_ids[lower_idx]

                    example = {col: row[col] for col in dataset.column_names}
                    example.update({
                        "chosen": row[f"selection_response_{higher_sel_id}"],
                        "reject": row[f"selection_response_{lower_sel_id}"],
                        "qwen_chosen_tokens": selection_tokens[higher_idx],
                        "qwen_reject_tokens": selection_tokens[lower_idx],
                        "qwen_chosen": selection_texts[higher_idx],
                        "qwen_reject": selection_texts[lower_idx],
                        "chosen_reward": float(g_values[higher_idx]),
                        "reject_reward": float(g_values[lower_idx]),
                        "g_chosen": float(g_values[higher_idx]),
                        "g_reject": float(g_values[lower_idx]),
                        "j_star": int(j_star),
                    })

                    yield example

        # Explicit features to speed up Arrow construction and use float64 for rewards/g-values
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
            "j_star": Value("int64"),
        })

        streamed = Dataset.from_generator(row_generator, features=Features(features))
        print('built dataset from generator!')
        streamed = streamed.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(streamed))
        return streamed

    train_processed = process_split(ds_dict['train'], is_train=True)
    test_processed = process_split(ds_dict['test'], is_train=False)

    out = DatasetDict({"train": train_processed, "test": test_processed})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    base_repo = repo_prefix + '_' + args.score_type + '_beta_' + str(args.beta) + '_multi_expand_tokenized'
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
