import argparse
import numpy as np
import re
import torch
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm
from transformers import AutoTokenizer

torch.set_printoptions(threshold=10_000)

SYS_PROMPT_LEN = 24


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize using a single pair per row: (max g_value, min g_value)."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/whole_sw_maxlen_8192_rescale",
                        help="HF dataset repo to load (expects selection_response_i and selection_i_mean columns)")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--slicing_idx", type=int, default=24)
    parser.add_argument("--output_repo_prefix", type=str, default=None,
                        help="If set, use this as the repo prefix for push_to_hub instead of input_repo")
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split to first N rows for debugging")
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
        # Keep the same preprocessed expectation for compatibility
        required_cols = ["qwen_prompt", "qwen_prompt_tokens"]
        for c in required_cols:
            if c not in dataset.column_names:
                raise ValueError(f"Expected preprocessed dataset to contain column '{c}'. Please run preprocess_common.py first.")

        # Identify available selection responses and corresponding pointwise means
        resp_pat = re.compile(r'^selection_response_(\d+)$')
        mean_pat_template = "selection_{}_mean"

        selection_ids_all = sorted([
            int(m.group(1))
            for name in dataset.column_names
            if (m := resp_pat.match(name))
        ])
        if not selection_ids_all:
            raise ValueError("Dataset is missing 'selection_response_i' columns required for downstream tokenization.")

        # Stream rows via generator to avoid building a huge dict-of-lists
        def row_generator():
            for row in tqdm(dataset):
                # Determine which selection ids have valid means for this row
                usable_ids = []
                g_values = []
                selection_tokens = []
                selection_texts = []

                for sel_id in selection_ids_all:
                    mean_col = mean_pat_template.format(sel_id)
                    if mean_col not in row or row[mean_col] is None:
                        continue
                    try:
                        g_val = float(np.mean(np.atleast_1d(np.array(row[mean_col], dtype=float))))
                    except Exception:
                        # Skip malformed entries
                        continue
                    # Tokenize selection response
                    response_text = row.get(f"selection_response_{sel_id}", None)
                    if response_text is None:
                        continue

                    sel_token = tokenizer.apply_chat_template(
                        get_message(response=response_text),
                        add_generation_prompt=False,
                        tokenize=True,
                        padding='max_length',
                        max_length=args.maxlen + slicing_idx_used,
                    )[slicing_idx_used:]
                    # Ensure shape and capture text
                    if len(sel_token) != args.maxlen:
                        # Skip if cannot produce exact length (strictness matches prior scripts)
                        continue
                    sel_text = tokenizer.decode(sel_token, skip_special_tokens=False)
                    if "Qwen" in args.model:
                        assert not sel_text.lstrip().startswith("<|im_start|>assistant"), "Qwen selection should not include assistant header"
                        assert ("<|eot_id|>" in sel_text) or ("<|im_end|>" in sel_text), "Qwen selection text should include end-of-turn marker"
                        last_id = int(sel_token[-1])
                        pid = tokenizer.pad_token_id
                        eid = tokenizer.eos_token_id
                        assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen selection last token should be PAD or EOS"

                    usable_ids.append(sel_id)
                    g_values.append(float(g_val))
                    selection_tokens.append(list(sel_token))
                    selection_texts.append(sel_text)

                if not usable_ids:
                    raise ValueError("No valid 'selection_i_mean' values available to compute g for this row.")

                # Build a single preference pair using the max and min g_value (both splits)
                selected_pairs = []
                if g_values:
                    g_array = np.array(g_values, dtype=float)
                    if g_array.size >= 2:
                        lower_idx = int(np.argmin(g_array))
                        higher_idx = int(np.argmax(g_array))
                        if higher_idx != lower_idx:
                            selected_pairs.append((higher_idx, lower_idx))

                for higher_idx, lower_idx in selected_pairs:
                    higher_sel_id = usable_ids[higher_idx]
                    lower_sel_id = usable_ids[lower_idx]

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
                    })

                    yield example

        # Explicit features to speed up Arrow construction and use float64
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

        streamed = Dataset.from_generator(row_generator, features=Features(features))
        print('built dataset from generator!')
        streamed = streamed.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(streamed))
        return streamed

    train_processed = process_split(ds_dict['train'], is_train=True)
    test_processed = process_split(ds_dict['test'], is_train=False)

    out = DatasetDict({"train": train_processed, "test": test_processed})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    base_repo = repo_prefix + '_rlcf_tokenized'
    out.push_to_hub(base_repo)


if __name__ == "__main__":
    main()


