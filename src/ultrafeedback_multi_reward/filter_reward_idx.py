import argparse
import re
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import get_message

torch.set_printoptions(threshold=10_000)
SYS_PROMPT_LEN = 30


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simple filter: use reward_idx from response_i_reward as g and expand pairwise."
    )
    parser.add_argument("--input_repo", type=str, required=True,
                        help="HF dataset repo pushed by preprocess_common.py (expects response_i and response_i_reward)")
    parser.add_argument("--selection_pairs", type=int, required=True,
                        help="Number of selection responses per row (first indices after sorting)")
    parser.add_argument("--reward_idx", type=int, required=True,
                        help="Index into response_i_reward to use as g")
    parser.add_argument("--output_repo_prefix", type=str, default=None,
                        help="If set, use as repo prefix for push_to_hub; otherwise reuse input_repo")
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split for debugging")
    parser.add_argument("--gap_ratio", type=float, default=1.0,
                        help="Fraction (0,1] of top pairs by (g_chosen - g_reject) to keep per row")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HF model id for tokenization")
    parser.add_argument("--maxlen", type=int, default=2048,
        help="Max response token length (excludes SYS_PROMPT_LEN)")
    return parser.parse_args()


def filter_same_responses(row):
    return row['chosen'] != row['reject']


def main():
    args = parse_arguments()

    # Init tokenizer for chosen/reject tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load both splits from preprocessed dataset
    ds_dict = load_dataset(args.input_repo)
    if 'train' not in ds_dict or 'test' not in ds_dict:
        raise ValueError("Preprocessed dataset must contain 'train' and 'test' splits.")

    # Optionally limit rows per split
    if args.limit_rows and args.limit_rows > 0:
        n_train = min(args.limit_rows, len(ds_dict['train']))
        n_test = min(args.limit_rows, len(ds_dict['test']))
        ds_dict = DatasetDict({
            'train': ds_dict['train'].select(range(n_train)),
            'test': ds_dict['test'].select(range(n_test)),
        })

    def process_split(dataset, use_gap_filter=False):
        print('split length:', len(dataset))

        # Identify response indices present (response_i and response_i_reward)
        resp_pat = re.compile(r'^response_(\d+)$')
        response_ids_all = sorted([
            int(m.group(1))
            for name in dataset.column_names
            if (m := resp_pat.match(name)) and (f"response_{m.group(1)}_reward" in dataset.column_names)
        ])
        if not response_ids_all:
            raise ValueError("Dataset must contain 'response_i' and matching 'response_i_reward' columns.")

        if len(response_ids_all) < args.selection_pairs:
            raise ValueError(f"Found {len(response_ids_all)} responses but require at least {args.selection_pairs} "
                             f"(selection_pairs).")

        selection_ids = response_ids_all[:args.selection_pairs]

        def row_generator():
            for row in tqdm(dataset):
                # Compute g-values for selections using reward_idx
                g_values = []
                try:
                    for sid in selection_ids:
                        rewards = row[f"response_{sid}_reward"]
                        if rewards is None or args.reward_idx < 0 or args.reward_idx >= len(rewards):
                            raise IndexError(f"reward_idx {args.reward_idx} out of bounds for response_{sid}_reward")
                        g_values.append(float(rewards[args.reward_idx]))
                except (KeyError, IndexError, TypeError):
                    # Skip malformed rows
                    continue

                # Expand to all pairwise comparisons among selection responses
                pair_candidates = []
                for idx_a in range(len(selection_ids)):
                    for idx_b in range(idx_a + 1, len(selection_ids)):
                        higher_idx, lower_idx = idx_a, idx_b
                        if g_values[higher_idx] < g_values[lower_idx]:
                            higher_idx, lower_idx = lower_idx, higher_idx

                        higher_sel_id = selection_ids[higher_idx]
                        lower_sel_id = selection_ids[lower_idx]

                        # Preserve original columns and append pairwise outputs
                        example = {col: row[col] for col in dataset.column_names}
                        g_chosen = float(g_values[higher_idx])
                        g_reject = float(g_values[lower_idx])
                        gap = g_chosen - g_reject
                        example.update({
                            "chosen": row[f"response_{higher_sel_id}"],
                            "reject": row[f"response_{lower_sel_id}"],
                            "chosen_reward": g_chosen,
                            "reject_reward": g_reject,
                            "g_chosen": g_chosen,
                            "g_reject": g_reject,
                            "gap": gap,
                        })

                        # Tokenize chosen and reject like in filter_tokenize.py
                        llama_chosen_token = tokenizer.apply_chat_template(
                            get_message(response=example["chosen"]),
                            add_generation_prompt=False,
                            tokenize=True,
                            padding='max_length',
                            max_length=args.maxlen + SYS_PROMPT_LEN,
                        )[SYS_PROMPT_LEN:]
                        llama_chosen_tokens = llama_chosen_token
                        llama_chosen = tokenizer.decode(llama_chosen_token, skip_special_tokens=False)
                        assert len(llama_chosen_token) == args.maxlen
                        assert llama_chosen_token[-1] == 128009 or llama_chosen_token[-1] == 128256
                        example["llama_chosen"] = llama_chosen
                        example["llama_chosen_tokens"] = llama_chosen_tokens

                        llama_reject_token = tokenizer.apply_chat_template(
                            get_message(response=example["reject"]),
                            add_generation_prompt=False,
                            tokenize=True,
                            padding='max_length',
                            max_length=args.maxlen + SYS_PROMPT_LEN,
                        )[SYS_PROMPT_LEN:]
                        llama_reject_tokens = llama_reject_token
                        llama_reject = tokenizer.decode(llama_reject_token, skip_special_tokens=False)
                        assert len(llama_reject_token) == args.maxlen
                        assert llama_reject_token[-1] == 128009 or llama_reject_token[-1] == 128256
                        example["llama_reject"] = llama_reject
                        example["llama_reject_tokens"] = llama_reject_tokens
                        pair_candidates.append((gap, example))

                if not use_gap_filter:
                    for _, ex in pair_candidates:
                        yield ex
                else:
                    # When using gap filtering, we still yield all expanded pairs here.
                    # Global filtering across the entire split is applied after dataset construction.
                    for _, ex in pair_candidates:
                        yield ex

        # Build explicit features to include new columns
        features = dataset.features.copy()
        features.update({
            "chosen": Value("string"),
            "reject": Value("string"),
            "chosen_reward": Value("float64"),
            "reject_reward": Value("float64"),
            "g_chosen": Value("float64"),
            "g_reject": Value("float64"),
            "gap": Value("float64"),
            "llama_chosen": Value("string"),
            "llama_chosen_tokens": Sequence(Value("int64")),
            "llama_reject": Value("string"),
            "llama_reject_tokens": Sequence(Value("int64")),
        })

        streamed = Dataset.from_generator(row_generator, features=Features(features))
        print('built dataset from generator!')
        streamed = streamed.filter(lambda row: filter_same_responses(row))
        print('filtered same responses:', len(streamed))
        # If gap filtering is requested, apply it globally across the entire expanded split.
        if use_gap_filter:
            n_total = len(streamed)
            ratio = args.gap_ratio
            if n_total == 0 or ratio <= 0.0:
                # Keep zero pairs if ratio <= 0 or dataset empty
                streamed = streamed.select([])
            else:
                ratio = min(1.0, ratio)
                top_k = int(np.ceil(ratio * n_total))
                top_k = min(n_total, top_k)
                # Sort by gap ascending, then select the last top_k as global top by gap
                sorted_ds = streamed.sort("gap")
                start_idx = max(0, len(sorted_ds) - top_k)
                indices = list(range(start_idx, len(sorted_ds)))
                streamed = sorted_ds.select(indices)
            print('applied global gap filter:', len(streamed))
        return streamed

    train_full = process_split(ds_dict['train'], use_gap_filter=False)
    test_full = process_split(ds_dict['test'], use_gap_filter=False)

    out_full = DatasetDict({"train": train_full, "test": test_full})
    repo_prefix = args.output_repo_prefix if args.output_repo_prefix else args.input_repo
    out_full.push_to_hub(repo_prefix + f"_rewardidx{args.reward_idx}_tokenized")

    train_gap = process_split(ds_dict['train'], use_gap_filter=True)
    test_gap = process_split(ds_dict['test'], use_gap_filter=True)

    out_gap = DatasetDict({"train": train_gap, "test": test_gap})
    out_gap.push_to_hub(repo_prefix + f"_rewardidx{args.reward_idx}_tokenized_gap_{args.gap_ratio}")


if __name__ == "__main__":
    main()


