import argparse
import re
from typing import List, Set

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
        description="Prepare SFT dataset: for each row, take the union of responses that are best per reward dimension."
    )
    parser.add_argument("--input_repo", type=str, required=True,
                        help="HF dataset repo pushed by preprocess_common.py (expects response_i and response_i_reward)")
    parser.add_argument("--output_repo", type=str, required=True,
                        help="HF dataset repo to push the SFT dataset")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HF model id for tokenization")
    parser.add_argument("--maxlen", type=int, default=2048,
                        help="Max response token length (excludes SYS_PROMPT_LEN)")
    parser.add_argument("--limit_rows", type=int, default=0,
                        help="If >0, limit each split for debugging")
    parser.add_argument(
        "--rewards_list",
        type=int,
        nargs="*",
        default=None,
        help="Optional 0-based indices of reward dimensions to consider. "
             "If omitted, uses all dimensions.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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

    def process_split(dataset):
        print('split length:', len(dataset))

        # Find response indices with corresponding reward arrays
        resp_pat = re.compile(r'^response_(\d+)$')
        response_ids_all = sorted([
            int(m.group(1))
            for name in dataset.column_names
            if (m := resp_pat.match(name)) and (f"response_{m.group(1)}_reward" in dataset.column_names)
        ])
        if not response_ids_all:
            raise ValueError("Dataset must contain 'response_i' and matching 'response_i_reward' columns.")

        def row_generator():
            for row in tqdm(dataset):
                # Load reward arrays for all responses in this row
                rewards_list: List[List[float]] = []
                responses: List[str] = []
                try:
                    for sid in response_ids_all:
                        rewards = row[f"response_{sid}_reward"]
                        resp = row[f"response_{sid}"]
                        if rewards is None or resp is None:
                            raise KeyError("Missing response or reward")
                        rewards_list.append(list(rewards))
                        responses.append(resp)
                except KeyError:
                    # Skip malformed rows
                    continue

                if not rewards_list:
                    continue

                # Ensure consistent reward dimension
                reward_dims = [len(r) for r in rewards_list]
                if len(set(reward_dims)) != 1:
                    # Skip malformed row with inconsistent reward dimensions
                    continue
                H = reward_dims[0]

                # Build matrix (S, H) for argmax per dimension
                reward_matrix = np.array(rewards_list, dtype=float)  # (S, H)
                # Choose which reward dimensions to consider
                if args.rewards_list:
                    # Deduplicate while preserving user-provided order
                    dims_to_use = list(dict.fromkeys(args.rewards_list))
                else:
                    dims_to_use = list(range(H))
                # Validate indices for this row; skip malformed rows
                if any(d < 0 or d >= H for d in dims_to_use):
                    continue
                # Select the index of the best response for each selected reward dimension
                best_indices_per_dim = np.argmax(reward_matrix[:, dims_to_use], axis=0).tolist()  # length len(dims_to_use)

                # Deduplicate selected responses while preserving order of dimensions
                selected_texts: List[str] = []
                seen: Set[str] = set()
                for idx in best_indices_per_dim:
                    text = responses[idx]
                    if text not in seen:
                        selected_texts.append(text)
                        seen.add(text)

                # Yield one SFT training example per selected response
                for resp_text in selected_texts:
                    example = {col: row[col] for col in dataset.column_names if col in row}
                    example["chosen"] = resp_text

                    # Tokenize chosen like other scripts
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
                    yield example

        # Build features with new columns for SFT
        features = dataset.features.copy()
        features.update({
            "chosen": Value("string"),
            "llama_chosen": Value("string"),
            "llama_chosen_tokens": Sequence(Value("int64")),
        })

        streamed = Dataset.from_generator(row_generator, features=Features(features))
        print('built SFT dataset from generator!', len(streamed))
        return streamed

    train_sft = process_split(ds_dict['train'])
    test_sft = process_split(ds_dict['test'])

    out = DatasetDict({"train": train_sft, "test": test_sft})
    out.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()


