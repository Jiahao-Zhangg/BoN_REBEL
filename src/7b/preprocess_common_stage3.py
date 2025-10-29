import argparse
import re
import torch
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer


torch.set_printoptions(threshold=10_000)


# WARNING: Magic number, make sure it works for your model
SYS_PROMPT_LEN = 24


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Common preprocessing: filter prompt/response lengths, ensure proper endings, and add prompt tokens.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--input_repo",
        type=str,
        required=True,
        help="HF dataset repo to load raw data (expects selection/current/base responses and scores)",
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        required=True,
        help="HF dataset repo to push the preprocessed dataset",
    )
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    parser.add_argument("--test_size", type=int, default=1000, help="Number of examples for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--limit_rows", type=int, default=0, help="If >0, use only the first N rows from input before splitting")
    parser.add_argument(
        "--slicing_idx",
        type=int,
        default=24,
        help="Fallback slicing index if model-specific detection not used",
    )
    # Intersection-based test split controls
    parser.add_argument(
        "--use_intersection_test",
        action="store_true",
        help=(
            "If set, build the test split as the intersection of the current input repo and the test split of "
            "--reference_test_repo using --id_column as the join key."
        ),
    )
    parser.add_argument(
        "--reference_test_repo",
        type=str,
        default="zjhhhh/stage2_preprocessed",
        help="HF dataset repo whose test split provides keys to intersect with (default: zjhhhh/stage1_preprocessed).",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="prompt",
        help="Column name used to compute intersection between datasets (default: prompt).",
    )
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


def prepare_tokenizers(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_left = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    if "Qwen" in model_name:
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

    return tokenizer, tokenizer_left


def determine_slicing_idx(model_name: str, fallback_idx: int) -> int:
    if "Qwen" in model_name:
        return SYS_PROMPT_LEN
    return fallback_idx


def main():
    args = parse_arguments()

    tokenizer, tokenizer_left = prepare_tokenizers(args.model)
    slicing_idx_used = determine_slicing_idx(args.model, args.slicing_idx)

    base_dataset = load_dataset(args.input_repo, split='train')
    print('initial length:', len(base_dataset))

    if args.limit_rows and args.limit_rows > 0:
        n = min(args.limit_rows, len(base_dataset))
        base_dataset = base_dataset.select(range(n))
        print(f'limited to first {n} rows')

    # Build splits
    if args.use_intersection_test:
        # Use the test split from a reference repo and take intersection by a key column
        ref_test = load_dataset(args.reference_test_repo, split="test")
        if args.id_column not in base_dataset.column_names:
            raise ValueError(
                f"id_column '{args.id_column}' not found in input dataset columns: {base_dataset.column_names}"
            )
        if args.id_column not in ref_test.column_names:
            raise ValueError(
                f"id_column '{args.id_column}' not found in reference test dataset columns: {ref_test.column_names}"
            )

        ref_keys = set(ref_test[args.id_column])
        print(f"reference test size: {len(ref_test)}; unique keys: {len(ref_keys)}")

        def in_ref(row, keys):
            return row[args.id_column] in keys

        def not_in_ref(row, keys):
            return row[args.id_column] not in keys

        test_split = base_dataset.filter(in_ref, fn_kwargs={"keys": ref_keys})
        train_split = base_dataset.filter(not_in_ref, fn_kwargs={"keys": ref_keys})
        print(f"after intersection -> train: {len(train_split)}, test: {len(test_split)}")
        split_dd = DatasetDict({"train": train_split, "test": test_split})
    else:
        # Random split to avoid leakage across methods
        split_dd = base_dataset.train_test_split(test_size=args.test_size, shuffle=True, seed=args.seed)

    def preprocess_split(ds):
        # 1) Filter overly long prompts
        ds = ds.filter(
            lambda row: tokenizer.apply_chat_template(
                get_message(row['prompt']),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).shape[-1] <= args.maxlen_prompt
        )

        # 2) Filter responses by length across all selection/current/base response columns
        response_pattern = re.compile(r'^(selection|current|base)_response_\d+$')
        response_columns = sorted([name for name in ds.column_names if response_pattern.match(name)])
        if not response_columns:
            raise ValueError("Dataset is missing response columns required for length filtering.")

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

        ds = ds.filter(responses_within_limit)

        # 3) Ensure proper ending (PAD/EOS)
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

                if len(response_token) != args.maxlen:
                    return False

                pad_id = tokenizer.pad_token_id
                eos_id = tokenizer.eos_token_id
                if not response_token:
                    return False
                last_id = int(response_token[-1])
                if pad_id is None and eos_id is None:
                    continue
                pad_ok = pad_id is not None and last_id == pad_id
                eos_ok = eos_id is not None and last_id == eos_id
                if not (pad_ok or eos_ok):
                    return False
            return True

        ds = ds.filter(responses_end_properly)

        # 4) Add prompt tokens
        qwen_prompts = []
        qwen_prompt_tokens = []
        for row in tqdm(ds):
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
                assert (
                    "<|start_header_id|>" in qwen_prompt or "<|im_start|>" in qwen_prompt
                ), "Qwen prompt missing chat header markers"
            qwen_prompts.append(qwen_prompt)
            qwen_prompt_tokens.append(qwen_prompt_token)
        ds = ds.add_column("qwen_prompt", qwen_prompts)
        ds = ds.add_column("qwen_prompt_tokens", qwen_prompt_tokens)
        return ds

    processed_train = preprocess_split(split_dd["train"])
    processed_test = preprocess_split(split_dd["test"])

    out = DatasetDict({"train": processed_train, "test": processed_test})
    out.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()

