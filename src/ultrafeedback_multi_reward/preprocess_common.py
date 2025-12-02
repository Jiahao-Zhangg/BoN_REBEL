import argparse
import re
import torch
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer


torch.set_printoptions(threshold=10_000)


# WARNING: Magic number, make sure it works for your model (LLaMA)
SYS_PROMPT_LEN = 30


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Common preprocessing: filter prompt/response lengths, ensure proper endings, and add prompt tokens.",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--input_repo",
        type=str,
        required=True,
        help="HF dataset repo to load raw data (expects response_i columns and scores)",
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
        default=30,
        help="Fallback slicing index if model-specific detection not used",
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

    # Align with filter_tokenize.py: ensure PAD exists for both tokenizers
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer, tokenizer_left


def determine_slicing_idx(model_name: str, fallback_idx: int) -> int:
    if "llama" in model_name.lower():
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

    # Split FIRST to avoid leakage across methods
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

        # 2) Filter responses by length across all response_i columns (match filter_tokenize.py format)
        response_pattern = re.compile(r'^response_\d+$')
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
        llama_prompts = []
        llama_prompt_tokens = []
        for row in tqdm(ds):
            llama_prompt_token = tokenizer_left.apply_chat_template(
                get_message(row['prompt']),
                add_generation_prompt=True,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen_prompt,
            )
            llama_prompt = tokenizer_left.decode(llama_prompt_token, skip_special_tokens=False)
            assert len(llama_prompt_token) == args.maxlen_prompt
            if "llama" in args.model.lower():
                # Mirror checks used in filter_tokenize.py for LLaMA prompts
                assert (llama_prompt_token[0] == 128000 or llama_prompt_token[0] == 128256) and llama_prompt_token[-1] == 271
            llama_prompts.append(llama_prompt)
            llama_prompt_tokens.append(llama_prompt_token)
        ds = ds.add_column("llama_prompt", llama_prompts)
        ds = ds.add_column("llama_prompt_tokens", llama_prompt_tokens)
        return ds

    processed_train = preprocess_split(split_dd["train"])
    processed_test = preprocess_split(split_dd["test"])

    out = DatasetDict({"train": processed_train, "test": processed_test})
    out.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()


