import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List

torch.set_printoptions(threshold=10_000)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from generate.py")
    parser.add_argument("--selection_response", type=int, default=3, help="count of selection_response_i columns to score (i starts at 1)")
    parser.add_argument("--base_response", type=int, default=2, help="count of base_response_i columns to score (i starts at 1)")
    parser.add_argument("--current_response", type=int, default=3, help="count of current_response_i columns to score (i starts at 1)")
    parser.add_argument("--start_idx", type=int, default=0, help="inclusive start index of dataset slice")
    parser.add_argument("--end_idx", type=int, default=None, help="exclusive end index of dataset slice (defaults to dataset end)")
    return parser.parse_args()


def get_message(instruction, response):
    return [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> float:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return score


def main():

    # init
    args = parse_arguments()
    dataset = load_dataset(args.input_repo, split='train')
    if args.end_idx is None:
        args.end_idx = len(dataset)
    dataset = dataset.select(range(args.start_idx, args.end_idx))

    # gather reward
    rewards = {}
    rm = ArmoRMPipeline(args.reward_model, trust_remote_code=True)

    column_names = set(dataset.column_names)

    # score all selection_response_i, base_response_i, and current_response_i
    categories = [
        ("selection_response", args.selection_response),
        ("base_response", args.base_response),
        ("current_response", args.current_response),
    ]

    # Filter out rows where prompt + ANY response would exceed tokenizer max_length
    response_columns = []
    for prefix, count in categories:
        for i in range(1, count + 1):
            col_name = f"{prefix}_{i}"
            if col_name in column_names:
                response_columns.append(col_name)

    if response_columns:
        max_len = rm.max_length
        print(f"Filtering rows longer than {max_len} tokens across {len(response_columns)} response columns...")

        def within_limit(row):
            prompt = row.get('prompt', '')
            for rc in response_columns:
                response = row.get(rc, None)
                if response is None:
                    continue
                messages = get_message(prompt, response)
                input_ids = rm.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    truncation=False,
                )
                seq_len = input_ids.shape[1] if input_ids.dim() == 2 else input_ids.shape[0]
                if seq_len > max_len:
                    return False
            return True

        before = len(dataset)
        dataset = dataset.filter(within_limit, desc="filter-long-rows")
        after = len(dataset)
        print(f"Filtered {before - after} rows; {after} remain.")

    for prefix, count in categories:
        for i in range(1, count + 1):
            col_name = f"{prefix}_{i}"
            if col_name not in column_names:
                continue
            reward_col = f"{col_name}_reward"
            print(f'gathering reward for {col_name}')
            rewards[reward_col] = []
            for row in tqdm(dataset):
                reward = rm(get_message(row['prompt'], row[col_name]))
                rewards[reward_col].append(reward)

    for k, v in rewards.items():
        dataset = dataset.add_column(k, v)

    dataset.push_to_hub(f"{args.input_repo}_armo_{args.end_idx}")


if __name__ == "__main__":
    main()