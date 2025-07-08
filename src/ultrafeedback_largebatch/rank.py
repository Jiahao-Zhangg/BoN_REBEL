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
    parser.add_argument("--selection_pairs", type=int, default=5, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=5, help="number of pairs to use for gradient estimation")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for the reward model")
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

    def tokenized_len(self, message: Dict[str, str]) -> int:
        return len(self.tokenizer.apply_chat_template(
            message,
            truncation=self.truncation,
            max_length=self.max_length,
        ))

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            scores = output.score.float().detach().cpu()
        return scores


def main():

    # init
    args = parse_arguments()
    dataset = load_dataset(args.input_repo, split='train')

    # gather reward
    rewards = {}
    rm = ArmoRMPipeline(args.reward_model, trust_remote_code=True)

    # gather reward
    total_pairs = args.selection_pairs + args.gradient_pairs
    for i in range(total_pairs):
        print(f'gathering reward for {i+1}th response')

        # Create messages
        messages = [(msg_i, get_message(row['prompt'], row[f'response_{i}'])) for msg_i, row in enumerate(dataset)]
        # Sort messages by tokenized length, but store original order
        messages.sort(key=lambda x: rm.tokenized_len(x[1]), reverse=True)
        # Generate rewards in batches
        _rewards, _indices = [], []
        for batch_idx in tqdm(range(0, len(messages), args.batch_size)):
            batch = messages[batch_idx:batch_idx+args.batch_size]
            batch_messages = [x[1] for x in batch]
            batch_indices = [x[0] for x in batch]
            batch_rewards = rm(batch_messages)
            _rewards.append(batch_rewards)
            _indices.extend(batch_indices)
        _rewards = torch.cat(_rewards)
        _indices = torch.tensor(_indices)
        # Revert rewards back to original order
        sorted_rewards = torch.zeros_like(_rewards)
        sorted_rewards[_indices] = _rewards
        rewards[f"response_{i}_reward"] = sorted_rewards.tolist()

    for k, v in rewards.items():
        dataset = dataset.add_column(k, v)

    dataset.push_to_hub(args.input_repo+'_armo')


if __name__ == "__main__":
    main()