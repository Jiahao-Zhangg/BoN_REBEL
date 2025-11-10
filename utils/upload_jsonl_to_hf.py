#!/usr/bin/env python3

import argparse
import os

from datasets import load_dataset
from huggingface_hub import login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local JSONL file to the Hugging Face Hub as a dataset.")
    parser.add_argument(
        "--file",
        default="shard_00071_scores_adversary.jsonl",
        help="Path to the local JSONL file.",
    )
    parser.add_argument(
        "--repo-id",
        default="zjhhhh/iter2_ver2_scores_adversary_71",
        help="Target repository id on the Hugging Face Hub (e.g., user_or_org/dataset_name).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push the dataset as a private repository.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, reads from HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Max shard size when uploading the dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if token:
        # Log in so push_to_hub can authenticate without prompting
        login(token)

    # Load the local JSONL as a dataset split
    ds = load_dataset("json", data_files=args.file, split="train")
    print(f"Loaded {len(ds)} rows from {args.file}")

    # Push the split to the Hub. This creates/updates the repo.
    ds.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
        token=token,
        max_shard_size=args.max_shard_size,
    )
    print(f"Pushed dataset to {args.repo_id}")


if __name__ == "__main__":
    main()

