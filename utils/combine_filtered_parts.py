#!/usr/bin/env python3
"""
Combine two filtered HF datasets and upload the merged result.

Defaults:
  - Inputs:  
      zjhhhh/generation_for_iter2_ver2_part1_filtered  
      zjhhhh/generation_for_iter2_ver2_part2_filtered
  - Output:  
      zjhhhh/generation_for_iter2_ver2_filtered

Usage examples:
  python combine_filtered_parts.py
  python combine_filtered_parts.py \
    --part1 zjhhhh/generation_for_iter2_ver2_part1_filtered \
    --part2 zjhhhh/generation_for_iter2_ver2_part2_filtered \
    --output zjhhhh/generation_for_iter2_ver2_filtered \
    --split train

Requires:
  pip install datasets huggingface_hub
  export HF_TOKEN=... (or pass --token)
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concatenate two HF datasets and push the result")
    p.add_argument(
        "--part1",
        default="zjhhhh/generation_for_iter2_multi_part1_filtered",
        help="First input dataset repo id",
    )
    p.add_argument(
        "--part2",
        default="zjhhhh/generation_for_iter2_multi_part2_filtered",
        help="Second input dataset repo id",
    )
    p.add_argument(
        "--output",
        default="zjhhhh/generation_for_iter2_multi_filtered",
        help="Output dataset repo id",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split name to load and push (default: train)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Upload the resulting dataset as private",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Hugging Face token; fallback to HUGGINGFACE_HUB_TOKEN or HF_TOKEN",
    )
    p.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Max shard size for push_to_hub",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from datasets import load_dataset, concatenate_datasets
        from huggingface_hub import login
    except Exception as e:
        print("Error: please install required packages: pip install datasets huggingface_hub")
        print(f"Import error: {e}")
        return 1

    token = (
        args.token
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if token:
        try:
            login(token=token)
        except TypeError:
            login(token)  # type: ignore[arg-type]

    print(f"Loading part1: {args.part1} (split={args.split})")
    ds1 = load_dataset(args.part1, split=args.split)
    print(f"Rows in part1: {len(ds1)}")

    print(f"Loading part2: {args.part2} (split={args.split})")
    ds2 = load_dataset(args.part2, split=args.split)
    print(f"Rows in part2: {len(ds2)}")

    print("Concatenating datasets ...")
    combined = concatenate_datasets([ds1, ds2])
    print(f"Combined rows: {len(combined)}")

    print(f"Pushing to {args.output} ...")
    combined.push_to_hub(
        repo_id=args.output,
        private=args.private,
        token=token,
        max_shard_size=args.max_shard_size,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

