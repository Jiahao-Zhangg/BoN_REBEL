#!/usr/bin/env python3
"""
Split a Hugging Face Hub dataset into two equal parts and upload them
as separate datasets.

Defaults:
  - Source: zjhhhh/generation_for_iter2_ver2
  - Targets: zjhhhh/generation_for_iter2_ver2_part1 and _part2

Requirements:
  - pip install datasets huggingface_hub
  - Auth: set HF_TOKEN or HUGGINGFACE_HUB_TOKEN env var (or pass --token)

Examples:
  python split_and_upload_hf_dataset.py \
    --source zjhhhh/generation_for_iter2_ver2 \
    --part1  zjhhhh/generation_for_iter2_ver2_part1 \
    --part2  zjhhhh/generation_for_iter2_ver2_part2 \
    --split train
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a HF dataset into two equal halves and push each half to its own repo",
    )
    p.add_argument(
        "--source",
        default="zjhhhh/generation_for_iter2_multi",
        help="Source dataset repo id (user_or_org/name)",
    )
    p.add_argument(
        "--part1",
        default=None,
        help="Target repo id for the first half. Defaults to <source>_part1",
    )
    p.add_argument(
        "--part2",
        default=None,
        help="Target repo id for the second half. Defaults to <source>_part2",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to load from source (default: train)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the target repos as private",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, reads HUGGINGFACE_HUB_TOKEN or HF_TOKEN",
    )
    p.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Max shard size when pushing to hub (default: 10GB)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy imports to keep CLI responsive
    try:
        from datasets import load_dataset
        from huggingface_hub import login
    except Exception as e:  # pragma: no cover
        print("Error: please install required packages: pip install datasets huggingface_hub")
        print(f"Import error: {e}")
        return 1

    token = (
        args.token
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if token:
        # Ensure authenticated for private source/target repos
        try:
            login(token=token)
        except TypeError:
            # Backward compatibility: older huggingface_hub uses positional token
            login(token)  # type: ignore[arg-type]

    # Derive default target repo ids if not provided
    part1_repo = args.part1 or f"{args.source}_part1"
    part2_repo = args.part2 or f"{args.source}_part2"

    print(f"Loading dataset '{args.source}' split='{args.split}' ...")
    # load_dataset uses cached Arrow storage; this shouldn't fully materialize in RAM
    ds = load_dataset(args.source, split=args.split)
    n = len(ds)
    if n == 0:
        print("Source split is empty; nothing to do.")
        return 0

    # Split into two contiguous halves (as equal as possible)
    mid = n // 2
    # If odd, the second part will contain one extra row to preserve all data
    idx1 = list(range(0, mid))
    idx2 = list(range(mid, n))

    print(f"Dataset size: {n}. Part1: {len(idx1)} rows, Part2: {len(idx2)} rows.")
    ds_part1 = ds.select(idx1)
    ds_part2 = ds.select(idx2)

    now = datetime.utcnow().isoformat() + "Z"

    print(f"Pushing Part1 to '{part1_repo}' ...")
    ds_part1.push_to_hub(
        repo_id=part1_repo,
        private=args.private,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=f"Upload first half of {args.source} ({len(idx1)}/{n}) on {now}",
    )
    print(f"Pushed Part1: {part1_repo}")

    print(f"Pushing Part2 to '{part2_repo}' ...")
    ds_part2.push_to_hub(
        repo_id=part2_repo,
        private=args.private,
        token=token,
        max_shard_size=args.max_shard_size,
        commit_message=f"Upload second half of {args.source} ({len(idx2)}/{n}) on {now}",
    )
    print(f"Pushed Part2: {part2_repo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

