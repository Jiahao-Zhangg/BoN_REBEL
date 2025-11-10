#!/usr/bin/env python3
"""
Prepare zjhhhh/generation_for_iter2 by sampling and renaming response_* columns
from two source datasets and merging them. The first dataset provides
(prompt, requirements, responses); the second provides (prompt, responses).

Sources:
- First:  zjhhhh/Qwen3b_iter1_min_generation -> provides 8 responses per row
          renamed to selection_response_1..4, current_response_1..2,
          adversary_response_1..2
- Second: zjhhhh/Qwen3b                  -> provides 2 responses per row
          renamed to base_response_1..2

Finally merges rows by matching prompt (requirements taken from the first dataset) and optionally pushes
to the Hub as zjhhhh/generation_for_iter2.

Usage example:
  python prepare_generation_for_iter2.py --push \
    --first-repo zjhhhh/Qwen3b_iter1_min_generation \
    --second-repo zjhhhh/Qwen3b \
    --target-repo zjhhhh/generation_for_iter2

Set --seed for reproducibility and pass HF auth via --token or HF_TOKEN env var.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from typing import List

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login


def find_response_columns(columns: List[str]) -> List[str]:
    pat = re.compile(r"^response_\d+$")
    response_cols = [c for c in columns if pat.match(c)]
    # Ensure canonical numeric ordering (response_0, response_1, ...)
    def key(c: str) -> int:
        return int(c.split("_")[-1])

    return sorted(response_cols, key=key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample and rename responses from two datasets and merge into a final dataset."
        )
    )
    parser.add_argument(
        "--first-repo",
        default="zjhhhh/Qwen3b_iter1_multi_generation",
        help="Dataset repo id for selection/current/adversary responses",
    )
    parser.add_argument(
        "--second-repo",
        default="zjhhhh/Qwen3b",
        help="Dataset repo id for base responses",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to use from both datasets (default: train)",
    )
    parser.add_argument(
        "--target-repo",
        default="zjhhhh/generation_for_iter2_multi",
        help="Target repo id to push the merged dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed per-row sampling")
    parser.add_argument(
        "--cache-dir", default=None, help="Optional HF datasets cache directory"
    )
    parser.add_argument(
        "--private", action="store_true", help="Push as a private repository"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token; falls back to HF_TOKEN env var",
    )
    parser.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Max shard size when pushing",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually push to the Hub",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if token:
        login(token)

    # Load datasets
    print(f"Loading first dataset:  {args.first_repo} (split={args.split})")
    ds_first: Dataset = load_dataset(
        args.first_repo, split=args.split, cache_dir=args.cache_dir
    )
    print(f"Loading second dataset: {args.second_repo} (split={args.split})")
    ds_second: Dataset = load_dataset(
        args.second_repo, split=args.split, cache_dir=args.cache_dir
    )

    # Identify response columns
    first_resp_cols = find_response_columns(ds_first.column_names)
    second_resp_cols = find_response_columns(ds_second.column_names)

    # Sanity checks
    need_first = 4 + 2 + 2  # selection 4, current 2, adversary 2
    if len(first_resp_cols) < need_first:
        raise ValueError(
            f"First dataset needs at least {need_first} response_* columns; found {len(first_resp_cols)}"
        )
    if len(second_resp_cols) < 2:
        raise ValueError(
            f"Second dataset needs at least 2 response_* columns; found {len(second_resp_cols)}"
        )

    # Precompute per-row shuffled orders deterministically
    n_first = len(ds_first)
    per_row_first_cols: List[List[str]] = []
    for i in range(n_first):
        r = random.Random(args.seed + i)
        order = first_resp_cols.copy()
        r.shuffle(order)
        per_row_first_cols.append(order[:need_first])

    n_second = len(ds_second)
    per_row_second_cols: List[List[str]] = []
    for i in range(n_second):
        r = random.Random(args.seed * 173 + i)  # different stream for second dataset
        order = second_resp_cols.copy()
        r.shuffle(order)
        per_row_second_cols.append(order[:2])

    # Map first dataset: selection/current/adversary
    def map_first(example, idx):
        sel_cur_adv = per_row_first_cols[idx]
        out = {
            "prompt": example["prompt"],
            "requirements": example["requirements"],
            # selection 4
            "selection_response_1": example[sel_cur_adv[0]],
            "selection_response_2": example[sel_cur_adv[1]],
            "selection_response_3": example[sel_cur_adv[2]],
            "selection_response_4": example[sel_cur_adv[3]],
            # current 2
            "current_response_1": example[sel_cur_adv[4]],
            "current_response_2": example[sel_cur_adv[5]],
            # adversary 2
            "adversary_response_1": example[sel_cur_adv[6]],
            "adversary_response_2": example[sel_cur_adv[7]],
        }
        return out

    ds_first_out = ds_first.map(
        map_first,
        with_indices=True,
        remove_columns=[c for c in ds_first.column_names if c not in ("prompt", "requirements")],
    )

    # Map second dataset: base (note: second dataset may not have 'requirements')
    def map_second(example, idx):
        base_cols = per_row_second_cols[idx]
        out = {
            "prompt": example["prompt"],
            "base_response_1": example[base_cols[0]],
            "base_response_2": example[base_cols[1]],
        }
        return out

    ds_second_out = ds_second.map(
        map_second,
        with_indices=True,
        remove_columns=[c for c in ds_second.column_names if c != "prompt"],
    )

    # Convert to pandas for a clean merge on (prompt, requirements)
    df_first = ds_first_out.to_pandas()
    df_second = ds_second_out.to_pandas()

    # Optional diagnostics for duplicates
    dup1 = df_first.duplicated(subset=["prompt", "requirements"]).sum()
    dup2 = df_second.duplicated(subset=["prompt"]).sum()
    if dup1:
        print(f"Warning: first dataset has {dup1} duplicate (prompt, requirements) rows")
    if dup2:
        print(f"Warning: second dataset has {dup2} duplicate prompt rows")

    keep_first_cols = [
        "prompt",
        "requirements",
        "selection_response_1",
        "selection_response_2",
        "selection_response_3",
        "selection_response_4",
        "current_response_1",
        "current_response_2",
        "adversary_response_1",
        "adversary_response_2",
    ]
    keep_second_cols = [
        "prompt",
        "base_response_1",
        "base_response_2",
    ]

    df_merged = pd.merge(
        df_first[keep_first_cols],
        df_second[keep_second_cols],
        on=["prompt"],
        how="inner",
        validate="many_to_many",  # be permissive but warn above if duplicates
    )

    # Reorder final columns
    final_cols = keep_first_cols + ["base_response_1", "base_response_2"]
    df_merged = df_merged[final_cols]

    print(f"Final merged rows: {len(df_merged)}")

    final_ds = Dataset.from_pandas(df_merged, preserve_index=False)
    dd = DatasetDict({args.split: final_ds})

    if args.push:
        dd.push_to_hub(
            repo_id=args.target_repo,
            private=args.private,
            token=token,
            max_shard_size=args.max_shard_size,
        )
        print(f"Pushed to {args.target_repo}")
    else:
        print("--push not set; skipping upload.")


if __name__ == "__main__":
    main()
