#!/usr/bin/env python3
"""
Combine 26 Hugging Face datasets:
  zjhhhh/Qwen3b_base_0, zjhhhh/Qwen3b_base_2000, ..., zjhhhh/Qwen3b_base_50000

Concatenate them into a single dataset and optionally push to the Hub as:
  zjhhhh/Qwen3b_base_generation

Usage example:
  python combine_qwen3b_base_generation.py --push \
    --target-repo zjhhhh/Qwen3b_base_generation

If authentication is needed, set env var HF_TOKEN or pass --token.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Set

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import login


def build_repo_ids(
    prefix: str = "zjhhhh/Qwen3b_iter1_multi_{value}",
    start: int = 0,
    end: int = 50000,
    step: int = 2000,
) -> List[str]:
    repo_ids: List[str] = []
    value = start
    while value <= end:
        repo_ids.append(prefix.format(value=value))
        value += step
    return repo_ids


def collect_datasets(
    repo_ids: Iterable[str],
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> List[Dataset]:
    datasets_list: List[Dataset] = []
    for repo_id in repo_ids:
        ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)
        datasets_list.append(ds)
    return datasets_list


def _union_columns(datasets_list: List[Dataset]) -> List[str]:
    union: Set[str] = set()
    for ds in datasets_list:
        union.update(ds.column_names)
    # Keep stable ordering: by first appearance across datasets
    ordered: List[str] = []
    seen: Set[str] = set()
    for ds in datasets_list:
        for name in ds.column_names:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    # Append any remaining columns not seen in the first datasets
    for name in sorted(union - set(ordered)):
        ordered.append(name)
    return ordered


def align_to_union_schema(datasets_list: List[Dataset]) -> List[Dataset]:
    """
    Ensure all datasets share the same set of columns by adding missing
    columns filled with None values. This is required for concatenation
    when schemas differ slightly across shards.
    """
    if not datasets_list:
        return datasets_list

    all_columns = _union_columns(datasets_list)
    aligned: List[Dataset] = []
    for ds in datasets_list:
        missing = [c for c in all_columns if c not in ds.column_names]
        if missing:
            for col in missing:
                # Fill with nulls to match dataset length
                ds = ds.add_column(col, [None] * len(ds))
        # Reorder columns to the union order for consistency
        ds = ds.select_columns(all_columns)
        aligned.append(ds)
    return aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Qwen3b_base shards and optionally push to HF."
    )
    parser.add_argument(
        "--source-prefix",
        default="zjhhhh/Qwen3b_iter1_multi_{value}",
        help="Format string for source repos; must include {value}.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start value (inclusive)")
    parser.add_argument("--end", type=int, default=50000, help="End value (inclusive)")
    parser.add_argument("--step", type=int, default=2000, help="Increment between values")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument(
        "--cache-dir", default=None, help="Optional local cache dir for datasets"
    )
    parser.add_argument(
        "--target-repo",
        default="zjhhhh/Qwen3b_iter1_multi_generation",
        help="Target repo id to push the combined dataset",
    )
    parser.add_argument(
        "--private", action="store_true", help="Upload as a private repository"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Defaults to HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Max shard size when uploading the dataset.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the combined dataset to the Hugging Face Hub",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if token:
        login(token)

    repo_ids = build_repo_ids(args.source_prefix, args.start, args.end, args.step)
    print(f"Merging {len(repo_ids)} repos:")
    for rid in repo_ids:
        print(f"  - {rid}")

    datasets_list = collect_datasets(repo_ids, split=args.split, cache_dir=args.cache_dir)

    # Align schemas if needed, then concatenate
    try:
        combined: Dataset
        if len(datasets_list) == 1:
            combined = datasets_list[0]
        else:
            combined = concatenate_datasets(datasets_list)
    except Exception:
        # Fallback: try aligning columns to union schema
        datasets_list = align_to_union_schema(datasets_list)
        if len(datasets_list) == 1:
            combined = datasets_list[0]
        else:
            combined = concatenate_datasets(datasets_list)

    print(f"Combined rows: {len(combined)}; columns: {combined.column_names}")

    dataset_dict = DatasetDict({args.split: combined})

    if args.push:
        dataset_dict.push_to_hub(
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
