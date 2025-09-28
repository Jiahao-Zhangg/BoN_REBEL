#!/usr/bin/env python3
"""Filter a Hugging Face dataset by the reward gap and push the result to the hub.

This script downloads ``zjhhhh/whole_sw_maxlen_8192_rescale_mean_beta_1.0_fixed_expand_tokenized``,
keeps the ``test`` split unchanged, and selects the top 40% of ``train`` rows with the
largest ``g_chosen - g_reject`` gap. The filtered dataset is then uploaded to
``zjhhhh/choose_gap_beta_1_tokenized``.

The script can be re-used with custom dataset names and ratios through CLI flags.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import whoami

DEFAULT_SOURCE_DATASET = "zjhhhh/whole_sw_maxlen_8192_nocheck_rescale_mean_beta_10.0_nocheck_expand_tokenized"
DEFAULT_TARGET_DATASET = "zjhhhh/choose_gap_beta_10_nocheck_tokenized"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_RATIO = 0.4
DEFAULT_SHUFFLE_SEED = None
GCHOSEN_COL = "g_chosen"
GREJECT_COL = "g_reject"
GAP_COL = "_gap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter train split by (g_chosen - g_reject) gap and push to hub")
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET, help="Dataset to download from the hub")
    parser.add_argument("--target-dataset", default=DEFAULT_TARGET_DATASET, help="Repository to push the filtered dataset")
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT, help="Name of the train split to filter")
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO, help="Fraction of rows to keep (0 < ratio <= 1)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the final push_to_hub call (useful for local inspection)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help="Seed used when shuffling the filtered train split (default: no seed)",
    )
    return parser.parse_args()


def require_authentication() -> None:
    try:
        info = whoami()
        print(f"✓ Authenticated as: {info['name']}")
    except Exception as exc:  # pragma: no cover - huggingface-cli handles auth
        print(f"❌ Not authenticated: {exc}")
        print("Run `huggingface-cli login` or set the HF_TOKEN environment variable.")
        raise SystemExit(1)


def compute_gap_filtered_dataset(train_split: Dataset, keep_ratio: float, shuffle_seed: int | None) -> Dataset:
    if GCHOSEN_COL not in train_split.column_names:
        raise ValueError(f"Missing required column '{GCHOSEN_COL}' in train split")
    if GREJECT_COL not in train_split.column_names:
        raise ValueError(f"Missing required column '{GREJECT_COL}' in train split")
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("--ratio must satisfy 0 < ratio <= 1")

    print(
        f"📊 Filtering train split with {len(train_split)} rows; keeping top {keep_ratio:.0%} by gap"
    )

    with_gap = train_split.map(lambda row: {GAP_COL: row[GCHOSEN_COL] - row[GREJECT_COL]})
    sorted_by_gap = with_gap.sort(GAP_COL, reverse=True)
    keep_count = max(1, math.floor(len(sorted_by_gap) * keep_ratio))
    print(f"📈 Selected top {keep_count} rows out of {len(sorted_by_gap)}")
    filtered = sorted_by_gap.select(range(keep_count)).remove_columns([GAP_COL])
    shuffle_msg = "randomising order" if shuffle_seed is None else f"shuffling with seed {shuffle_seed}"
    print(f"🔀 {shuffle_msg}")
    return filtered.shuffle(seed=shuffle_seed)


def build_new_dataset(dataset: DatasetDict, train_split: str, filtered_train: Dataset) -> DatasetDict:
    updated_splits: Dict[str, Dataset] = {}
    for split_name, split_data in dataset.items():
        if split_name == train_split:
            updated_splits[split_name] = filtered_train
        else:
            updated_splits[split_name] = split_data
    return DatasetDict(updated_splits)


def main() -> None:
    args = parse_args()
    require_authentication()

    print(f"📥 Downloading dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset)
    if args.train_split not in dataset:
        raise ValueError(f"Split '{args.train_split}' not found in dataset {args.source_dataset}")

    filtered_train = compute_gap_filtered_dataset(dataset[args.train_split], args.ratio, args.shuffle_seed)
    new_dataset = build_new_dataset(dataset, args.train_split, filtered_train)

    print(f"📦 Dataset ready with splits: {list(new_dataset.keys())}")
    if not args.dry_run:
        print(f"📤 Uploading filtered dataset to: {args.target_dataset}")
        new_dataset.push_to_hub(args.target_dataset)
        print("✅ Upload finished")
    else:
        print("🚫 Dry run enabled; skipping push_to_hub")


if __name__ == "__main__":
    main()
