#!/usr/bin/env python3

import argparse
import os
from collections.abc import Sequence
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import login


def build_repo_ids(prefix: str, start: int, end: int) -> List[str]:
    return [prefix.format(index=i) for i in range(start, end + 1)]


def collect_datasets(
    repo_ids: Iterable[str],
    split: str,
    cache_dir: Optional[str],
) -> List[Dataset]:
    datasets_list: List[Dataset] = []
    for repo_id in repo_ids:
        ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)
        datasets_list.append(ds)
    return datasets_list


def infer_score_columns(dataset: Dataset) -> List[str]:
    score_columns: List[str] = []
    for column in dataset.column_names:
        if column.startswith(("selection_", "current_")) and column.endswith(
            ("_mean", "_majority")
        ):
            score_columns.append(column)
    return score_columns


def rescale_columns(dataset: Dataset, columns: Iterable[str], divisor: float) -> Dataset:
    columns = list(columns)
    if not columns:
        return dataset

    def _scale(batch):
        for column in columns:
            values = batch[column]
            scaled = []
            for value in values:
                if value is None:
                    scaled.append(None)
                elif isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes)
                ):
                    iter_values = list(value)
                    scaled.append(
                        [None if x is None else x / divisor for x in iter_values]
                    )
                else:
                    scaled.append(value / divisor)
            batch[column] = scaled
        return batch

    return dataset.map(_scale, batched=True)


def _value_has_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_value_has_none(item) for item in value)
    return False


def count_rows_with_none(
    dataset: Dataset, columns: Iterable[str]
) -> tuple[Dict[str, int], int]:
    columns = list(columns)
    counts = {column: 0 for column in columns}
    any_none_rows = 0

    if not columns:
        return counts, any_none_rows

    for row in dataset:
        row_has_none = False
        for column in columns:
            value = row[column]
            if _value_has_none(value):
                counts[column] += 1
                row_has_none = True
        if row_has_none:
            any_none_rows += 1

    return counts, any_none_rows


def _row_is_valid(row, columns: List[str]) -> bool:
    for column in columns:
        if _value_has_none(row[column]):
            return False
    return True


def filter_rows_without_none(dataset: Dataset, columns: Iterable[str]) -> Dataset:
    columns = list(columns)
    if not columns:
        return dataset
    return dataset.filter(_row_is_valid, fn_kwargs={"columns": columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, combine, rescale, and upload Hugging Face datasets."
    )
    parser.add_argument(
        "--repo-prefix",
        default="zjhhhh/iter2_ver2_scores_adversary_{index}",
        help="Format string for source repos. Must contain {index}.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive start index for shards.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=71,
        help="Inclusive end index for shards.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download and merge.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for downloaded datasets.",
    )
    parser.add_argument(
        "--score-columns",
        nargs="*",
        default=None,
        help="Explicit list of columns to rescale. Leave empty to auto-detect.",
    )
    parser.add_argument(
        "--divisor",
        type=float,
        default=4.0,
        help="Value to divide the score columns by.",
    )
    parser.add_argument(
        "--target-repo",
        default="zjhhhh/iter2_ver2_scores_adversary_rescaled",
        help="Repository name for the combined dataset.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload the resulting dataset as a private repository.",
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
        help="Actually push the dataset to the Hugging Face Hub.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if token:
        login(token)

    repo_ids = build_repo_ids(args.repo_prefix, args.start_index, args.end_index)
    datasets_list = collect_datasets(repo_ids, args.split, args.cache_dir)

    if len(datasets_list) == 1:
        combined = datasets_list[0]
    else:
        combined = concatenate_datasets(datasets_list)

    score_columns = (
        args.score_columns
        if args.score_columns
        else infer_score_columns(combined)
    )

    none_counts, total_rows_with_none = count_rows_with_none(combined, score_columns)
    print("Rows containing None values per column:")
    for column, count in none_counts.items():
        print(f"  {column}: {count}")
    print(f"Rows with None in any target column: {total_rows_with_none}")

    if total_rows_with_none > 0:
        combined = filter_rows_without_none(combined, score_columns)
        print(f"Removed {total_rows_with_none} rows containing None values.")
    else:
        print("No rows removed; all target columns contained valid values.")

    combined = rescale_columns(combined, score_columns, args.divisor)

    if total_rows_with_none > 0:
        _, remaining_none_rows = count_rows_with_none(combined, score_columns)
        print(f"Rows with None after filtering: {remaining_none_rows}")

    none_counts, total_rows_with_none = count_rows_with_none(combined, score_columns)
    print("Rows containing None values per column:")
    for column, count in none_counts.items():
        print(f"  {column}: {count}")
    print(f"Rows with None in any target column: {total_rows_with_none}")

    dataset_dict = DatasetDict({args.split: combined})

    if args.push:
        dataset_dict.push_to_hub(
            repo_id=args.target_repo,
            private=args.private,
            token=token,
            max_shard_size=args.max_shard_size,
        )


if __name__ == "__main__":
    main()
