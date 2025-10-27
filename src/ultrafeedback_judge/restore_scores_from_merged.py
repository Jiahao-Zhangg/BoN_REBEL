#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset


def _detect_base_and_model_counts(column_names: List[str]) -> Tuple[int, int]:
    base_cols = [c for c in column_names if c.startswith("base_response_")]
    model_cols = [c for c in column_names if c.startswith("model_response_")]
    # Robustly parse trailing indices
    def _idx(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:
            return -1
    n_base = 0
    if base_cols:
        n_base = 1 + max(_idx(c) for c in base_cols)
    n_model = 0
    if model_cols:
        n_model = 1 + max(_idx(c) for c in model_cols)
    return n_base, n_model


def _max_checks_for_prompt(row: Dict[str, object], n_model: int, n_base: int) -> int:
    max_len = 0
    for ai in range(n_model):
        for bj in range(n_base):
            for suffix in ("_mean", "_majority"):
                key = f"judge_{ai}_{bj}{suffix}"
                vals = row.get(key)
                if isinstance(vals, list):
                    max_len = max(max_len, len(vals))
    return max_len


def _value_at(lst: Optional[List], idx: int):
    if not isinstance(lst, list):
        return None
    if idx < 0 or idx >= len(lst):
        return None
    return lst[idx]


def score_merged_dataset(
    ds: Dataset,
    beta: float,
    n_response_base: Optional[int] = None,
    n_response_model: Optional[int] = None,
) -> float:
    column_names = list(ds.column_names)
    detected_base, detected_model = _detect_base_and_model_counts(column_names)

    # Determine effective counts (use provided limits if any; otherwise detected)
    n_base = detected_base if n_response_base is None else max(0, min(detected_base, n_response_base))
    n_model = detected_model if n_response_model is None else max(0, min(detected_model, n_response_model))

    if n_base == 0 or n_model == 0:
        return float("nan")

    # Group by prompt (merged datasets usually have one row per prompt)
    grouped: Dict[str, List[dict]] = {}
    for row in ds:
        p = row["prompt"]
        grouped.setdefault(p, []).append(row)

    prompt_vals: List[float] = []
    for p, rows in grouped.items():
        # There should typically be a single merged row per prompt
        merged_row = rows[0]
        num_checks = _max_checks_for_prompt(merged_row, n_model, n_base)
        if num_checks <= 0:
            continue

        check_objectives: List[float] = []
        for check_idx in range(num_checks):
            base_exp_terms: List[float] = []
            for bj in range(n_base):
                vals: List[float] = []
                for ai in range(n_model):
                    mean_key = f"judge_{ai}_{bj}_mean"
                    maj_key = f"judge_{ai}_{bj}_majority"
                    mean_list = merged_row.get(mean_key)
                    maj_list = merged_row.get(maj_key)
                    v = _value_at(mean_list, check_idx)
                    if v is None:
                        v = _value_at(maj_list, check_idx)
                    if v is None:
                        continue
                    try:
                        vals.append(float(v) / 4.0)
                    except Exception:
                        continue
                if vals:
                    expected_score = sum(vals) / len(vals)
                    base_exp_terms.append(math.exp(-expected_score / beta))
            if not base_exp_terms:
                continue
            avg_exp = sum(base_exp_terms) / len(base_exp_terms)
            if avg_exp <= 0:
                continue
            check_objectives.append(-beta * math.log(avg_exp))
        if check_objectives:
            prompt_vals.append(min(check_objectives))

    if not prompt_vals:
        return float("nan")
    return sum(prompt_vals) / len(prompt_vals)


def mean_of_all_mean_columns(
    ds: Dataset,
    n_response_base: Optional[int] = None,
    n_response_model: Optional[int] = None,
) -> float:
    """Compute the arithmetic mean of all numeric entries across all
    judge_{ai}_{bj}_mean columns in the dataset.

    Returns NaN if no numeric entries are found or if base/model counts cannot be detected.
    """
    column_names = list(ds.column_names)
    detected_base, detected_model = _detect_base_and_model_counts(column_names)

    n_base = detected_base if n_response_base is None else max(0, min(detected_base, n_response_base))
    n_model = detected_model if n_response_model is None else max(0, min(detected_model, n_response_model))

    if n_base == 0 or n_model == 0:
        return float("nan")

    value_sum: float = 0.0
    value_count: int = 0

    for row in ds:
        for ai in range(n_model):
            for bj in range(n_base):
                mean_key = f"judge_{ai}_{bj}_mean"
                mean_list = row.get(mean_key)
                if not isinstance(mean_list, list):
                    continue
                for v in mean_list:
                    try:
                        value_sum += float(v)
                        value_count += 1
                    except Exception:
                        # Skip non-numeric entries
                        continue

    if value_count == 0:
        return float("nan")
    return value_sum / value_count


def load_merged_dataset(repo_id: str) -> Dataset:
    try:
        return load_dataset(repo_id, split="test")
    except Exception:
        return load_dataset(repo_id, split="train")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_repos", type=str, nargs='+', required=True, help="One or more merged dataset repos (pushed via push_to_hub)")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--score_json_path", type=str, default=None, help="Optional path for the JSON score summary output")
    parser.add_argument("--n_response_base", type=int, default=None, help="Optional: only consider the first N base responses")
    parser.add_argument("--n_response_model", type=int, default=None, help="Optional: only consider the first N model responses")
    return parser.parse_args()


def main():
    args = parse_args()
    scores: List[float] = []
    for repo_id in args.merged_repos:
        ds = load_merged_dataset(repo_id)
        s = score_merged_dataset(ds, args.beta, args.n_response_base, args.n_response_model)
        print(f"Score for {repo_id}: {s}")
        global_mean_mean = mean_of_all_mean_columns(ds, args.n_response_base, args.n_response_model)
        print(f"Mean of all judge_*_mean values for {repo_id}: {global_mean_mean}")
        scores.append(s)

    out_path = Path(args.score_json_path) if args.score_json_path else Path("restored_scores.json")
    payload = {"repos": args.merged_repos, "scores": scores, "beta": args.beta}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved restored score summary to {out_path}")


if __name__ == "__main__":
    main()


