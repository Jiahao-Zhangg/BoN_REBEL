import argparse
import re
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute correlation between (i) objective per base defined as the "
            "average current-vs-base preference score, and (ii) the gap among "
            "selection-vs-base preference scores (max - min), flattened across prompts."
        )
    )
    parser.add_argument("--repo", type=str, required=True, help="HF dataset repo ID to load")
    parser.add_argument("--split", type=str, default=None, help="Split to load; defaults to test then train")
    parser.add_argument(
        "--prefer_mean_columns",
        action="store_true",
        help="Prefer *_judged_preference_mean over *_majority if both exist (default: true)",
    )
    parser.add_argument(
        "--prefer_majority_columns",
        action="store_true",
        help="Prefer *_judged_preference_majority over *_mean if requested",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help=(
            "If set, extract scalar from per-check score vectors using row['j_fixed'] "
            "(as produced by filter_tokenize_judge_fixed.py); otherwise aggregate via mean."
        ),
    )
    return parser.parse_args()


def load_repo(repo_id: str, split: Optional[str]) -> Dataset:
    if split:
        return load_dataset(repo_id, split=split)
    # Try common splits
    try:
        return load_dataset(repo_id, split="test")
    except Exception:
        return load_dataset(repo_id, split="train")


def detect_response_counts(column_names: List[str]) -> Tuple[int, int, int]:
    selection_cols = [c for c in column_names if re.match(r"^selection_response_\d+$", c)]
    current_cols = [c for c in column_names if re.match(r"^current_response_\d+$", c)]
    base_cols = [c for c in column_names if re.match(r"^base_response_\d+$", c)]

    def max_index(cols: List[str]) -> int:
        max_idx = 0
        for name in cols:
            try:
                idx = int(name.split("_")[-1])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                continue
        return max_idx

    return max_index(selection_cols), max_index(current_cols), max_index(base_cols)


def aggregate_numeric(value, *, fixed: bool, fixed_index: Optional[int]) -> Optional[float]:
    # Accept scalar numbers or lists of numbers; ignore non-numerics
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        # If fixed is requested and we have a valid index, extract that coordinate
        if fixed and fixed_index is not None:
            try:
                if 0 <= int(fixed_index) < len(value):
                    return float(value[int(fixed_index)])
            except Exception:
                # Fall through to mean aggregation on failure
                pass
        # Otherwise mean over numeric entries
        nums: List[float] = []
        for v in value:
            try:
                nums.append(float(v))
            except Exception:
                continue
        if len(nums) == 0:
            return None
        return float(np.mean(nums))
    # Try to coerce strings like "3" to float
    try:
        return float(value)
    except Exception:
        return None


def get_best_key(base: int, other: int, kind: str, prefer_majority: bool) -> List[str]:
    # kind in {"selection", "current"}; 1-based indices
    mean_key = f"{kind}_{other}_base_{base}_mean"
    maj_key = f"{kind}_{other}_base_{base}_majority"
    return [maj_key, mean_key] if prefer_majority else [mean_key, maj_key]


def main() -> None:
    args = parse_args()
    prefer_majority = bool(args.prefer_majority_columns and not args.prefer_mean_columns)

    ds = load_repo(args.repo, args.split)
    # ds = ds.select(range(100))
    n_selection, n_current, n_base = detect_response_counts(list(ds.column_names))
    if n_base == 0:
        raise RuntimeError("No base_response_* columns detected; dataset schema mismatch with preprocess_common.py")

    # Accumulate across all prompts and base indices
    objective_values: List[float] = []  # average current-vs-base score per base
    gap_values: List[float] = []        # (max selection-vs-base) - (min selection-vs-base) per base
    best_base_row_gaps: List[float] = []  # per-row: gap for base with minimal objective
    worst_base_row_gaps: List[float] = [] # per-row: gap for base with maximal objective
    min_selection_gaps: List[float] = []  # per-row: min gap across bases
    max_selection_gaps: List[float] = []  # per-row: max gap across bases
    g_diff_values: List[float] = []       # per-row: g_chosen - g_reject if available
    per_row_selection_mean_gaps: List[float] = []  # per-row: gap across selection means over bases

    for row in ds:
        # Optional: collect row-level g difference if present
        if "g_chosen" in row and "g_reject" in row:
            try:
                g_diff_values.append(float(row["g_chosen"]) - float(row["g_reject"]))
            except Exception:
                pass
        # Track per-row objectives and selection gaps for each base to compute best-base metric
        objectives_for_bases: List[Optional[float]] = []
        selection_gaps_for_bases: List[Optional[float]] = []

        for base_idx in range(1, n_base + 1):
            # Aggregate current-vs-base objective (average across k in current)
            current_scores: List[float] = []
            for cur_idx in range(1, n_current + 1):
                for key in get_best_key(base_idx, cur_idx, kind="current", prefer_majority=prefer_majority):
                    if key in row:
                        j_fixed = None
                        if args.fixed and "j_fixed" in row and row["j_fixed"] is not None:
                            try:
                                j_fixed = int(row["j_fixed"])  # already 0-based index from producer
                            except Exception:
                                j_fixed = None
                        val = aggregate_numeric(row.get(key), fixed=args.fixed, fixed_index=j_fixed)
                        if val is not None:
                            current_scores.append(val)
                        break
            objective_for_base: Optional[float] = float(np.mean(current_scores)) if len(current_scores) > 0 else None
            # Aggregate selection-vs-base gap (max - min across i in selection)
            selection_scores: List[float] = []
            for sel_idx in range(1, n_selection + 1):
                for key in get_best_key(base_idx, sel_idx, kind="selection", prefer_majority=prefer_majority):
                    if key in row:
                        j_fixed = None
                        if args.fixed and "j_fixed" in row and row["j_fixed"] is not None:
                            try:
                                j_fixed = int(row["j_fixed"])  # already 0-based index from producer
                            except Exception:
                                j_fixed = None
                        val = aggregate_numeric(row.get(key), fixed=args.fixed, fixed_index=j_fixed)
                        if val is not None:
                            selection_scores.append(val)
                        break
            gap_for_base: Optional[float] = (
                (max(selection_scores) - min(selection_scores)) if len(selection_scores) >= 2 else None
            )

            if objective_for_base is not None and gap_for_base is not None:
                objective_values.append(objective_for_base)
                gap_values.append(gap_for_base)

            # Save per-row values for best-base computation
            objectives_for_bases.append(objective_for_base)
            selection_gaps_for_bases.append(gap_for_base)

        # Per-row: choose best base as the one with minimal objective; record its selection gap
        valid_objectives = [(idx, obj) for idx, obj in enumerate(objectives_for_bases, start=1) if obj is not None]
        if valid_objectives:
            best_idx, _ = min(valid_objectives, key=lambda t: t[1])
            best_gap = selection_gaps_for_bases[best_idx - 1]
            if best_gap is not None:
                best_base_row_gaps.append(best_gap)
            # Worst-base: choose base with maximal objective
            worst_idx, _ = max(valid_objectives, key=lambda t: t[1])
            worst_gap = selection_gaps_for_bases[worst_idx - 1]
            if worst_gap is not None:
                worst_base_row_gaps.append(worst_gap)

        # Row-level min/max selection gap across bases
        valid_gaps = [g for g in selection_gaps_for_bases if g is not None]
        if len(valid_gaps) > 0:
            min_selection_gaps.append(min(valid_gaps))
            max_selection_gaps.append(max(valid_gaps))

        # New metric: for each row, average each selection response over bases, then gap across selections
        selection_means: List[float] = []
        for sel_idx in range(1, n_selection + 1):
            base_scores: List[float] = []
            for base_idx in range(1, n_base + 1):
                for key in get_best_key(base_idx, sel_idx, kind="selection", prefer_majority=prefer_majority):
                    if key in row:
                        j_fixed = None
                        if args.fixed and "j_fixed" in row and row["j_fixed"] is not None:
                            try:
                                j_fixed = int(row["j_fixed"])  # 0-based index
                            except Exception:
                                j_fixed = None
                        val = aggregate_numeric(row.get(key), fixed=args.fixed, fixed_index=j_fixed)
                        if val is not None:
                            base_scores.append(val)
                        break
            if len(base_scores) > 0:
                selection_means.append(float(np.mean(base_scores)))
        if len(selection_means) >= 2:
            per_row_selection_mean_gaps.append(max(selection_means) - min(selection_means))

    if len(objective_values) == 0 or len(gap_values) == 0:
        print("No comparable numeric values found to compute correlation. Check dataset columns and judge outputs.")
        return

    if len(objective_values) != len(gap_values):
        # Defensive: align lengths if any discrepancy slipped through
        n = min(len(objective_values), len(gap_values))
        objective_values = objective_values[:n]
        gap_values = gap_values[:n]

    x = np.asarray(objective_values, dtype=float)
    y = np.asarray(gap_values, dtype=float)

    if x.size < 2:
        print("Not enough points to compute correlation (need at least 2).")
        return

    corr_matrix = np.corrcoef(x, y)
    pearson_r = float(corr_matrix[0, 1])

    print(f"Samples used: {x.size}")
    print(f"Objective (current-vs-base avg) mean: {float(np.mean(x)):.6f} std: {float(np.std(x)):.6f}")
    print(f"Selection gap (max-min) mean: {float(np.mean(y)):.6f} std: {float(np.std(y)):.6f}")
    print(f"Pearson correlation r: {pearson_r:.6f}")

    # Additional metric: per-row best-base selection gap (mean across rows)
    if len(best_base_row_gaps) > 0:
        bb_arr = np.asarray(best_base_row_gaps, dtype=float)
        print(f"Best-base selection gap: mean: {float(np.mean(bb_arr)):.6f} std: {float(np.std(bb_arr)):.6f} rows: {bb_arr.size}")
    if len(worst_base_row_gaps) > 0:
        wb_arr = np.asarray(worst_base_row_gaps, dtype=float)
        print(f"Worst-base selection gap: mean: {float(np.mean(wb_arr)):.6f} std: {float(np.std(wb_arr)):.6f} rows: {wb_arr.size}")
    if len(min_selection_gaps) > 0:
        min_arr = np.asarray(min_selection_gaps, dtype=float)
        print(f"Min selection gap across bases: mean: {float(np.mean(min_arr)):.6f} std: {float(np.std(min_arr)):.6f} rows: {min_arr.size}")
    if len(max_selection_gaps) > 0:
        max_arr = np.asarray(max_selection_gaps, dtype=float)
        print(f"Max selection gap across bases: mean: {float(np.mean(max_arr)):.6f} std: {float(np.std(max_arr)):.6f} rows: {max_arr.size}")

    # Additional metric: average of (g_chosen - g_reject) if available
    if len(g_diff_values) > 0:
        gd_arr = np.asarray(g_diff_values, dtype=float)
        print(f"Average g_chosen - g_reject: mean: {float(np.mean(gd_arr)):.6f} std: {float(np.std(gd_arr)):.6f} rows: {gd_arr.size}")

    # New summary: per-row gap across selection means over bases
    if len(per_row_selection_mean_gaps) > 0:
        pr_arr = np.asarray(per_row_selection_mean_gaps, dtype=float)
        print(
            f"Per-row gap of selection means over bases: mean: {float(np.mean(pr_arr)):.6f} std: {float(np.std(pr_arr)):.6f} rows: {pr_arr.size}"
        )


if __name__ == "__main__":
    main()

