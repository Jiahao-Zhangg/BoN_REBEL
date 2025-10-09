#!/usr/bin/env python3
"""
Download a list of HF datasets, compute g_chosen - g_reject margins, and
produce a correlation matrix across the datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

import numpy as np

import pandas as pd
from datasets import Dataset, load_dataset

try:
    import importlib

    features_module = importlib.import_module("datasets.features.features")
    Sequence = getattr(features_module, "Sequence", None)
    feature_types = getattr(features_module, "_FEATURE_TYPES", None)
    if Sequence is not None and feature_types is not None and "List" not in feature_types:
        feature_types["List"] = Sequence
except Exception:
    Sequence = None  # Fallback: leave unpatched if module structure changes.

DEFAULT_DATASETS: List[str] = [
    "zjhhhh/choose_gap_min_ver2_tokenized",
    "zjhhhh/baseline2_tokenized",
    "zjhhhh/choose_gap_beta_1_multi_tokenized",
    "zjhhhh/choose_gap_beta_10_multi_tokenized",
    "zjhhhh/choose_gap_beta_0.1_multi_tokenized",
    "zjhhhh/choose_gap_beta_1_nocheck_tokenized",
]

JOIN_KEY_CANDIDATES: List[str] = [
    "prompt_id",
    "conversation_id",
    "dialog_id",
    "example_id",
    "question_id",
    "id",
    "prompt",
]

EXPLICIT_LABELS = {
    "zjhhhh/choose_gap_min_ver2_tokenized": "Nogame",
    "zjhhhh/baseline2_tokenized": "Baseline",
    "zjhhhh/choose_gap_beta_1_multi_tokenized": "Multi-beta-1",
    "zjhhhh/choose_gap_beta_10_multi_tokenized": "Multi-beta-10",
    "zjhhhh/choose_gap_beta_0.1_multi_tokenized": "Multi-beta-0.1",
    "zjhhhh/choose_gap_beta_1_nocheck_tokenized": "Nocheck",
}

MARGIN_COLUMN_OVERRIDES = {
    "zjhhhh/baseline2_tokenized": ("chosen_reward", "reject_reward"),
}


def format_display_label(dataset_name: str) -> str:
    if dataset_name in EXPLICIT_LABELS:
        return EXPLICIT_LABELS[dataset_name]

    lowered = dataset_name.lower()
    if "nocheck" in lowered:
        return "Nocheck"
    if "baseline" in lowered:
        return "Baseline"
    if "min_ver2" in lowered or "min_expand_ver2" in lowered:
        return "Nogame"

    beta_match = re.search(r"beta[_-]?([0-9]+(?:\.[0-9]+)?)_multi", lowered)
    if beta_match:
        beta_str = beta_match.group(1)
        return f"Multi-beta-{beta_str}"

    return dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets, build g_chosen - g_reject margins, and compute their correlation."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="List of datasets to load from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--join-key",
        default=None,
        help="Column name shared by all datasets to align rows before computing correlation. "
        "If omitted, the script looks for a common key automatically. If none found, row order is used.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading datasets (needed for some custom builders).",
    )
    parser.add_argument(
        "--corr-csv",
        default="outputs/choose_gap_margin_correlation.csv",
        help="Where to write the correlation matrix as CSV. Set to '' to skip writing.",
    )
    parser.add_argument(
        "--margins-csv",
        default="",
        help="Optional path to save the aligned margin values used for the correlation.",
    )
    parser.add_argument(
        "--corr-plot",
        default="outputs/choose_gap_margin_correlation.png",
        help=(
            "Optional file path (or prefix when multiple methods) for correlation heatmaps. "
            "Set to '' to skip plotting."
        ),
    )
    parser.add_argument(
        "--corr-methods",
        nargs="+",
        default=["pearson", "spearman", "kendall"],
        choices=["pearson", "spearman", "kendall"],
        help="Correlation methods to compute. Defaults to all three.",
    )
    parser.add_argument(
        "--topk-file",
        default="outputs/choose_gap_top9.txt",
        help="Path to write the top-9 margin values per dataset. Set to '' to skip.",
    )
    parser.add_argument(
        "--density-plot",
        default="outputs/choose_gap_density.png",
        help=(
            "Optional file path (or prefix when multiple pairs) for margin density curves. "
            "Set to '' to skip plotting."
        ),
    )
    parser.add_argument(
        "--density-pairs",
        nargs="+",
        default=["Nogame,Multi-beta-1.0", "Multi-beta-1.0,Nocheck"],
        help=(
            "Pairs of column labels (comma separated) to plot as PDF curves. "
            "Example: 'Nogame,Multi-beta-1.0'."
        ),
    )
    parser.add_argument(
        "--difference-plot",
        default="outputs/choose_gap_difference.png",
        help=(
            "Optional path to save line plots of margin differences between dataset pairs. "
            "Set to '' to skip."
        ),
    )
    parser.add_argument(
        "--difference-pairs",
        nargs="+",
        default=["Multi-beta-1.0,Nogame", "Multi-beta-1.0,Nocheck"],
        help=(
            "Pairs of column labels (comma separated) whose margin differences will be plotted "
            "as base-minus-comparison curves."
        ),
    )
    return parser.parse_args()


def download_datasets(
    dataset_names: Iterable[str],
    split: str,
    trust_remote_code: bool,
) -> Dict[str, Dataset]:
    datasets: Dict[str, Dataset] = {}
    for name in dataset_names:
        print(f"Loading dataset '{name}' (split='{split}')")
        try:
            ds = load_dataset(name, split=split, trust_remote_code=trust_remote_code)
        except ValueError as exc:
            if not trust_remote_code and "Feature type 'List'" in str(exc):
                raise ValueError(
                    f"Failed to load '{name}' because the dataset uses custom feature types. "
                    "Re-run with --trust-remote-code to enable loading remote code."
                ) from exc
            raise
        datasets[name] = ds
        print(f"  -> {len(ds):,} rows, {len(ds.column_names)} columns")
    return datasets


def infer_join_key(datasets: Dict[str, Dataset], explicit_key: Optional[str]) -> Optional[str]:
    if explicit_key:
        missing = [name for name, ds in datasets.items() if explicit_key not in ds.column_names]
        if missing:
            raise ValueError(
                f"Join key '{explicit_key}' missing from datasets: {', '.join(missing)}"
            )
        print(f"Using user-specified join key: '{explicit_key}'")
        return explicit_key

    for candidate in JOIN_KEY_CANDIDATES:
        if all(candidate in ds.column_names for ds in datasets.values()):
            print(f"Detected join key '{candidate}' present in all datasets.")
            return candidate

    print("No shared join key detected; falling back to row index alignment.")
    return None


def margin_series_from_dataset(
    dataset: Dataset,
    dataset_name: str,
    join_key: Optional[str],
) -> pd.Series:
    chosen_col, reject_col = MARGIN_COLUMN_OVERRIDES.get(
        dataset_name, ("g_chosen", "g_reject")
    )

    required = {chosen_col, reject_col}
    missing = required.difference(dataset.column_names)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset '{dataset_name}' is missing required columns: {missing_cols}")

    columns_to_pull = [chosen_col, reject_col]
    if join_key:
        columns_to_pull.append(join_key)

    df = dataset.select_columns(columns_to_pull).to_pandas()
    df[chosen_col] = pd.to_numeric(df[chosen_col], errors="coerce")
    df[reject_col] = pd.to_numeric(df[reject_col], errors="coerce")
    df = df.dropna(subset=[chosen_col, reject_col])
    df["margin"] = df[chosen_col] - df[reject_col]

    if join_key:
        # Order duplicates consistently: sort each prompt's margins ascending, then index them.
        df = df.sort_values([join_key, "margin"], kind="mergesort").reset_index(drop=True)
        # Preserve duplicate prompts by tagging each occurrence with its position.
        df["_dup_index"] = df.groupby(join_key).cumcount()
        margin_series = df.set_index([join_key, "_dup_index"])["margin"]
    else:
        margin_series = df["margin"].reset_index(drop=True)

    display_name = format_display_label(dataset_name)
    margin_series.name = display_name
    key_desc = f"{join_key} + duplicate index" if join_key else "index"
    print(
        f"Prepared margin series for '{dataset_name}' (label '{display_name}'): "
        f"{len(margin_series):,} aligned rows (alignment key: {key_desc})"
    )
    return margin_series


def main() -> None:
    args = parse_args()

    datasets = download_datasets(args.datasets, args.split, args.trust_remote_code)
    join_key = infer_join_key(datasets, args.join_key)

    margin_series_list = [
        margin_series_from_dataset(ds, name, join_key) for name, ds in datasets.items()
    ]

    # Use outer join; correlations are computed pairwise using overlapping rows.
    aligned_margins = pd.concat(margin_series_list, axis=1, join="outer")
    aligned_margins = aligned_margins.dropna(how="all")
    print(
        "Aligned margin matrix shape: "
        f"{aligned_margins.shape} (after dropping rows with all-missing margins)"
    )

    if aligned_margins.empty:
        raise RuntimeError(
            "Aligned margin matrix is empty. Ensure datasets contain at least one overlapping key."
        )

    presence_mask = aligned_margins.notna()
    coverage = presence_mask.sum()
    print("\nNon-null margin counts per dataset (post-alignment):")
    for dataset_name, count in coverage.items():
        print(f"  {dataset_name}: {count:,}")

    overlap_counts = presence_mask.astype(int).T.dot(presence_mask.astype(int))
    print("\nPairwise overlap counts:")
    print(overlap_counts)

    corr_results = {}
    for method in args.corr_methods:
        corr_matrix = aligned_margins.corr(method=method)
        corr_results[method] = corr_matrix
        print(f"\nCorrelation matrix ({method}):")
        print(corr_matrix)

        if args.corr_csv:
            output_path = Path(args.corr_csv)
            if len(args.corr_methods) == 1:
                method_path = output_path
            else:
                method_path = output_path.with_name(f"{output_path.stem}_{method}{output_path.suffix}")
            method_path.parent.mkdir(parents=True, exist_ok=True)
            corr_matrix.to_csv(method_path, float_format="%.6f")
            print(f"Saved {method} correlation matrix to '{method_path}'.")

    if args.margins_csv:
        margins_path = Path(args.margins_csv)
        margins_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_margins.to_csv(margins_path, index=True, float_format="%.6f")
        print(f"Saved aligned margins to '{margins_path}'.")

    if args.topk_file:
        topk_path = Path(args.topk_file)
        topk_path.parent.mkdir(parents=True, exist_ok=True)
        with topk_path.open("w", encoding="utf-8") as handle:
            for column in aligned_margins.columns:
                series = aligned_margins[column].dropna().sort_values(ascending=False)
                top_values = series.head(9).tolist()
                handle.write(f"{column}:\n")
                for idx, value in enumerate(top_values, start=1):
                    handle.write(f"  {idx}. {value:.6f}\n")
                handle.write("\n")
        print(f"Wrote top-9 margin values per dataset to '{topk_path}'.")

    if args.density_plot:
        try:
            import matplotlib.pyplot as plt

            base_path = Path(args.density_plot)
            base_path.parent.mkdir(parents=True, exist_ok=True)

            color_defaults = plt.rcParams.get("axes.prop_cycle")
            base_colors = color_defaults.by_key().get("color", []) if color_defaults else []
            if len(base_colors) < 2:
                base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

            pairs: List[tuple[str, str]] = []
            for raw_pair in args.density_pairs:
                parts = [p.strip() for p in raw_pair.split(",") if p.strip()]
                if len(parts) != 2:
                    print(f"Ignoring density pair '{raw_pair}' (expected two comma-separated labels).")
                    continue
                pairs.append((parts[0], parts[1]))

            if not pairs:
                print("No valid density pairs specified; skipping density plots.")
            else:
                for pair_index, (label_a, label_b) in enumerate(pairs):
                    missing = [label for label in (label_a, label_b) if label not in aligned_margins.columns]
                    if missing:
                        print(
                            f"Skipping density plot for '{label_a}' vs '{label_b}' "
                            f"(missing columns: {', '.join(missing)})."
                        )
                        continue

                    fig, ax = plt.subplots(figsize=(8, 5))
                    plotted_any = False
                    for idx, label in enumerate((label_a, label_b)):
                        values = aligned_margins[label].dropna().to_numpy()
                        if values.size < 2:
                            print(
                                f"Skipping curve for '{label}' in pair '{label_a}' vs '{label_b}' "
                                "because it has fewer than two values."
                            )
                            continue
                        color = base_colors[idx % len(base_colors)]
                        counts, bin_edges = np.histogram(values, bins=256, density=True)
                        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        ax.plot(centers, counts, label=label, color=color, linewidth=1.5)
                        plotted_any = True

                    if plotted_any:
                        ax.set_xlabel("g_chosen - g_reject margin")
                        ax.set_ylabel("Estimated PDF")
                        ax.set_title(f"Margin Distribution (Estimated PDF): {label_a} vs {label_b}")
                        ax.grid(alpha=0.2)
                        ax.legend()
                        suffix = f"{re.sub(r'[^A-Za-z0-9]+', '_', label_a).strip('_')}_vs_{re.sub(r'[^A-Za-z0-9]+', '_', label_b).strip('_')}"
                        if len(pairs) == 1:
                            density_path = base_path
                        else:
                            density_path = base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")
                        fig.tight_layout()
                        fig.savefig(density_path, dpi=200)
                        plt.close(fig)
                        print(f"Saved margin density plot for '{label_a}' vs '{label_b}' to '{density_path}'.")
                    else:
                        plt.close(fig)
                        print(
                            f"No valid curves to plot for '{label_a}' vs '{label_b}'. Skipping figure."
                        )
        except ImportError as exc:
            print(f"Skipping density plot because matplotlib is unavailable: {exc}")

    if args.difference_plot:
        try:
            import matplotlib.pyplot as plt

            diff_pairs: List[tuple[str, str]] = []
            for raw_pair in args.difference_pairs:
                parts = [p.strip() for p in raw_pair.split(",") if p.strip()]
                if len(parts) != 2:
                    print(f"Ignoring difference pair '{raw_pair}' (expected two comma-separated labels).")
                    continue
                diff_pairs.append((parts[0], parts[1]))

            if not diff_pairs:
                print("No valid difference pairs specified; skipping difference plot.")
            else:
                color_defaults = plt.rcParams.get("axes.prop_cycle")
                base_colors = color_defaults.by_key().get("color", []) if color_defaults else []
                if len(base_colors) < len(diff_pairs):
                    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

                fig, ax = plt.subplots(figsize=(9, 5))
                plotted_any = False
                for idx, (base_label, compare_label) in enumerate(diff_pairs):
                    missing = [
                        label
                        for label in (base_label, compare_label)
                        if label not in aligned_margins.columns
                    ]
                    if missing:
                        print(
                            f"Skipping difference curve '{base_label} - {compare_label}' "
                            f"(missing columns: {', '.join(missing)})."
                        )
                        continue

                    diff_series = (aligned_margins[base_label] - aligned_margins[compare_label]).dropna()
                    if diff_series.empty:
                        print(
                            f"Skipping difference curve '{base_label} - {compare_label}' "
                            "because no overlapping rows were found."
                        )
                        continue

                    sorted_diff = np.sort(diff_series.to_numpy())
                    x = np.arange(sorted_diff.size)
                    color = base_colors[idx % len(base_colors)] if base_colors else None
                    ax.plot(
                        x,
                        sorted_diff,
                        label=f"{base_label} - {compare_label}",
                        color=color,
                        linewidth=1.5,
                    )
                    plotted_any = True

                if plotted_any:
                    ax.set_xlabel("Aligned prompt index (sorted by margin difference)")
                    ax.set_ylabel("Margin difference")
                    ax.set_title("Margin Difference Curves")
                    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.4)
                    ax.grid(alpha=0.2)
                    ax.legend()
                    diff_path = Path(args.difference_plot)
                    diff_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.tight_layout()
                    fig.savefig(diff_path, dpi=200)
                    plt.close(fig)
                    print(f"Saved margin difference plot to '{diff_path}'.")
                else:
                    plt.close(fig)
                    print("No margin difference curves were generated; skipping figure.")
        except ImportError as exc:
            print(f"Skipping difference plot because matplotlib is unavailable: {exc}")

    if args.corr_plot:
        try:
            import matplotlib.pyplot as plt

            base_path = Path(args.corr_plot)
            base_path.parent.mkdir(parents=True, exist_ok=True)

            for method, matrix in corr_results.items():
                fig, ax = plt.subplots(figsize=(max(6, matrix.shape[0] * 1.1), 6))
                data = matrix.to_numpy()
                im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0)
                ax.set_xticks(range(len(matrix.columns)))
                ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(matrix.index)))
                ax.set_yticklabels(matrix.index)
                ax.set_title(f"Correlation of g_chosen - g_reject margins ({method.title()})")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Correlation")

                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        value = data[i, j]
                        if pd.isna(value):
                            label = "NaN"
                        else:
                            label = f"{value:.2f}"
                        ax.text(
                            j,
                            i,
                            label,
                            ha="center",
                            va="center",
                            color="black" if pd.isna(value) or abs(value) < 0.5 else "white",
                            fontsize=8,
                        )

                fig.tight_layout()
                if len(corr_results) == 1:
                    plot_path = base_path
                else:
                    plot_path = base_path.with_name(f"{base_path.stem}_{method}{base_path.suffix}")
                fig.savefig(plot_path, dpi=200)
                plt.close(fig)
                print(f"Saved correlation heatmap ({method}) to '{plot_path}'.")
        except ImportError as exc:
            print(f"Skipping heatmap creation because matplotlib is unavailable: {exc}")


if __name__ == "__main__":
    main()
