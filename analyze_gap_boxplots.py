#!/usr/bin/env python3
"""Generate boxplots and correlations of reward gaps for selected Hugging Face datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    label: str
    chosen_key: str
    reject_key: str
    splits: Tuple[str, ...] = ("train", "test")


@dataclass
class DatasetResult:
    label: str
    prompts: List[str]
    gaps: List[float]


DATASETS: List[DatasetConfig] = [
    DatasetConfig(
        name="MisDrifter/filtered_wholedataset_armo_tokenized",
        label="reward",
        chosen_key="chosen_reward",
        reject_key="reject_reward",
    ),
    DatasetConfig(
        name="zjhhhh/whole_sw_maxlen_8192_rescale_mean_maxlenp_1024_beta_10.0_fixed_tokenized",
        label="beta_10",
        chosen_key="g_chosen",
        reject_key="g_reject",
    ),
    DatasetConfig(
        name="zjhhhh/whole_sw_maxlen_8192_rescale_mean_maxlenp_1024_beta_1.0_fixed_tokenized",
        label="beta_1",
        chosen_key="g_chosen",
        reject_key="g_reject",
    ),
    DatasetConfig(
        name="zjhhhh/whole_sw_maxlen_8192_rescale_mean_maxlenp_1024_beta_0.1_fixed_tokenized",
        label="beta_0.1",
        chosen_key="g_chosen",
        reject_key="g_reject",
    ),
    DatasetConfig(
        name="zjhhhh/whole_sw_maxlen_8192_rescale_mean_maxlenp_1024_beta_0.01_fixed_tokenized",
        label="beta_0.01",
        chosen_key="g_chosen",
        reject_key="g_reject",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reward_gap_boxplot.png"),
        help="Path to save the generated boxplot (default: reward_gap_boxplot.png)",
    )
    parser.add_argument(
        "--correlation-output",
        type=Path,
        default=Path("reward_gap_correlation_heatmap.png"),
        help="Path to save the correlation heatmap (default: reward_gap_correlation_heatmap.png)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of samples to analyze per split. Use 0 for all samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for potential future sampling logic (unused).",
    )
    return parser.parse_args()


def sample_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples and len(dataset) > max_samples:
        return dataset.select(range(max_samples))
    return dataset


def compute_prompt_gaps(dataset: Dataset, config: DatasetConfig) -> Tuple[List[str], List[float]]:
    prompts: List[str] = []
    gaps: List[float] = []
    missing = 0

    for record in dataset:
        try:
            prompt = record["prompt"]
            chosen = record[config.chosen_key]
            reject = record[config.reject_key]
        except KeyError:
            missing += 1
            continue

        if prompt is None or chosen is None or reject is None:
            missing += 1
            continue

        prompts.append(prompt)
        gaps.append(float(chosen) - float(reject))

    if missing:
        print(
            f"[warning] {config.name} ({config.label}): skipped {missing} rows without required fields"
        )

    return prompts, gaps


def load_and_prepare(config: DatasetConfig, max_samples: int) -> DatasetResult:
    combined_prompts: List[str] = []
    combined_gaps: List[float] = []

    for split in config.splits:
        print(f"Loading {config.name} ({config.label}) split={split}")
        try:
            dataset = load_dataset(config.name, split=split)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"[warning] {config.name} ({config.label}): unable to load split '{split}': {exc}")
            continue

        dataset = sample_dataset(dataset, max_samples=max_samples)
        print(f"  -> using {len(dataset)} samples")

        prompts, gaps = compute_prompt_gaps(dataset, config)
        combined_prompts.extend(prompts)
        combined_gaps.extend(gaps)

        dataset.cleanup_cache_files()

    return DatasetResult(label=config.label, prompts=combined_prompts, gaps=combined_gaps)


def create_boxplot(results: Iterable[DatasetResult], output_path: Path) -> None:
    gaps = [result.gaps for result in results]
    labels = [result.label for result in results]

    plt.figure(figsize=(10, 6))
    plt.boxplot(gaps, labels=labels, showfliers=False)
    plt.ylabel("Gap")
    plt.title("Chosen vs Reject Gap Distribution")
    plt.grid(axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved boxplot to {output_path}")


def build_prompt_gap_maps(results: Iterable[DatasetResult]) -> Dict[str, Dict[str, float]]:
    prompt_maps: Dict[str, Dict[str, float]] = {}
    for result in results:
        prompt_maps[result.label] = {prompt: gap for prompt, gap in zip(result.prompts, result.gaps)}
    return prompt_maps


def compute_correlation_matrix(results: List[DatasetResult]) -> np.ndarray:
    n = len(results)
    corr_matrix = np.full((n, n), np.nan)

    prompt_maps = build_prompt_gap_maps(results)

    for i, res_i in enumerate(results):
        map_i = prompt_maps[res_i.label]
        corr_matrix[i, i] = 1.0
        keys_i = set(map_i.keys())

        for j in range(i + 1, n):
            res_j = results[j]
            map_j = prompt_maps[res_j.label]
            common_prompts = keys_i.intersection(map_j.keys())

            if len(common_prompts) < 2:
                print(
                    f"[warning] Not enough overlapping prompts between {res_i.label} and {res_j.label}"
                )
                continue

            ordered_prompts = sorted(common_prompts)
            values_i = np.array([map_i[prompt] for prompt in ordered_prompts], dtype=float)
            values_j = np.array([map_j[prompt] for prompt in ordered_prompts], dtype=float)

            corr = np.corrcoef(values_i, values_j)[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            print(
                f"Correlation {res_i.label} vs {res_j.label}: {corr:.4f} using {len(common_prompts)} prompts"
            )

    return corr_matrix


def create_correlation_heatmap(results: List[DatasetResult], output_path: Path) -> None:
    if len(results) < 2:
        print("[warning] Not enough datasets to compute correlations")
        return

    corr_matrix = compute_correlation_matrix(results)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson correlation")
    labels = [result.label for result in results]
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Reward Gap Correlation (Aligned by Prompt)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr_matrix[i, j]
            if np.isnan(value):
                display = "NA"
            else:
                display = f"{value:.2f}"
            plt.text(j, i, display, ha="center", va="center", color="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved correlation heatmap to {output_path}")


def main() -> None:
    args = parse_args()
    results: List[DatasetResult] = []

    for config in DATASETS:
        result = load_and_prepare(config, max_samples=args.max_samples)
        if not result.gaps:
            print(f"[warning] {config.name} ({config.label}): no valid samples found")
            continue
        results.append(result)

    if not results:
        raise RuntimeError("No data available to analyze. Check dataset keys and availability.")

    create_boxplot(results, args.output)
    create_correlation_heatmap(results, args.correlation_output)


if __name__ == "__main__":
    main()
