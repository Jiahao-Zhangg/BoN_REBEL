#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_repo_prefix", type=str, required=True, help="Prefix used in step 2; repos are {prefix}_{model_name}")
    parser.add_argument("--check_points", type=str, nargs='+', required=True, help="The same checkpoints evaluated in step 2")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--score_json_path", type=str, default=None)
    return parser.parse_args()


def sanitize_model_name(model_id: str) -> str:
    return model_id.strip().replace("/", "__").replace(" ", "_")


def _collect_normalized_scores(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_collect_normalized_scores(item))
        return out
    try:
        return [float(value) / 4]
    except Exception:
        return []


def score_dataset(ds, beta: float) -> float:
    import math

    base_cols = sorted([c for c in ds.column_names if c.startswith("base_response_")], key=lambda x: int(x.split("_")[-1]))
    model_cols = sorted([c for c in ds.column_names if c.startswith("model_response_")], key=lambda x: int(x.split("_")[-1]))
    n_base = len(base_cols)
    n_model = len(model_cols)

    grouped = {}
    for row in ds:
        p = row["prompt"]
        grouped.setdefault(p, []).append(row)

    prompt_vals = []
    for p, rows in grouped.items():
        check_objectives = []
        for row in rows:
            base_exp_terms = []
            for b in range(n_base):
                vals = []
                for a in range(n_model):
                    mean_key = f"judge_{a}_{b}_mean"
                    maj_key = f"judge_{a}_{b}_majority"
                    values = _collect_normalized_scores(row.get(mean_key, None))
                    if not values:
                        values = _collect_normalized_scores(row.get(maj_key, None))
                    if values:
                        vals.extend(values)
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
        return float('nan')

    return sum(prompt_vals) / len(prompt_vals)


def _split_requirements_to_checks_for_prompt(requirements_str: str) -> List[str]:
    if not requirements_str:
        return [""]
    expanded = []
    counter = 1
    remaining = requirements_str
    while len(remaining) > 0:
        assert remaining.startswith(f"{counter})"), f"Malformed requirements at counter {counter}: {remaining[:80]}"
        needle = f"/100)\n{counter+1})"
        pos = remaining.find(needle)
        if pos > 0:
            curr = remaining[len(f"{counter})"):pos + len("/100)\n")]
        else:
            curr = remaining[len(f"{counter})"):]
        expanded.append(curr.strip())
        remaining = remaining[len(curr) + len(f"{counter})"):]
        counter += 1
    return expanded


def score_merged_dataset(ds, beta: float) -> float:
    import math

    base_cols = sorted([c for c in ds.column_names if c.startswith("base_response_")], key=lambda x: int(x.split("_")[-1]))
    model_cols = sorted([c for c in ds.column_names if c.startswith("model_response_")], key=lambda x: int(x.split("_")[-1]))
    n_base = len(base_cols)
    n_model = len(model_cols)

    prompt_vals: List[float] = []
    for row in ds:
        prompt = row["prompt"]
        requirements_str = row.get("requirements", "")
        checks = _split_requirements_to_checks_for_prompt(requirements_str)

        check_objectives: List[float] = []
        for check_idx in range(len(checks)):
            base_exp_terms = []
            for b in range(n_base):
                vals = []
                for a in range(n_model):
                    mean_key = f"judge_{a}_{b}_mean"
                    maj_key = f"judge_{a}_{b}_majority"
                    v = row.get(mean_key, None)
                    if v is None:
                        v = row.get(maj_key, None)
                    # v can be list (merged) or scalar
                    if isinstance(v, (list, tuple)):
                        if check_idx < len(v) and v[check_idx] is not None:
                            vals.extend(_collect_normalized_scores(v[check_idx]))
                    else:
                        # Fallback if merged missing; treat as scalar per-prompt
                        vals.extend(_collect_normalized_scores(v))
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
        return float('nan')

    return sum(prompt_vals) / len(prompt_vals)


def main():
    args = parse_arguments()
    from datasets import load_dataset

    scores = []
    for model_id in args.check_points:
        model_name = sanitize_model_name(model_id.split(":")[0])
        repo_id = f"{args.output_repo_prefix}_{model_name}"
        try:
            ds = load_dataset(repo_id, split='test')
        except Exception:
            ds = load_dataset(repo_id, split='train')
        # Detect merged (list) judge columns and use merged-aware scoring to match v2
        sample_key = next((c for c in ds.column_names if c.startswith("judge_") and (c.endswith("_mean") or c.endswith("_majority"))), None)
        use_merged = False
        if sample_key is not None and len(ds) > 0:
            v0 = ds[0].get(sample_key, None)
            use_merged = isinstance(v0, (list, tuple))
        if use_merged:
            score = score_merged_dataset(ds, args.beta)
        else:
            score = score_dataset(ds, args.beta)
        print(f"Score for {model_name}: {score}")
        scores.append(score)

    summary_path = Path(args.score_json_path) if args.score_json_path else Path(f"{sanitize_model_name(args.output_repo_prefix)}_scores.json")
    with open(summary_path, "w") as f:
        json.dump({"checkpoints": args.check_points, "scores": scores}, f, indent=2)
    print(f"Saved score summary to {summary_path}")


if __name__ == "__main__":
    main()


