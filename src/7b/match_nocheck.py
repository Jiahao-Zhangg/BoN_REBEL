import argparse
from datasets import load_dataset, DatasetDict, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match nocheck dataset (scalar scores) to preprocessed dataset by prompt, per split, and push.")
    parser.add_argument("--preprocessed_repo", type=str, required=True,
                        help="Repo of the preprocessed dataset with train/test splits")
    parser.add_argument("--nocheck_repo", type=str, required=True,
                        help="Repo that contains scalar nocheck scores (assumed in train split)")
    parser.add_argument("--output_repo", type=str, required=True,
                        help="Target repo to push merged dataset with nocheck scores attached")
    return parser.parse_args()


def build_index_by_prompt(ds):
    index = {}
    for row in ds:
        p = row.get("prompt")
        if p is not None and p not in index:
            index[p] = row
    return index


def merge_split(pre_ds, nocheck_ds):
    """
    Build split starting from nocheck rows, restricted to prompts present in pre_ds.
    Upload nocheck-based rows and attach essential preprocessed fields (e.g., qwen_prompt tokens).
    """
    # Build preprocessed prompt -> row index
    pre_index = build_index_by_prompt(pre_ds)

    nocheck_cols = list(nocheck_ds.column_names)
    merged_rows = {name: [] for name in nocheck_cols}
    extra_cols = {"qwen_prompt": [], "qwen_prompt_tokens": []}
    added_any = {"qwen_prompt": False, "qwen_prompt_tokens": False}

    for row in nocheck_ds:
        prompt = row.get("prompt")
        if prompt is None:
            continue
        pre_row = pre_index.get(prompt)
        if pre_row is None:
            continue

        # Keep nocheck row as primary
        for name in nocheck_cols:
            merged_rows[name].append(row[name])

        # Attach preprocessed-only convenience fields if present
        for k in ("qwen_prompt", "qwen_prompt_tokens"):
            if k in pre_row:
                extra_cols[k].append(pre_row[k])
                added_any[k] = True

    merged = Dataset.from_dict(merged_rows)
    for k, values in extra_cols.items():
        if added_any[k] and len(values) == len(merged):
            merged = merged.add_column(k, values)
    return merged


def main():
    args = parse_args()

    pre_ds = load_dataset(args.preprocessed_repo)
    if 'train' not in pre_ds or 'test' not in pre_ds:
        raise ValueError("Preprocessed repo must have train and test splits")

    nocheck_train = load_dataset(args.nocheck_repo, split='train')

    merged_train = merge_split(pre_ds['train'], nocheck_train)
    merged_test = merge_split(pre_ds['test'], nocheck_train)

    out = DatasetDict({
        'train': merged_train,
        'test': merged_test,
    })
    out.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()


