import argparse
from typing import List, Tuple

from datasets import load_dataset, concatenate_datasets, Dataset, DownloadConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge HF datasets from '{shard_ids}'-templated repos and push to output."
    )
    parser.add_argument("--input_repo", type=str, required=True,help="HF repo template with {shard_ids}, e.g. 'MisDrifter/{shard_ids}'")
    parser.add_argument("--n_shard",type=int, required=True, help="Number of shards (expand 0..n_shard-1)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--output_repo", type=str, required=True, help="Target HF dataset repo to push merged dataset")
    parser.add_argument("--private", action="store_true", help="Push as private")
    return parser.parse_args()



def expand_repos(template: str, n_shard: int) -> List[str]:
    if "{shard_ids}" not in template:
        raise ValueError("--input_repo must contain '{shard_ids}' placeholder")
    repos: List[str] = []
    for shard_idx in range(n_shard):
        repo_id = template.format(shard_ids=shard_idx)
        repos.append(repo_id)
    return repos


def load_all_shards(repo_ids: List[str], split: str) -> Tuple[List[Dataset], List[str], List[str]]:
    datasets: List[Dataset] = []
    used_repo_ids: List[str] = []
    skipped_repo_ids: List[str] = []
    for repo_id in repo_ids:
        try:
            ds = load_dataset(
                repo_id,
                split=split,
                download_config=DownloadConfig(force_download=True),
            )
            datasets.append(ds)
            used_repo_ids.append(repo_id)
            print(f"Loaded {repo_id}:{split} with {len(ds)} rows")
        except Exception as e:
            skipped_repo_ids.append(repo_id)
            print(f"Skipping {repo_id}:{split} due to error: {e}")
    return datasets, used_repo_ids, skipped_repo_ids


def merge_and_push(datasets_list: List[Dataset], output_repo: str, private: bool) -> None:
    if not datasets_list:
        raise RuntimeError("No datasets loaded; nothing to merge")
    if len(datasets_list) == 1:
        merged = datasets_list[0]
    else:
        merged = concatenate_datasets(datasets_list)
    print(f"Merged total rows: {len(merged)} from {len(datasets_list)} shards")

    # Push to Hugging Face Hub
    # If the dataset doesn't exist, it will be created on push
    # Requires HUGGINGFACE_HUB_TOKEN to be set for auth
    merged.push_to_hub(output_repo, private=private)
    print(f"Pushed merged dataset to {output_repo}")


def main():
    args = parse_args()
    repo_ids = expand_repos(args.input_repo, args.n_shard)
    shards, used_repo_ids, skipped_repo_ids = load_all_shards(repo_ids, args.split)
    if not shards:
        raise RuntimeError("No datasets loaded after skipping missing shards; aborting merge")
    merge_and_push(shards, args.output_repo, args.private)
    print("=== Merge Summary ===")
    print(f"Loaded from {len(used_repo_ids)} shard repos:")
    for rid in used_repo_ids:
        print(f" - {rid}")
    if skipped_repo_ids:
        print(f"Skipped {len(skipped_repo_ids)} shard repos:")
        for rid in skipped_repo_ids:
            print(f" - {rid}")


if __name__ == "__main__":
    main()