import argparse
import re
from typing import List

from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import list_datasets


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_pre", type=str, required=True, help="Prefix of repos to merge, e.g. MisDrifter/filtered_dataset_deduplicated_armo_")
    parser.add_argument("--output_repo", type=str, required=True, help="Destination merged repo id, e.g. MisDrifter/filtered_dataset_deduplicated_armo")
    return parser.parse_args()


def extract_numeric_suffix(repo_id: str) -> int:
    # Expect ids like <prefix><number>, return number for sorting; non-matching -> -1
    m = re.search(r"(\d+)$", repo_id)
    return int(m.group(1)) if m else -1


def find_repos(prefix: str) -> List[str]:
    # List all datasets under the user/org, filter by prefix
    results = list_datasets(search=prefix)
    repo_ids = [r.id for r in results if r.id.startswith(prefix)]
    repo_ids.sort(key=extract_numeric_suffix)
    return repo_ids


def main():
    args = parse_arguments()
    repos = find_repos(args.repo_pre)
    if not repos:
        raise ValueError(f"No datasets found with prefix: {args.repo_pre}")

    print(f"Found {len(repos)} repos to merge:")
    for r in repos:
        print(f" - {r}")

    merged = None
    for idx, repo_id in enumerate(repos):
        print(f"Loading {repo_id}...")
        ds = load_dataset(repo_id, split='train')
        if merged is None:
            merged = ds
        else:
            merged = concatenate_datasets([merged, ds])

    print(f"Merged rows: {len(merged)}. Pushing to {args.output_repo}...")
    merged.push_to_hub(args.output_repo)
    print("Done.")


if __name__ == "__main__":
    main()


