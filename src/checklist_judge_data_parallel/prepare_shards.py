import os
import math
import argparse
from typing import Optional

from datasets import load_dataset, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load HF dataset and save local shards of 1k prompts each.")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/Qwen2.5_3B_generation",
                        help="HF dataset repo or path, e.g., 'zjhhhh/Qwen2.5_3B_generation'")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Number of prompts per shard")
    parser.add_argument("--out_dir", type=str, default="./local_shards",
                        help="Directory to write shard folders (HF save_to_disk)")
    parser.add_argument("--max_shards", type=Optional[int], default=None,
                        help="Optional cap on number of shards to generate")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.input_repo, split=args.split)
    total = len(ds)
    if total == 0:
        raise RuntimeError("Loaded dataset is empty")

    n_shards = math.ceil(total / args.shard_size)
    if args.max_shards is not None:
        n_shards = min(n_shards, int(args.max_shards))

    print(f"Loaded {args.input_repo}:{args.split} with {total} rows")
    print(f"Writing {n_shards} shards to {args.out_dir} with shard_size={args.shard_size}")

    for idx in range(n_shards):
        start = idx * args.shard_size
        end = min(total, (idx + 1) * args.shard_size)
        if start >= end:
            break

        shard = ds.select(range(start, end))
        shard_path = os.path.join(args.out_dir, f"shard_{idx:05d}")
        # Save as HF Arrow dataset directory for reliable re-load with load_from_disk
        shard.save_to_disk(shard_path)
        print(f"Saved shard {idx} [{start}:{end}) -> {shard_path} ({len(shard)} rows)")

    # Optionally also write a simple manifest
    manifest = {
        "input_repo": args.input_repo,
        "split": args.split,
        "total_rows": total,
        "shard_size": args.shard_size,
        "num_shards": n_shards,
        "out_dir": os.path.abspath(args.out_dir),
    }
    try:
        import json
        with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write manifest: {e}")


if __name__ == "__main__":
    main()
