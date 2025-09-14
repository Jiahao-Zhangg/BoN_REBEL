# hf_to_jsonl_shards.py
import argparse, os, json
from datasets import load_dataset

def build_prompt(example, prompt_field, template=None):
    text = example[prompt_field]
    if template:
        return template.replace("{text}", str(text))
    return str(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                    help="HF dataset name or local path, e.g. 'tatsu-lab/alpaca'")
    ap.add_argument("--split", type=str, default="train",
                    help="Dataset split to use, e.g. 'train', 'test'")
    ap.add_argument("--prompt-field", type=str, required=True,
                    help="Column name to use as prompt")
    ap.add_argument("--template", type=str, default=None,
                    help="Optional template, must contain {text} placeholder")
    ap.add_argument("--num-shards", type=int, required=True,
                    help="Number of shard files to create")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Output directory for shard files")
    ap.add_argument("--streaming", action="store_true",
                    help="Enable streaming mode (avoid loading entire dataset)")
    ap.add_argument("--end-idx", type=int, default=None,
                    help="Only take the first N rows from the dataset")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    writers = []
    for i in range(args.num_shards):
        writers.append(open(os.path.join(args.outdir, f"shard_{i:05d}.jsonl"), "w", encoding="utf-8"))

    for idx, ex in enumerate(ds):
        # Stop if we've reached end-idx
        if args.end_idx is not None and idx >= args.end_idx:
            break

        prompt = build_prompt(ex, args.prompt_field, args.template)
        rec = {"id": ex.get("id", idx), "prompt": prompt}
        shard_id = idx % args.num_shards
        writers[shard_id].write(json.dumps(rec, ensure_ascii=False) + "\n")

    for w in writers:
        w.close()

if __name__ == "__main__":
    main()
