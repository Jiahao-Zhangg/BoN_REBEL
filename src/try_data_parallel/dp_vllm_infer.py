#!/usr/bin/env python3
"""
Pure Data-Parallel vLLM offline inference (single Python file)
- One worker PROCESS per GPU (TP=1, no cross-GPU comms)
- Per-GPU log files: outs/gpu{ID}.log
- Progress bars disabled for clean multi-proc logging
- Robust on RTX 4090: enforce_eager=True, VLLM_USE_CUDA_GRAPHS=0
- Precise timing: total wall time, init time, pure inference time
- Metrics per GPU (JSON) + final summary
- Chunked host batching (set --chunk-size 0 for one-shot per shard)

Example (2 GPUs):
  python dp_vllm_infer.py \
    --model /data/models/Qwen3-14B-Instruct \
    --shard-dir shards \
    --out-dir outs \
    --gpus 1 2 \
    --dtype float16 \
    --max-model-len 8192 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-new-tokens 256 \
    --chunk-size 512 \
    --max-num-batched-tokens 16384
"""

import os
import sys
import json
import glob
import time
import argparse
from multiprocessing import Process, set_start_method
from typing import List, Dict, Any


# ---------------------- helpers ----------------------
def list_shards(shard_dir: str) -> List[str]:
    """List shard files sorted by numeric suffix: shard_00000.jsonl ..."""
    paths = glob.glob(os.path.join(shard_dir, "shard_*.jsonl"))
    def shard_key(p):
        bn = os.path.basename(p)
        num = bn.replace("shard_", "").replace(".jsonl", "")
        try:
            return int(num)
        except ValueError:
            return 1 << 30
    return sorted(paths, key=shard_key)


def split_by_gpu(shards: List[str], gpu_indices: List[int]) -> List[List[str]]:
    """Evenly assign shards to GPUs by modulo of their sorted index."""
    assignments = [[] for _ in gpu_indices]
    for pos, path in enumerate(shards):
        assignments[pos % len(gpu_indices)].append(path)
    return assignments


# ---------------------- worker ----------------------
def run_worker(
    gpu_id: int,
    assigned_shards: List[str],
    model_path: str,
    out_dir: str,
    dtype: str,
    max_model_len: int,
    gpu_mem_util: float,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop: List[str],
    chunk_size: int,
    max_num_batched_tokens: int = None,  # optional vLLM tuning
):
    """
    Worker process bound to one GPU:
    - Redirect stdout/stderr to outs/gpu{ID}.log
    - Isolate CUDA device & disable progress bars
    - Import vLLM inside the worker
    - Load model (TP=1, eager) and process assigned shards
    - Record per-shard & total timings; write metrics JSON
    """
    # ----- per-GPU log file -----
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"gpu{gpu_id}.log")
    log_f = open(log_path, "w", buffering=1, encoding="utf-8")  # line-buffered
    sys.stdout = log_f
    sys.stderr = log_f

    # ----- isolate GPU & runtime env -----
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_NO_PROGRESS_BAR"] = "1"    # force disable progress bars
    # RTX 4090 stability: disable CUDA graphs unless you confirm it's stable
    os.environ.setdefault("VLLM_USE_CUDA_GRAPHS", "0")

    # ----- import vLLM inside worker AFTER env is set -----
    from vllm import LLM, SamplingParams

    # ----- model init with timing -----
    t_init0 = time.time()
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=1,       # pure DP (no tensor-parallel comms)
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        enforce_eager=True,           # robust on 4090; flip to False after you verify stability
    )
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

    print(f"[GPU{gpu_id}] Initializing LLM with args: {llm_kwargs}", flush=True)
    llm = LLM(**llm_kwargs)
    t_init1 = time.time()
    init_time = t_init1 - t_init0
    print(f"[GPU{gpu_id}] Model init took {init_time:.2f} s", flush=True)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop if stop else None,
    )

    # metrics container
    metrics: Dict[str, Any] = {
        "gpu_id": gpu_id,
        "init_time_sec": round(init_time, 3),
        "shards": [],                 # list of {name, num_items, wall_time_sec, gen_time_sec}
        "total_gen_time_sec": 0.0,
        "worker_total_time_sec": 0.0, # init + sum(shard wall times)
        "log_file": os.path.basename(log_path),
    }

    # ----- per-shard processing -----
    worker_wall_start = time.time()
    total_shard_wall = 0.0
    total_gen_time = 0.0

    for shard_path in assigned_shards:
        shard_name = os.path.basename(shard_path)
        out_path = os.path.join(out_dir, f"out_{shard_name}")
        logp = f"[GPU{gpu_id}] {shard_name}"

        print(f"{logp} -> loading shard", flush=True)
        prompts, ids = [], []
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompts.append(rec["prompt"])
                ids.append(rec.get("id"))

        total = len(prompts)
        print(f"{logp} -> {total} prompts", flush=True)

        shard_wall_start = time.time()
        shard_gen_time = 0.0

        with open(out_path, "w", encoding="utf-8") as fout:
            if chunk_size and chunk_size > 0:
                # ---- chunked host batching ----
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    batch_prompts = prompts[start:end]
                    batch_ids = ids[start:end]

                    print(f"{logp} -> starting chunk [{start}:{end}]", flush=True)
                    t_g0 = time.time()
                    outputs = llm.generate(batch_prompts, sampling_params)
                    t_g1 = time.time()
                    shard_gen_time += (t_g1 - t_g0)
                    print(f"{logp} -> finished  chunk [{start}:{end}] (gen {t_g1 - t_g0:.2f}s)", flush=True)

                    for rid, out in zip(batch_ids, outputs):
                        result = {"id": rid, "prompt": out.prompt, "output": out.outputs[0].text}
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

                    print(f"{logp} -> processed {end}/{total}", flush=True)
            else:
                # ---- one-shot per shard ----
                print(f"{logp} -> starting ONE-SHOT for all {total} prompts", flush=True)
                t_g0 = time.time()
                outputs = llm.generate(prompts, sampling_params)
                t_g1 = time.time()
                shard_gen_time += (t_g1 - t_g0)
                print(f"{logp} -> finished  ONE-SHOT (gen {t_g1 - t_g0:.2f}s)", flush=True)

                for rid, out in zip(ids, outputs):
                    result = {"id": rid, "prompt": out.prompt, "output": out.outputs[0].text}
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        shard_wall = time.time() - shard_wall_start
        total_shard_wall += shard_wall
        total_gen_time += shard_gen_time

        print(f"{logp} -> DONE {total} items | shard wall={shard_wall:.2f}s | shard gen={shard_gen_time:.2f}s | out={out_path}", flush=True)

        metrics["shards"].append({
            "name": shard_name,
            "num_items": total,
            "wall_time_sec": round(shard_wall, 3),
            "gen_time_sec": round(shard_gen_time, 3),
        })

    worker_wall = (time.time() - worker_wall_start) + init_time
    metrics["total_gen_time_sec"] = round(total_gen_time, 3)
    metrics["worker_total_time_sec"] = round(worker_wall, 3)

    metrics_path = os.path.join(out_dir, f"metrics_gpu{gpu_id}.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)
    print(f"[GPU{gpu_id}] Metrics -> {metrics_path}", flush=True)

    # Flush & close log
    log_f.flush()
    log_f.close()


# ---------------------- merge & summary ----------------------
def merge_outputs(out_dir: str, merged_name: str = "all_outputs.jsonl"):
    """Concatenate all out_shard_*.jsonl into a single file."""
    out_paths = sorted(glob.glob(os.path.join(out_dir, "out_shard_*.jsonl")))
    merged_path = os.path.join(out_dir, merged_name)
    with open(merged_path, "w", encoding="utf-8") as fout:
        for p in out_paths:
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    print(f"[MERGE] -> {merged_path}")


def summarize_metrics(out_dir: str, t_main_start: float, t_main_end: float):
    """Read per-GPU metrics and print an overall summary."""
    metric_files = sorted(glob.glob(os.path.join(out_dir, "metrics_gpu*.json")))
    if not metric_files:
        print("[SUMMARY] No metrics files found.", file=sys.stderr)
        return

    all_metrics = []
    for mf in metric_files:
        with open(mf, "r", encoding="utf-8") as f:
            all_metrics.append(json.load(f))

    main_wall = t_main_end - t_main_start
    max_init = max(m["init_time_sec"] for m in all_metrics)
    max_worker_total = max(m["worker_total_time_sec"] for m in all_metrics)
    max_pure_infer = max(m["total_gen_time_sec"] for m in all_metrics)

    print("\n===== SUMMARY =====")
    print(f"Main wall time (overall):         {main_wall:.2f} s")
    print(f"Max model init time (per GPU):    {max_init:.2f} s")
    print(f"Critical path (init+infer, GPU):  {max_worker_total:.2f} s")
    print(f"Critical path (infer only, GPU):  {max_pure_infer:.2f} s")
    print(f"Per-GPU logs: {', '.join(sorted(os.path.basename(m['log_file']) for m in all_metrics))}")
    print("===================\n")


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Pure DP vLLM offline inference (single Python program) with per-GPU logs.")
    # IO
    ap.add_argument("--model", type=str, required=True, help="Local model path or HF repo name")
    ap.add_argument("--shard-dir", type=str, required=True, help="Directory containing shard_*.jsonl")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory to write outputs + metrics + logs")
    # GPUs
    ap.add_argument("--gpus", type=int, nargs="+", required=True, help="GPU IDs, e.g., --gpus 0 1")
    # Engine / model
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bf16"], help="Model dtype")
    ap.add_argument("--max-model-len", type=int, default=8192, help="Max (prompt + generated) tokens")
    ap.add_argument("--gpu-mem-util", type=float, default=0.92, help="GPU memory utilization target")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="Optional vLLM engine cap on total batched tokens (e.g., 32768)")
    # Sampling
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per request")
    ap.add_argument("--stop", type=str, nargs="*", default=None, help="Optional stop tokens, space-separated")
    # Host chunking
    ap.add_argument("--chunk-size", type=int, default=512, help="Prompts per llm.generate() call (0 = one-shot)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Discover and assign shards
    shards = list_shards(args.shard_dir)
    if not shards:
        print(f"[ERROR] No shards found in: {args.shard_dir}", file=sys.stderr)
        sys.exit(1)

    gpu_indices = args.gpus
    assignments = split_by_gpu(shards, gpu_indices)

    print(f"Found {len(shards)} shard(s). Using GPUs: {gpu_indices}")
    for gpu_id, shard_list in zip(gpu_indices, assignments):
        print(f"  GPU {gpu_id} -> {len(shard_list)} shard(s)")
    print(f"Per-GPU logs will be written to: {args.out_dir}/gpu{{ID}}.log")

    # Launch one worker process per GPU
    procs: List[Process] = []
    t_main_start = time.time()
    for gpu_id, shard_list in zip(gpu_indices, assignments):
        p = Process(
            target=run_worker,
            args=(
                gpu_id,
                shard_list,
                args.model,
                args.out_dir,
                args.dtype,
                args.max_model_len,
                args.gpu_mem_util,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
                args.stop,
                args.chunk_size,
                args.max_num_batched_tokens,
            ),
        )
        p.start()
        procs.append(p)

    # Wait for all workers and check exit codes
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"[ERROR] A worker exited with code {p.exitcode}", file=sys.stderr)
            sys.exit(p.exitcode)

    # Merge outputs after all workers finish
    merge_outputs(args.out_dir, merged_name="all_outputs.jsonl")
    t_main_end = time.time()

    # Print final timing summary
    summarize_metrics(args.out_dir, t_main_start, t_main_end)


if __name__ == "__main__":
    # Force 'spawn' BEFORE anything that might touch CUDA/vLLM
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
