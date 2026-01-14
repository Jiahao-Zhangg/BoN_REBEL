import os
import json
import time
import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


# ------------------ Structured outputs for guided decoding ------------------
class Preference5ScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=-1, le=4)


PREFERENCE_5SCORE_GUIDED_DECODING = GuidedDecodingParams(
    json=Preference5ScoreOutput.model_json_schema()
)


# ------------------ Utilities ------------------
def set_seed(seed=5775709):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_tag(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def get_message(instruction: str):
    return [{"role": "user", "content": instruction}]


def extract_verdict(response_text: str):
    try:
        parsed = json.loads(response_text)
    except Exception:
        return None
    return parsed.get("verdict", None)


def is_valid_5score(response: str) -> bool:
    try:
        score = int(response)
        return -1 <= score <= 4
    except Exception:
        return False


def reverse_score(score: int) -> int:
    if score == -1:
        return -1
    return 4 - int(score)


def get_numeric_mode(values: List[int], score_range: Optional[Tuple[int, int]] = None):
    if not values:
        return None
    if score_range is not None:
        min_s, max_s = score_range
        values = [int(v) for v in values if min_s <= int(v) <= max_s]
    else:
        values = [int(v) for v in values]
    if not values:
        return None
    counts: Dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    max_c = max(counts.values())
    modes = [k for k, c in counts.items() if c == max_c]
    if len(modes) == 1:
        return modes[0]
    return float(sum(modes) / len(modes))


def parse_models_arg(models: str, models_file: Optional[str]) -> List[str]:
    out: List[str] = []
    if models:
        out.extend([m.strip() for m in models.split(",") if m.strip()])
    if models_file:
        with open(models_file, "r") as f:
            for line in f:
                val = line.strip()
                if val:
                    out.append(val)
    if not out:
        raise ValueError("No models provided. Use --models or --models_file.")
    seen = set()
    ordered: List[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def parse_aliases(models: List[str], aliases: str) -> List[str]:
    if not aliases:
        return [safe_tag(m) for m in models]
    values = [a.strip() for a in aliases.split(",") if a.strip()]
    if len(values) != len(models):
        raise ValueError(f"--aliases length {len(values)} must match --models length {len(models)}")
    return values


def load_input_dataset(input_repo: str, split: str) -> Dataset:
    if os.path.isdir(input_repo):
        ds = load_from_disk(input_repo)
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            raise ValueError(f"Split '{split}' not found in local dataset: {list(ds.keys())}")
        return ds
    try:
        return load_dataset(input_repo, split=split)
    except Exception:
        ds_dict = load_dataset(input_repo)
        if split in ds_dict:
            return ds_dict[split]
        raise


def load_existing_responses(path: str, n_rows: int) -> List[Optional[Tuple[str, str]]]:
    responses: List[Optional[Tuple[str, str]]] = [None] * n_rows
    if not os.path.exists(path):
        return responses
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            idx = row.get("idx")
            if idx is None or not isinstance(idx, int):
                continue
            if idx < 0 or idx >= n_rows:
                continue
            r1 = row.get("response_1")
            r2 = row.get("response_2")
            if r1 is None or r2 is None:
                continue
            responses[idx] = (r1, r2)
    return responses


def resolve_scores_repo(template: str, alias_a: str, alias_b: str) -> str:
    tag_a = safe_tag(alias_a)
    tag_b = safe_tag(alias_b)
    if "{model_a" in template or "{model_b" in template:
        return template.format(model_a=tag_a, model_b=tag_b)
    return f"{template}_{tag_a}_{tag_b}"


def apply_postfix(repo_id: str, postfix: str) -> str:
    if not postfix:
        return repo_id
    tag = safe_tag(postfix).lstrip("_")
    if not tag:
        return repo_id
    return f"{repo_id}_{tag}"


def load_pair_checkpoints(ckpt_root: str, n_rows: int, pair_results: Dict[str, List]):
    if not os.path.isdir(ckpt_root):
        return
    loaded = 0
    for name in os.listdir(ckpt_root):
        if not name.endswith(".json"):
            continue
        path = os.path.join(ckpt_root, name)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        label = data.get("label")
        if not label:
            continue
        if data.get("n_rows") != n_rows:
            continue
        mean_vals = data.get("mean")
        maj_vals = data.get("majority")
        if mean_vals is None or maj_vals is None:
            continue
        pair_results[label + "_mean"] = mean_vals
        pair_results[label + "_majority"] = maj_vals
        loaded += 1
    if loaded:
        print(f"Resumed {loaded} score checkpoints from {ckpt_root}")


def save_pair_checkpoint(ckpt_root: str, label: str, n_rows: int, pair_results: Dict[str, List]):
    os.makedirs(ckpt_root, exist_ok=True)
    payload = {
        "label": label,
        "n_rows": n_rows,
        "mean": pair_results.get(label + "_mean", []),
        "majority": pair_results.get(label + "_majority", []),
    }
    tmp = os.path.join(ckpt_root, f"{label}.json.tmp")
    out = os.path.join(ckpt_root, f"{label}.json")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, out)


def run_pair_and_reduce(
    prompts: List[str],
    checks: List[str],
    resp_a_list: List[str],
    resp_b_list: List[str],
    label: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompt_template: str,
    sampling_params: SamplingParams,
    switch_position: bool,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    rendered = []
    for prompt, check, resp_a, resp_b in zip(prompts, checks, resp_a_list, resp_b_list):
        filled = prompt_template.format(
            prompt=prompt,
            response_a=resp_a,
            response_b=resp_b,
            check=check,
        )
        rendered.append(tokenizer.apply_chat_template(get_message(filled), tokenize=False, add_generation_prompt=True))

    responses = llm.generate(rendered, sampling_params)
    texts = [[o.text for o in r.outputs] for r in responses]

    verdict_lists: List[List[int]] = []
    for per_row in texts:
        vals = [extract_verdict(t) for t in per_row]
        vals = [v for v in vals if v is not None and is_valid_5score(v)]
        verdict_lists.append([int(v) for v in vals])

    orig_mean: List[Optional[float]] = []
    orig_majority: List[Optional[float]] = []
    for vals in verdict_lists:
        vals_no_missing = [v for v in vals if v != -1]
        if not vals_no_missing:
            orig_mean.append(None)
            orig_majority.append(None)
        else:
            orig_mean.append(float(np.mean(vals_no_missing)))
            orig_majority.append(get_numeric_mode(vals, (0, 4)))

    reduced_mean = orig_mean
    reduced_majority = orig_majority

    if switch_position:
        rendered_sw = []
        for prompt, check, resp_a, resp_b in zip(prompts, checks, resp_b_list, resp_a_list):
            filled = prompt_template.format(
                prompt=prompt,
                response_a=resp_a,
                response_b=resp_b,
                check=check,
            )
            rendered_sw.append(tokenizer.apply_chat_template(get_message(filled), tokenize=False, add_generation_prompt=True))

        responses_sw = llm.generate(rendered_sw, sampling_params)
        texts_sw = [[o.text for o in r.outputs] for r in responses_sw]

        sw_mean: List[Optional[float]] = []
        sw_majority: List[Optional[float]] = []
        for per_row in texts_sw:
            vals = [extract_verdict(t) for t in per_row]
            vals = [v for v in vals if v is not None and is_valid_5score(v)]
            sw_vals = [reverse_score(int(v)) for v in vals]
            sw_no_missing = [v for v in sw_vals if v != -1]
            if not sw_no_missing:
                sw_mean.append(None)
                sw_majority.append(None)
            else:
                sw_mean.append(float(np.mean(sw_no_missing)))
                sw_majority.append(get_numeric_mode(sw_vals, (0, 4)))

        new_mean: List[Optional[float]] = []
        new_majority: List[Optional[float]] = []
        for om, sm, oj, sj in zip(reduced_mean, sw_mean, reduced_majority, sw_majority):
            if om is None and sm is None:
                new_mean.append(None)
            elif om is None:
                new_mean.append(sm)
            elif sm is None:
                new_mean.append(om)
            else:
                new_mean.append(0.5 * (om + sm))

            if oj is None and sj is None:
                new_majority.append(None)
            elif oj is None:
                new_majority.append(sj)
            elif sj is None:
                new_majority.append(oj)
            else:
                new_majority.append(0.5 * (float(oj) + float(sj)))

        reduced_mean = new_mean
        reduced_majority = new_majority

    return reduced_mean, reduced_majority


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score pairwise preference_5score comparisons using pre-generated responses."
    )
    parser.add_argument("--input_repo", type=str, required=True, help="HF repo or local dataset path")
    parser.add_argument("--test_split", type=str, default="test", help="Split to load from input repo")
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index into the split")
    parser.add_argument("--num_prompts", type=int, default=-1,
                        help="Number of prompts to use from start_idx (<=0 means all remaining)")

    parser.add_argument("--models", type=str, default="", help="Comma-separated list of model ids")
    parser.add_argument("--models_file", type=str, default=None, help="Optional file with one model id per line")
    parser.add_argument("--aliases", type=str, default="",
                        help="Comma-separated list of aliases aligned with --models")

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--responses_dir", type=str, default="",
                        help="Directory containing per-model response JSONL files (defaults to output_dir/responses)")
    parser.add_argument("--scores_repo_template", type=str, default="zjhhhh/{model_a}_{model_b}")
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hf_postfix", type=str, default="",
                        help="Optional suffix to append to the hub repo id (e.g., shard id)")

    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--judge_world_size", type=int, default=2)
    parser.add_argument("--judge_max_tokens", type=int, default=256)
    parser.add_argument("--judge_temperature", type=float, default=0.6)
    parser.add_argument("--judge_top_p", type=float, default=0.95)
    parser.add_argument("--judge_top_k", type=int, default=20)
    parser.add_argument("--judge_n_samples", type=int, default=5)
    parser.add_argument("--judge_seed", type=int, default=42)
    parser.add_argument("--switch_position", dest="switch_position", action="store_true", default=False)
    parser.add_argument("--no_switch_position", dest="switch_position", action="store_false")
    return parser.parse_args()


def main():
    st = time.time()
    args = parse_args()

    models = parse_models_arg(args.models, args.models_file)
    aliases = parse_aliases(models, args.aliases)
    alias_by_model = {m: a for m, a in zip(models, aliases)}
    alias_tag_by_model = {m: safe_tag(a) for m, a in alias_by_model.items()}
    ds = load_input_dataset(args.input_repo, args.test_split)
    if args.start_idx < 0 or args.start_idx >= len(ds):
        raise ValueError(f"--start_idx {args.start_idx} is out of range for split size {len(ds)}")
    if args.num_prompts > 0:
        end_idx = min(args.start_idx + args.num_prompts, len(ds))
    else:
        end_idx = len(ds)
    ds = ds.select(range(args.start_idx, end_idx))
    if args.prompt_column not in ds.column_names:
        raise ValueError(
            f"Missing prompt column '{args.prompt_column}' in dataset. Columns: {ds.column_names}"
        )
    if "requirements" not in ds.column_names:
        raise ValueError("Input dataset must include a 'requirements' column for per-check scoring.")

    prompts = [row[args.prompt_column] for row in ds]
    n_rows = len(prompts)
    print(f"Loaded {n_rows} prompts from {args.input_repo} ({args.test_split})")

    expanded = []
    for orig_idx in range(n_rows):
        row = ds[orig_idx]
        req_str = row["requirements"]
        counter = 1
        chunks: List[str] = []
        while len(req_str) > 0:
            assert req_str.startswith(f"{counter})"), (
                f"Malformed requirements at row {orig_idx}: expected '{counter})' but got: {req_str[:20]}...")
            marker = f"/100)\n{counter+1})"
            pos = req_str.find(marker)
            if pos > 0:
                curr = req_str[len(f"{counter})"): pos + len("/100)\n")]
            else:
                curr = req_str[len(f"{counter})"):]
            chunks.append(curr)
            req_str = req_str[len(curr) + len(f"{counter})"):]
            counter += 1
        chunks = [c.strip() for c in chunks]
        for c in chunks:
            new_row = {k: row[k] for k in ds.column_names}
            new_row["check"] = c.split("(importance:")[0].strip()
            try:
                new_row["importance"] = int(c.split("(importance:")[1].split("/")[0].strip())
            except Exception:
                new_row["importance"] = None
            new_row["orig_index"] = orig_idx
            expanded.append(new_row)

    eds: Dataset = Dataset.from_list(expanded)
    n_expanded = len(eds)
    print(f"Expanded to {n_expanded} check-rows across {n_rows} prompts")

    responses_root = args.responses_dir or os.path.join(args.output_dir, "responses")
    scores_root = os.path.join(args.output_dir, "scores")
    os.makedirs(scores_root, exist_ok=True)

    responses_by_model: Dict[str, List[Optional[Tuple[str, str]]]] = {}

    for model_id in models:
        alias_tag = alias_tag_by_model[model_id]
        out_path = os.path.join(responses_root, f"{alias_tag}_generattion.jsonl")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Missing responses file for {model_id}: {out_path}")
        responses_by_model[model_id] = load_existing_responses(out_path, n_rows)

    for model_id in models:
        if any(v is None for v in responses_by_model[model_id]):
            raise ValueError(f"Missing responses for model {model_id} in {responses_root}. Re-run generation.")

    prompt_file = Path(__file__).parent / "prompt_preference_5score_explanation.txt"
    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    judge_llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.judge_world_size,
    )
    set_seed(args.judge_seed)
    sampling_params = SamplingParams(
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        top_k=args.judge_top_k,
        n=args.judge_n_samples,
        max_tokens=args.judge_max_tokens,
        seed=args.judge_seed,
        guided_decoding=PREFERENCE_5SCORE_GUIDED_DECODING,
    )

    for model_a, model_b in combinations(models, 2):
        tag_a = alias_tag_by_model[model_a]
        tag_b = alias_tag_by_model[model_b]
        pair_name = f"{tag_a}_vs_{tag_b}"
        out_path = os.path.join(scores_root, f"{pair_name}.jsonl")
        ckpt_root = os.path.join(scores_root, f"{pair_name}_checkpoints")

        responses_a = responses_by_model[model_a]
        responses_b = responses_by_model[model_b]

        a1_list = [r[0] for r in responses_a]
        a2_list = [r[1] for r in responses_a]
        b1_list = [r[0] for r in responses_b]
        b2_list = [r[1] for r in responses_b]

        exp_prompts = [eds[i][args.prompt_column] for i in range(n_expanded)]
        exp_checks = [eds[i]["check"] for i in range(n_expanded)]
        exp_a1 = [a1_list[eds[i]["orig_index"]] for i in range(n_expanded)]
        exp_a2 = [a2_list[eds[i]["orig_index"]] for i in range(n_expanded)]
        exp_b1 = [b1_list[eds[i]["orig_index"]] for i in range(n_expanded)]
        exp_b2 = [b2_list[eds[i]["orig_index"]] for i in range(n_expanded)]

        pair_results: Dict[str, List] = {}
        load_pair_checkpoints(ckpt_root, n_expanded, pair_results)

        combos = [
            ("a1_b1", exp_a1, exp_b1),
            ("a1_b2", exp_a1, exp_b2),
            ("a2_b1", exp_a2, exp_b1),
            ("a2_b2", exp_a2, exp_b2),
        ]

        for label, resp_a, resp_b in combos:
            if label + "_mean" in pair_results and label + "_majority" in pair_results:
                continue
            mean_vals, maj_vals = run_pair_and_reduce(
                prompts=exp_prompts,
                checks=exp_checks,
                resp_a_list=resp_a,
                resp_b_list=resp_b,
                label=label,
                llm=judge_llm,
                tokenizer=judge_tokenizer,
                prompt_template=prompt_template,
                sampling_params=sampling_params,
                switch_position=args.switch_position,
            )
            pair_results[label + "_mean"] = mean_vals
            pair_results[label + "_majority"] = maj_vals
            save_pair_checkpoint(ckpt_root, label, n_expanded, pair_results)

        tmp_out = out_path + ".tmp"
        with open(tmp_out, "w") as f:
            idx_map: Dict[int, List[int]] = {}
            for i in range(n_expanded):
                oi = eds[i]["orig_index"]
                idx_map.setdefault(oi, []).append(i)
            skip_prefixes = (
                "response_",
                "selection_response_",
                "base_response_",
                "current_response_",
                "adversary_response_",
            )
            skip_exact = {"selection", "base", "current"}
            for idx in range(n_rows):
                row = {
                    "idx": idx,
                    "model_a": model_a,
                    "model_b": model_b,
                    "response_a_1": a1_list[idx],
                    "response_a_2": a2_list[idx],
                    "response_b_1": b1_list[idx],
                    "response_b_2": b2_list[idx],
                }
                for key in ds.column_names:
                    if key not in row:
                        if key in skip_exact:
                            continue
                        if key.startswith(skip_prefixes):
                            continue
                        row[key] = ds[idx][key]
                for label, _, _ in combos:
                    mean_vals = pair_results.get(label + "_mean", [])
                    maj_vals = pair_results.get(label + "_majority", [])
                    mean_list = []
                    maj_list = []
                    for exp_i in idx_map.get(idx, []):
                        mean_list.append(mean_vals[exp_i] if exp_i < len(mean_vals) else None)
                        maj_list.append(maj_vals[exp_i] if exp_i < len(maj_vals) else None)
                    row[f"score_{label}_mean"] = mean_list
                    row[f"score_{label}_majority"] = maj_list
                f.write(json.dumps(row) + "\n")
        os.replace(tmp_out, out_path)
        print(f"Wrote {n_rows} rows -> {out_path}")

        if args.push_to_hub:
            repo_id = resolve_scores_repo(args.scores_repo_template, alias_by_model[model_a], alias_by_model[model_b])
            repo_id = apply_postfix(repo_id, args.hf_postfix)
            rows = []
            with open(out_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        for key in ("selection", "base", "current"):
                            row.pop(key, None)
                        rows.append(row)
            try:
                Dataset.from_list(rows).push_to_hub(repo_id)
                print(f"Pushed scores -> {repo_id}")
            except Exception as e:
                print(f"Failed to push scores for {pair_name}: {e}")

    print(f"Time taken: {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
