import re
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Literal, List, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class PreferenceBinaryOutput(BaseModel):
    explanation: str
    verdict: Literal["A", "B"]


PREFERENCE_BINARY_GUIDED_DECODING = GuidedDecodingParams(
    json=PreferenceBinaryOutput.model_json_schema()
)


class PreferenceTernaryOutput(BaseModel):
    explanation: str
    verdict: Literal["A", "B", "Tie"]


PREFERENCE_TERNARY_GUIDED_DECODING = GuidedDecodingParams(
    json=PreferenceTernaryOutput.model_json_schema()
)


class PreferenceScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=0, le=10)


PREFERENCE_SCORE_GUIDED_DECODING = GuidedDecodingParams(
    json=PreferenceScoreOutput.model_json_schema()
)


class Preference5ScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=-1, le=4)


PREFERENCE_5SCORE_GUIDED_DECODING = GuidedDecodingParams(
    json=Preference5ScoreOutput.model_json_schema()
)


def set_seed(seed: int = 5775709):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_winner(values: List[str]):
    counts = Counter(values)
    counts = {k: counts.get(k, 0) for k in ["A", "B", "Tie"]}

    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]

    if len(winners) == 3:
        return "Tie"
    if "Tie" in winners and len(winners) == 2:
        return next(k for k in winners if k != "Tie")
    if set(winners) == {"A", "B"}:
        return "Tie"
    return winners[0]


def is_valid_response(response, judge_type):
    if judge_type == "reward":
        try:
            score = int(response)
            return 0 <= score <= 100
        except Exception:
            return False
    elif judge_type == "preference_binary":
        return response in ["A", "B"]
    elif judge_type == "preference_ternary":
        return response in ["A", "B", "Tie"]
    elif judge_type == "preference_score":
        try:
            score = int(response)
            return 0 <= score <= 10
        except Exception:
            return False
    elif judge_type == "preference_5score":
        try:
            score = int(response)
            return -1 <= score <= 4
        except Exception:
            return False
    else:
        return True


def filter_valid_responses(responses, judge_type):
    return [r for r in responses if is_valid_response(r, judge_type)]


def reverse_score(score, judge_type):
    if judge_type == "preference_5score":
        if score == -1:
            return -1
        else:
            return 4 - int(score)
    elif judge_type in ["preference_binary", "preference_ternary"]:
        if score == "A":
            return "B"
        elif score == "B":
            return "A"
        else:
            return "Tie"
    else:
        return score


def extract_verdict(response_text: str):
    try:
        parsed = json.loads(response_text)
    except Exception:
        return None
    return parsed.get("verdict", None)


def get_message(instruction: str):
    return [{"role": "user", "content": instruction}]


def get_numeric_mode(values, score_range=None):
    """
    Get the mode (most frequent value) from a list of numeric values.
    If multiple modes exist, return the mean of the tied modes.
    """
    if not values:
        return None

    if score_range is not None:
        min_s, max_s = score_range
        values = [int(v) for v in values if min_s <= int(v) <= max_s]
    else:
        values = [int(v) for v in values]

    if not values:
        return None

    counts = Counter(values)
    max_c = max(counts.values())
    modes = [k for k, c in counts.items() if c == max_c]
    if len(modes) == 1:
        return modes[0]
    return float(sum(modes) / len(modes))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Comprehensive checklist judge scoring on full requirements (no per-check expansion)."
    )
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--judge_type",
        type=str,
        default="preference_5score",
        choices=["preference_binary", "preference_ternary", "preference_score", "preference_5score"],
    )
    parser.add_argument("--input_repo", type=str, default="zjhhhh/human-scored-1.5B")
    parser.add_argument("--output_repo", type=str, default="zjhhhh/fullchecks_ternary_intransitivity_large")
    parser.add_argument("--selection_pairs", type=int, default=8, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=0, help="number of pairs to use for gradient estimation")
    parser.add_argument("--max_tokens", type=int, default=256, help="max tokens to generate by the judge model")
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--fixed_criteria", action="store_true", default=False)

    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--switch_position",
        action="store_true",
        default=False,
        help="collect preferences in both directions to handle positional bias",
    )

    return parser.parse_args()


def remove_existing_judged_columns(dataset: Dataset) -> Dataset:
    judged_cols = [c for c in dataset.column_names if re.match(r"response_\d+_\d+_judged_.*", c)]
    if judged_cols:
        dataset = dataset.remove_columns(judged_cols)
    return dataset

FIXED_CRITERIA = "1) Does the response satisfy the following two criteria: 1) The response directly address the request without excessive or off-topic information not necessary for addressing the user's instruction? 2) The response should match the context and the instruction, whether it requires professionalism, friendliness, formality, or neutrality. (importance: 100/100)"
def judge(
    llm,
    tokenizer,
    judge_type,
    prompt_template,
    guided_decoding,
    dataset,
    total_pairs,
    max_tokens,
    world_size,
    n_samples,
    temperature,
    top_p,
    top_k,
    switch_position,
    fixed_criteria,
):
    required_response_cols = [f"response_{idx}" for idx in range(total_pairs)]
    for col in required_response_cols:
        if col not in dataset.column_names:
            raise ValueError(f"Missing required column '{col}' in dataset.")

    dataset = remove_existing_judged_columns(dataset)

    set_seed(0)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=n_samples,
        max_tokens=max_tokens,
        seed=0,
        guided_decoding=guided_decoding,
    )
    if fixed_criteria:
        dataset = dataset.map(lambda row: {"requirements": FIXED_CRITERIA})
    # Process all pairs for preference comparison without per-check expansion
    for i in range(total_pairs):
        for j in range(i + 1, total_pairs):
            print(f"gathering preference for response {i+1} vs {j+1}")

            prompts = []
            for row in dataset:
                filled = prompt_template.format(
                    prompt=row["prompt"],
                    response_a=row[f"response_{i}"],
                    response_b=row[f"response_{j}"],
                    check=row.get("requirements", row.get("check", "")),
                )
                messages = get_message(filled)
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            responses = llm.generate(prompts, sampling_params)
            texts = [[o.text for o in r.outputs] for r in responses]

            verdict_lists = []
            for per_row in texts:
                vals = [extract_verdict(t) for t in per_row]
                vals = [v for v in vals if v is not None]
                vals = filter_valid_responses(vals, judge_type)
                verdict_lists.append(vals)

            if judge_type in ["preference_score", "preference_5score", "reward"]:
                orig_mean = []
                orig_majority = []
                for vals in verdict_lists:
                    if len(vals) == 0:
                        orig_mean.append(None)
                        orig_majority.append(None)
                    else:
                        ints = [int(v) for v in vals]
                        if judge_type == "preference_5score":
                            ints_no_missing = [x for x in ints if x != -1]
                            if len(ints_no_missing) == 0:
                                orig_mean.append(None)
                            else:
                                orig_mean.append(float(np.mean(ints_no_missing)))
                            score_range = (0, 4)
                            orig_majority.append(get_numeric_mode(vals, score_range))
                        else:
                            orig_mean.append(float(np.mean(ints)))
                            if judge_type == "preference_score":
                                score_range = (0, 10)
                            elif judge_type == "reward":
                                score_range = (0, 100)
                            else:
                                score_range = None
                            orig_majority.append(get_numeric_mode(vals, score_range))

                reduced_mean = orig_mean
                reduced_majority = orig_majority
            else:
                winners = [get_winner(vals) if len(vals) > 0 else None for vals in verdict_lists]
                reduced_majority = winners
                reduced_mean = winners

            if switch_position:
                print(f"gathering preference for response {j+1} vs {i+1} (switched)")
                prompts_sw = []
                for row in dataset:
                    filled = prompt_template.format(
                        prompt=row["prompt"],
                        response_a=row[f"response_{j}"],
                        response_b=row[f"response_{i}"],
                        check=row.get("requirements", row.get("check", "")),
                    )
                    messages = get_message(filled)
                    prompts_sw.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

                responses_sw = llm.generate(prompts_sw, sampling_params)
                texts_sw = [[o.text for o in r.outputs] for r in responses_sw]
                verdict_lists_sw = []
                for per_row in texts_sw:
                    vals = [extract_verdict(t) for t in per_row]
                    vals = [v for v in vals if v is not None]
                    vals = filter_valid_responses(vals, judge_type)
                    verdict_lists_sw.append(vals)

                if judge_type in ["preference_score", "preference_5score", "reward"]:
                    sw_mean = []
                    sw_majority = []
                    for sw_vals in verdict_lists_sw:
                        if len(sw_vals) == 0:
                            sw_mean.append(None)
                            sw_majority.append(None)
                        else:
                            sw_ints = [int(v) for v in sw_vals]
                            sw_ints = [reverse_score(v, judge_type) for v in sw_ints]
                            if judge_type == "preference_5score":
                                sw_no_missing = [x for x in sw_ints if x != -1]
                                if len(sw_no_missing) == 0:
                                    sw_mean.append(None)
                                else:
                                    sw_mean.append(float(np.mean(sw_no_missing)))
                                score_range = (0, 4)
                                sw_majority.append(get_numeric_mode(sw_no_missing, score_range))
                            else:
                                sw_mean.append(float(np.mean(sw_ints)))
                                if judge_type == "preference_score":
                                    score_range = (0, 10)
                                elif judge_type == "reward":
                                    score_range = (0, 100)
                                else:
                                    score_range = None
                                sw_majority.append(get_numeric_mode(sw_ints, score_range))

                    new_mean = []
                    new_majority = []
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
                else:
                    winners_sw = []
                    for vals_b in verdict_lists_sw:
                        rb = [reverse_score(v, judge_type) for v in vals_b]
                        winners_sw.append(get_winner(rb) if len(rb) > 0 else None)

                    winners_final = []
                    for wo, ws in zip(reduced_mean, winners_sw):
                        if wo is None and ws is None:
                            winners_final.append(None)
                        elif wo is None:
                            winners_final.append(ws)
                        elif ws is None:
                            winners_final.append(wo)
                        else:
                            winners_final.append(wo if wo == ws else "Tie")

                    reduced_majority = winners_final
                    reduced_mean = winners_final

            mean_col = f"response_{i}_{j}_judged_preference_mean"
            maj_col = f"response_{i}_{j}_judged_preference_majority"

            print(f"Combined samples mean (first 5): {reduced_mean[:5]}...")
            print(f"Combined samples majority (first 5): {reduced_majority[:5]}...")
            print("--------------------------------")

            dataset = dataset.add_column(mean_col, reduced_mean)
            dataset = dataset.add_column(maj_col, reduced_majority)
            dataset = dataset.filter(lambda row: row[mean_col] is not None)

    return dataset


def main():
    st = time.time()
    args = parse_arguments()

    if args.judge_type == "preference_binary":
        filename = "prompt_preference_binary.txt"
        guided_decoding = PREFERENCE_BINARY_GUIDED_DECODING
    elif args.judge_type == "preference_ternary":
        # filename = "prompt_preference_ternary.txt"
        filename = "prompt_preference_fullcheck_ternary.txt"
        guided_decoding = PREFERENCE_TERNARY_GUIDED_DECODING
    elif args.judge_type == "preference_score":
        filename = "prompt_preference_score.txt"
        guided_decoding = PREFERENCE_SCORE_GUIDED_DECODING
    elif args.judge_type == "preference_5score":
        # Use the full-checks template to score comprehensively on all requirements
        filename = "prompt_preference_5score_fullchecks.txt"
        guided_decoding = PREFERENCE_5SCORE_GUIDED_DECODING
    else:
        raise ValueError(f"Unsupported judge_type: {args.judge_type}")

    with open(Path(__file__).parent / filename, "r") as f:
        prompt_template = f.read()

    dataset = load_dataset(args.input_repo, split="train")
    print(f"Loaded dataset with {len(dataset)} rows from {args.input_repo}")

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.world_size,
    )

    total_pairs = args.selection_pairs + args.gradient_pairs
    dataset = judge(
        llm,
        tokenizer,
        args.judge_type,
        prompt_template,
        guided_decoding,
        dataset,
        total_pairs,
        args.max_tokens,
        args.world_size,
        args.n_samples,
        args.temperature,
        args.top_p,
        args.top_k,
        args.switch_position,
        args.fixed_criteria,
    )

    dataset.push_to_hub(args.output_repo)
    print(f"time taken: {time.time() - st}")


if __name__ == "__main__":
    main()
