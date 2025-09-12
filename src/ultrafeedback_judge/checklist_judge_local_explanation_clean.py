import os
from collections import Counter
from pathlib import Path
from typing import Literal, List, Optional

from datasets import load_dataset, Dataset
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

import argparse
import json
import torch
import time
import random
import numpy as np


class PreferenceBinaryOutput(BaseModel):
    explanation: str
    verdict: Literal["A", "B"]
PREFERENCE_BINARY_GUIDED_DECODING = GuidedDecodingParams(json=PreferenceBinaryOutput.model_json_schema())

class PreferenceTernaryOutput(BaseModel):
    explanation: str
    verdict: Literal["A", "B", "Tie"]
PREFERENCE_TERNARY_GUIDED_DECODING = GuidedDecodingParams(json=PreferenceTernaryOutput.model_json_schema())

class PreferenceScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=0, le=10)
PREFERENCE_SCORE_GUIDED_DECODING = GuidedDecodingParams(json=PreferenceScoreOutput.model_json_schema())

class Preference5ScoreOutput(BaseModel):
    explanation: str
    verdict: int = Field(ge=-1, le=4)
PREFERENCE_5SCORE_GUIDED_DECODING = GuidedDecodingParams(json=Preference5ScoreOutput.model_json_schema())


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_winner(values):
    counts = Counter(values)
    counts = {k: counts.get(k, 0) for k in ["A", "B", "Tie"]}

    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]

    # Case 1: all three tied
    if len(winners) == 3:
        return "Tie"

    # Case 2: Tie between 'Tie' and another letter -> return the letter
    if "Tie" in winners and len(winners) == 2:
        return next(k for k in winners if k != "Tie")

    # Case 3: Tie between 'a' and 'b' (no 'Tie') -> return 'Tie'
    if set(winners) == {"A", "B"}:
        return "Tie"

    # Otherwise: unique mode
    return winners[0]


def get_numeric_mode(values, score_range=None):
    """
    Get the mode (most frequent value) from a list of numeric values.
    For numeric scores ranging from 0 to max_score.
    
    Args:
        values: List of numeric values
        score_range: Tuple of (min_score, max_score) for the valid range
        
    Returns:
        The most frequent value, or the median of tied values if multiple modes exist
    """
    if not values:
        return None
    
    # Convert to integers and filter valid values if score_range is provided
    if score_range:
        min_score, max_score = score_range
        values = [int(v) for v in values if min_score <= int(v) <= max_score]
    else:
        values = [int(v) for v in values]
    
    if not values:
        return None
    
    counts = Counter(values)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    
    # If there's a unique mode, return it
    if len(modes) == 1:
        return modes[0]
    
    # If there are multiple modes, return the median of the modes
    # This provides a reasonable tie-breaking strategy for numeric values
    modes.sort()
    return modes[len(modes) // 2]


def is_valid_response(response, judge_type):
    """
    Check if a response is valid for the given judge type.
    
    Args:
        response: The response to validate (string)
        judge_type: The type of judge to determine valid responses
        
    Returns:
        bool: True if response is valid, False otherwise
    """
    if judge_type == "reward":
        try:
            score = int(response)
            return 0 <= score <= 100
        except:
            return False
    elif judge_type == "preference_binary":
        return response in ["A", "B"]
    elif judge_type == "preference_ternary":
        return response in ["A", "B", "Tie"]
    elif judge_type == "preference_score":
        try:
            score = int(response)
            return 0 <= score <= 10
        except:
            return False
    elif judge_type == "preference_5score":
        try:
            score = int(response)
            return -1 <= score <= 4
        except:
            return False
    elif judge_type == "preference_5score_ver2":
        try:
            score = int(response)
            return -3 <= score <= 2
        except:
            return False
    else:
        # For unknown types, assume valid to avoid breaking
        return True


def filter_valid_responses(responses, judge_type):
    """
    Filter out invalid responses based on judge type.
    
    Args:
        responses: List of response strings
        judge_type: The type of judge to determine valid responses
        
    Returns:
        List of valid responses only
    """
    return [r for r in responses if is_valid_response(r, judge_type)]


def reverse_score(score, judge_type):
    """
    Reverse the score to handle positional bias when switching response positions.
    
    Args:
        score: The original score (can be string or numeric)
        judge_type: The type of judge to determine how to reverse the score
        
    Returns:
        Reversed score of the same type as input
    """
    if judge_type == "preference_5score":
        # For 5score (-1 to 4), reverse using 4-x (but keep -1 as -1)
        if score == -1:
            return -1
        else:
            return 4 - int(score)
    elif judge_type in ["preference_binary", "preference_ternary"]:
        # For categorical preferences, swap A and B, keep Tie
        if score == "A":
            return "B"
        elif score == "B":
            return "A"
        else:  # "Tie"
            return "Tie"
    elif judge_type == "preference_5score_ver2":
        # For 5score_ver2 (-3 to 2), reverse using 2-x (but keep -3 as -3)
        if score == -3:
            return -3
        else:
            return - int(score)
    else:
        # For other preference types, return as is
        return score


def extract_verdict(response):
    try:
        parsed = json.loads(response)
    except:
        return None
    verdict = parsed.get('verdict', None)
    return verdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--judge_type", type=str, default="preference_5score", choices=["preference_binary", "preference_ternary", "preference_score", "preference_5score"])
    parser.add_argument("--input_repo", type=str, default="zjhhhh/human-scored-1.5B")
    parser.add_argument("--selection_pairs", type=int, default=2, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=0, help="number of pairs to use for gradient estimation")
    parser.add_argument("--max_tokens", type=int, default=256, help="max tokens to generate by the judge model")
    parser.add_argument("--world_size", type=int, default=4)

    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--switch_position", action="store_true", default=False, help="collect preferences in both directions to handle positional bias")

    return parser.parse_args()


def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def extract_verdict(response):
    try:
        parsed = json.loads(response)
    except:
        return None
    verdict = parsed.get('verdict', None)
    return verdict


def judge(
    llm, tokenizer, judge_type, prompt_template, guided_decoding, dataset, total_pairs, 
    max_tokens, world_size, n_samples, temperature, top_p, top_k, switch_position,
):
    # Process all pairs for preference comparison
    for i in range(total_pairs):
        for j in range(i + 1, total_pairs):
            # Collect preferences in original direction (i vs j)
            print(f'gathering preference for response {i+1} vs {j+1}')
            
            prompts = [
                get_message(
                    prompt_template.format(
                        prompt=row['prompt'], 
                        response_a=row[f'response_{i}'], 
                        response_b=row[f'response_{j}'], 
                        check=row['check']
                    )
                ) for row in tqdm(dataset)
            ]
            prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

            # generate verdict
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
            response = llm.generate(prompts, sampling_params)

            # Collect all samples (original + switched if enabled)
            all_samples = []
            
            # Process original responses
            orig_texts = [[extract_verdict(r.text) for r in result.outputs] for result in response]
            for orig_samples in orig_texts:
                # Filter valid responses
                orig_filtered = filter_valid_responses([s for s in orig_samples if s is not None], judge_type)
                all_samples.append(orig_filtered)

            # If switch_position is enabled, also collect preferences in reversed direction
            if switch_position:
                print(f'gathering preference for response {j+1} vs {i+1} (switched)')
                
                prompts_switched = [
                    get_message(
                        prompt_template.format(
                            prompt=row['prompt'], 
                            response_a=row[f'response_{j}'], 
                            response_b=row[f'response_{i}'], 
                            check=row['check']
                        )
                    ) for row in tqdm(dataset)
                ]
                prompts_switched = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts_switched]

                # Generate switched responses
                response_switched = llm.generate(prompts_switched, sampling_params)
                
                # Process switched responses and combine with original
                switched_texts = [[extract_verdict(r.text) for r in result.outputs] for result in response_switched]
                for idx, switched_samples in enumerate(switched_texts):
                    # Filter and reverse each switched sample
                    switched_filtered = filter_valid_responses([s for s in switched_samples if s is not None], judge_type)
                    reversed_switched = [reverse_score(sample, judge_type) for sample in switched_filtered]
                    # Combine with original samples
                    all_samples[idx].extend(reversed_switched)

            # Now calculate mean and majority from all combined samples
            if judge_type in ["preference_score", "preference_5score", "reward"]:
                # For numeric scores, calculate both mean and majority
                output_mean = []
                output_majority = []
                
                # Determine score range based on judge type
                if judge_type == "preference_5score":
                    score_range = (-1, 4)
                elif judge_type == "preference_score":
                    score_range = (0, 10)
                elif judge_type == "reward":
                    score_range = (0, 100)
                else:
                    score_range = None
                
                for samples in all_samples:
                    if len(samples) > 0:
                        numeric_samples = [int(s) for s in samples]
                        output_mean.append(sum(numeric_samples) / len(numeric_samples))
                        output_majority.append(get_numeric_mode(samples, score_range))
                    else:
                        output_mean.append(None)
                        output_majority.append(None)
            else:
                # For categorical scores, majority is the winner
                output_mean = []
                output_majority = []
                for samples in all_samples:
                    if len(samples) > 0:
                        winner = get_winner(samples)
                        output_mean.append(winner)
                        output_majority.append(winner)
                    else:
                        output_mean.append(None)
                        output_majority.append(None)

            print(f"Combined samples mean: {output_mean[:5]}...")  # Show first 5 for debugging
            print(f"Combined samples majority: {output_majority[:5]}...")
            print("--------------------------------")

            dataset = dataset.add_column(f"response_{i}_{j}_judged_preference_mean", output_mean)
            dataset = dataset.add_column(f"response_{i}_{j}_judged_preference_majority", output_majority)
            # filter out invalid judgements (None's) - filter based on mean column
            dataset = dataset.filter(lambda row: row[f"response_{i}_{j}_judged_preference_mean"] is not None)

    # Group by prompt and merge the judged preference columns
    merged_data = {}
    
    for row in dataset:
        prompt = row['prompt']
        if prompt not in merged_data:
            # Initialize with the first occurrence, keeping all non-response columns
            merged_data[prompt] = {k: v for k, v in row.items() if not k.startswith('response_') or not '_judged_preference' in k}
            # Initialize preference lists for all pairs
            for i in range(total_pairs):
                for j in range(i + 1, total_pairs):
                    if f'response_{i}_{j}_judged_preference_mean' in row:
                        merged_data[prompt][f'response_{i}_{j}_judged_preference_mean'] = []
                        merged_data[prompt][f'response_{i}_{j}_judged_preference_majority'] = []
        
        # Append judged preferences to the corresponding lists
        for i in range(total_pairs):
            for j in range(i + 1, total_pairs):
                if f'response_{i}_{j}_judged_preference_mean' in row and row[f'response_{i}_{j}_judged_preference_mean'] is not None:
                    if f'response_{i}_{j}_judged_preference_mean' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}_{j}_judged_preference_mean'] = []
                        merged_data[prompt][f'response_{i}_{j}_judged_preference_majority'] = []
                    merged_data[prompt][f'response_{i}_{j}_judged_preference_mean'].append(row[f'response_{i}_{j}_judged_preference_mean'])
                    merged_data[prompt][f'response_{i}_{j}_judged_preference_majority'].append(row[f'response_{i}_{j}_judged_preference_majority'])
    
    # Convert back to dataset
    dataset = Dataset.from_list(list(merged_data.values()))

    return dataset


def main():
    # init
    st = time.time()
    args = parse_arguments()

    # prompt template
    if args.judge_type == "preference_binary":
        filename = "prompt_preference_binary.txt"
        guided_decoding = PREFERENCE_BINARY_GUIDED_DECODING
    elif args.judge_type == "preference_ternary":
        filename = "prompt_preference_ternary.txt"
        guided_decoding = PREFERENCE_TERNARY_GUIDED_DECODING
    elif args.judge_type == "preference_score":
        filename = "prompt_preference_score.txt"
        guided_decoding = PREFERENCE_SCORE_GUIDED_DECODING
    elif args.judge_type == "preference_5score":
        filename = "prompt_preference_5score_explanation.txt"
        guided_decoding = PREFERENCE_5SCORE_GUIDED_DECODING
    elif args.judge_type == "preference_5score_ver2":
        filename = "prompt_preference_5score_ver2.txt"
        guided_decoding = PREFERENCE_5SCORE_GUIDED_DECODING
    with open(Path(__file__).parent / filename, "r") as f:
        prompt_template = f.read()

    # dataset
    dataset = load_dataset(args.input_repo, split='train')
    
    # Split requirements and create new rows
    expanded_data = []
    for row in dataset:
        # Loop and extract requirements
        requirements_str: str = row['requirements']
        counter = 1
        requirements = []
        while len(requirements_str) > 0:
            assert requirements_str.startswith(f"{counter})")
            if requirements_str.find(f"/100)\n{counter+1})") > 0:
                curr_requirement = requirements_str[len(f"{counter})"):requirements_str.find(f"/100)\n{counter+1})") + len("/100)\n")]
            else:
                curr_requirement = requirements_str[len(f"{counter})"):]
            requirements.append(curr_requirement)
            requirements_str = requirements_str[len(curr_requirement) + len(f"{counter})"):]
            counter += 1
        requirements = list(map(lambda x: x.strip(), requirements))

        for req in requirements:
            new_row = dict(row)
            new_row['check'] = req.split('(importance:')[0].strip()
            new_row['importance'] = int(req.split('(importance:')[1].split('/')[0].strip())

            expanded_data.append(new_row)

    # Create new dataset from expanded data
    dataset = Dataset.from_list(expanded_data)

    # load model
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
    )

    ground_truth = [
        [2,1,3,2,1,2],
        [4,4,4,3],
        [3,3,3,2],
        [4,2,2,3,3],
        [1,3,2,3],
        [0,0,0,1,2],
        [2,2,2,2,3,4],
        [2,2,2,2],
        [1,3,1,2,3],
        [3,2,2,4,3],
    ]
    gt_flat = [item for sublist in ground_truth for item in sublist]
    preds_flat_mean = [item for sublist in dataset['response_0_1_judged_preference_mean'] for item in sublist]
    preds_flat_majority = [item for sublist in dataset['response_0_1_judged_preference_majority'] for item in sublist]

    # Calculate correlation for both mean and majority
    correlation_matrix_mean = np.corrcoef(gt_flat, preds_flat_mean)
    pearson_r_mean = correlation_matrix_mean[0, 1]
    print(f'Pearson correlation (mean): {pearson_r_mean}')
    
    correlation_matrix_majority = np.corrcoef(gt_flat, preds_flat_majority)
    pearson_r_majority = correlation_matrix_majority[0, 1]
    print(f'Pearson correlation (majority): {pearson_r_majority}')

    # import pdb; pdb.set_trace()
    dataset.push_to_hub(f'MisDrifter/template_2_{args.judge_model}')
    print(f'time taken: {time.time() - st}')


if __name__ == "__main__":
    main()