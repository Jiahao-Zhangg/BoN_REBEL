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
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--judge_type", type=str, default="preference_5score", choices=["preference_binary", "preference_ternary", "preference_score", "preference_5score"])
    parser.add_argument("--input_repo", type=str, default="viswavi/wildchecklists")
    parser.add_argument("--selection_pairs", type=int, default=3, help="number of selection responses")
    parser.add_argument("--base_pairs", type=int, default=2, help="number of base responses")
    parser.add_argument("--current_pairs", type=int, default=2, help="number of current responses")
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
    llm, tokenizer, judge_type, prompt_template, guided_decoding, dataset, 
    max_tokens, world_size, n_samples, temperature, top_p, top_k, switch_position,
    selection_pairs, base_pairs, current_pairs,
):
    # Define the response columns and their types dynamically based on arguments
    selection_responses = [f'selection_response_{i+1}' for i in range(selection_pairs)]
    base_responses = [f'base_response_{j+1}' for j in range(base_pairs)]
    current_responses = [f'current_response_{k+1}' for k in range(current_pairs)]
    
    # Process selection vs base comparisons
    for i, selection_col in enumerate(selection_responses):
        for j, base_col in enumerate(base_responses):
            print(f'gathering preference for {selection_col} vs {base_col}')
            
            prompts = [
                get_message(
                    prompt_template.format(
                        prompt=row['prompt'], 
                        response_a=row[selection_col], 
                        response_b=row[base_col], 
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

            # add to dataset
            output = list(map(lambda x: [r.text for r in x.outputs], response))    # Get responses
            output = list(map(lambda x: [r for r in x if r is not None], output))  # Filter out None's
            output = list(map(lambda x: [extract_verdict(r) for r in x], output))  # Get verdict from json
            output = list(map(lambda x: filter_valid_responses(x, judge_type), output))  # Filter invalid responses

            if judge_type in ["preference_score", "preference_5score"]:
                output = list(map(lambda x: [int(r) for r in x], output))
                if judge_type == "preference_5score":
                    output = list(map(lambda x: [r for r in x if r != -1], output))  # Filter out -1 values for 5score
                output = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output))  # Average the scored preferences
            else:
                output = list(map(lambda x: get_winner(x) if len(x) > 0 else None, output))  # Pick most common answer

            # If switch_position is enabled, also collect preferences in reversed direction
            if switch_position:
                print(f'gathering preference for {base_col} vs {selection_col} (switched)')
                
                prompts_switched = [
                    get_message(
                        prompt_template.format(
                            prompt=row['prompt'], 
                            response_a=row[base_col], 
                            response_b=row[selection_col], 
                            check=row['check']
                        )
                    ) for row in tqdm(dataset)
                ]
                prompts_switched = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts_switched]

                # Generate switched responses
                response_switched = llm.generate(prompts_switched, sampling_params)
                
                # Process switched responses
                output_switched = list(map(lambda x: [r.text for r in x.outputs], response_switched))    # Get responses
                output_switched = list(map(lambda x: [r for r in x if r is not None], output_switched))
                output_switched = list(map(lambda x: [extract_verdict(r) for r in x], output_switched))  # Get verdict from json
                output_switched = list(map(lambda x: filter_valid_responses(x, judge_type), output_switched))

                if judge_type in ["preference_score", "preference_5score"]:
                    output_switched = list(map(lambda x: [int(r) for r in x], output_switched))
                    if judge_type == "preference_5score":
                        output_switched = list(map(lambda x: [r for r in x if r != -1], output_switched))  # Filter out -1 values for 5score
                    output_switched = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output_switched))
                else:
                    output_switched = list(map(lambda x: get_winner(x) if len(x) > 0 else None, output_switched))

                # Reverse the switched scores and combine with original
                output_switched_reversed = [reverse_score(score, judge_type) if score is not None else None for score in output_switched]

                print(output)
                print(output_switched_reversed)
                print("--------------------------------")
                # Combine original and reversed scores (double the samples)
                if judge_type in ["preference_score", "preference_5score"]:
                    # For numeric scores, average the original and reversed scores
                    combined_output = []
                    for orig, rev in zip(output, output_switched_reversed):
                        if orig is not None and rev is not None:
                            combined_output.append((orig + rev) / 2)
                        elif orig is not None:
                            combined_output.append(orig)
                        elif rev is not None:
                            combined_output.append(rev)
                        else:
                            combined_output.append(None)
                    output = combined_output
                else:
                    # For categorical scores, extend the list to include both samples
                    # This effectively doubles the sample size for get_winner calculation
                    extended_samples = []
                    orig_texts = [[extract_verdict(r.text) for r in result.outputs] for result in response]
                    switched_texts = [[extract_verdict(r.text) for r in result.outputs] for result in response_switched]
                    for orig_samples, switched_samples in zip(orig_texts, switched_texts):
                        # Filter and reverse each switched sample and combine with original
                        orig_filtered = filter_valid_responses([s for s in orig_samples if s is not None], judge_type)
                        switched_filtered = filter_valid_responses([s for s in switched_samples if s is not None], judge_type)
                        reversed_switched = [reverse_score(sample, judge_type) for sample in switched_filtered]
                        combined_samples = orig_filtered + reversed_switched
                        extended_samples.append(combined_samples)
                    # Recompute winner with doubled samples
                    output = list(map(lambda x: get_winner(x) if len(x) > 0 else None, extended_samples))
                print(output)

            column_name = f"selection_{i+1}_base_{j+1}_judged_preference"
            dataset = dataset.add_column(column_name, output)
            # filter out invalid judgements (None's)
            dataset = dataset.filter(lambda row: row[column_name] is not None)

    # Process current vs base comparisons
    for k, current_col in enumerate(current_responses):
        for j, base_col in enumerate(base_responses):
            print(f'gathering preference for {current_col} vs {base_col}')
            
            prompts = [
                get_message(
                    prompt_template.format(
                        prompt=row['prompt'], 
                        response_a=row[current_col], 
                        response_b=row[base_col], 
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

            # add to dataset
            output = list(map(lambda x: [r.text for r in x.outputs], response))    # Get responses
            output = list(map(lambda x: [r for r in x if r is not None], output))  # Filter out None's
            output = list(map(lambda x: [extract_verdict(r) for r in x], output))  # Get verdict from json
            output = list(map(lambda x: filter_valid_responses(x, judge_type), output))  # Filter invalid responses

            if judge_type in ["preference_score", "preference_5score"]:
                output = list(map(lambda x: [int(r) for r in x], output))
                if judge_type == "preference_5score":
                    output = list(map(lambda x: [r for r in x if r != -1], output))  # Filter out -1 values for 5score
                output = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output))  # Average the scored preferences
            else:
                output = list(map(lambda x: get_winner(x) if len(x) > 0 else None, output))  # Pick most common answer

            # If switch_position is enabled, also collect preferences in reversed direction
            if switch_position:
                print(f'gathering preference for {base_col} vs {current_col} (switched)')
                
                prompts_switched = [
                    get_message(
                        prompt_template.format(
                            prompt=row['prompt'], 
                            response_a=row[base_col], 
                            response_b=row[current_col], 
                            check=row['check']
                        )
                    ) for row in tqdm(dataset)
                ]
                prompts_switched = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts_switched]

                # Generate switched responses
                response_switched = llm.generate(prompts_switched, sampling_params)
                
                # Process switched responses
                output_switched = list(map(lambda x: [r.text for r in x.outputs], response_switched))    # Get responses
                output_switched = list(map(lambda x: [r for r in x if r is not None], output_switched))
                output_switched = list(map(lambda x: [extract_verdict(r) for r in x], output_switched))  # Get verdict from json
                output_switched = list(map(lambda x: filter_valid_responses(x, judge_type), output_switched))

                if judge_type in ["preference_score", "preference_5score"]:
                    output_switched = list(map(lambda x: [int(r) for r in x], output_switched))
                    if judge_type == "preference_5score":
                        output_switched = list(map(lambda x: [r for r in x if r != -1], output_switched))  # Filter out -1 values for 5score
                    output_switched = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output_switched))
                else:
                    output_switched = list(map(lambda x: get_winner(x) if len(x) > 0 else None, output_switched))

                # Reverse the switched scores and combine with original
                output_switched_reversed = [reverse_score(score, judge_type) if score is not None else None for score in output_switched]

                print(output)
                print(output_switched_reversed)
                print("--------------------------------")
                # Combine original and reversed scores (double the samples)
                if judge_type in ["preference_score", "preference_5score"]:
                    # For numeric scores, average the original and reversed scores
                    combined_output = []
                    for orig, rev in zip(output, output_switched_reversed):
                        if orig is not None and rev is not None:
                            combined_output.append((orig + rev) / 2)
                        elif orig is not None:
                            combined_output.append(orig)
                        elif rev is not None:
                            combined_output.append(rev)
                        else:
                            combined_output.append(None)
                    output = combined_output
                else:
                    # For categorical scores, extend the list to include both samples
                    # This effectively doubles the sample size for get_winner calculation
                    extended_samples = []
                    orig_texts = [[extract_verdict(r.text) for r in result.outputs] for result in response]
                    switched_texts = [[extract_verdict(r.text) for r in result.outputs] for result in response_switched]
                    for orig_samples, switched_samples in zip(orig_texts, switched_texts):
                        # Filter and reverse each switched sample and combine with original
                        orig_filtered = filter_valid_responses([s for s in orig_samples if s is not None], judge_type)
                        switched_filtered = filter_valid_responses([s for s in switched_samples if s is not None], judge_type)
                        reversed_switched = [reverse_score(sample, judge_type) for sample in switched_filtered]
                        combined_samples = orig_filtered + reversed_switched
                        extended_samples.append(combined_samples)
                    # Recompute winner with doubled samples
                    output = list(map(lambda x: get_winner(x) if len(x) > 0 else None, extended_samples))
                print(output)

            column_name = f"current_{k+1}_base_{j+1}_judged_preference"
            dataset = dataset.add_column(column_name, output)
            # filter out invalid judgements (None's)
            dataset = dataset.filter(lambda row: row[column_name] is not None)

    # Group by prompt and merge the judged preference columns
    merged_data = {}
    
    # Define all possible preference column names dynamically
    all_preference_columns = []
    # Selection vs base preferences
    for i in range(selection_pairs):
        for j in range(base_pairs):
            all_preference_columns.append(f'selection_{i+1}_base_{j+1}_judged_preference')
    # Current vs base preferences  
    for k in range(current_pairs):
        for j in range(base_pairs):
            all_preference_columns.append(f'current_{k+1}_base_{j+1}_judged_preference')
    
    for row in dataset:
        prompt = row['prompt']
        if prompt not in merged_data:
            # Initialize with the first occurrence, keeping all non-preference columns
            merged_data[prompt] = {k: v for k, v in row.items() if k not in all_preference_columns}
            # Initialize preference lists for all preference columns
            for col_name in all_preference_columns:
                if col_name in row:
                    merged_data[prompt][col_name] = []
        
        # Append judged preferences to the corresponding lists
        for col_name in all_preference_columns:
            if col_name in row and row[col_name] is not None:
                if col_name not in merged_data[prompt]:
                    merged_data[prompt][col_name] = []
                merged_data[prompt][col_name].append(row[col_name])
    
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

    dataset = judge(
        llm,
        tokenizer,
        args.judge_type,
        prompt_template,
        guided_decoding,
        dataset,
        args.max_tokens,
        args.world_size,
        args.n_samples,
        args.temperature,
        args.top_p,
        args.top_k,
        args.switch_position,
        args.selection_pairs,
        args.base_pairs,
        args.current_pairs,
    )

    # Print column names to show what preferences were computed
    print("Computed preference columns:")
    for col in dataset.column_names:
        if 'judged_preference' in col:
            print(f"  {col}")
    
    # Example: if you want to compute correlation for a specific preference pair, you can do:
    # ground_truth = [...]  # Your ground truth data
    # preds_flat = [item for sublist in dataset['selection_1_base_1_judged_preference'] for item in sublist]
    # correlation_matrix = np.corrcoef(gt_flat, preds_flat)
    # pearson_r = correlation_matrix[0, 1]
    # print(f'Pearson correlation: {pearson_r}')

    # import pdb; pdb.set_trace()
    dataset.push_to_hub('zjhhhh/' + f'{args.judge_model.replace("/", "_")}_temp_{args.temperature}_maxtok_{args.max_tokens}')
    print(f'time taken: {time.time() - st}')


if __name__ == "__main__":
    main()