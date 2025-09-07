import os
from collections import Counter
from pathlib import Path
import re

from openai import OpenAI
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

import argparse
import torch
import time
import random
import numpy as np


REWARD_GUIDED_DECODING = GuidedDecodingParams(choice=[str(i) for i in range(101)])

PREFERENCE_BINARY_GUIDED_DECODING = GuidedDecodingParams(choice=["A", "B"])

PREFERENCE_TERNARY_GUIDED_DECODING = GuidedDecodingParams(choice=["A", "B", "Tie"])

PREFERENCE_SCORE_GUIDED_DECODING = GuidedDecodingParams(choice=[str(i) for i in range(11)])

PREFERENCE_5SCORE_GUIDED_DECODING = GuidedDecodingParams(choice=[str(i) for i in range(-1, 5)])


## You can also do more complex guided decoding with Pydantic. E.g.:
# from pydantic import BaseModel, Field
# from typing import Literal, List, Optional
# class JudgeOutput(BaseModel):
#     verdict: Literal["A", "B"]
#     confidence: float = Field(ge=0, le=1)
#     reasons: Optional[List[str]] = None
# guided = GuidedDecodingParams(json_schema=JudgeOutput.model_json_schema())


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_number_from_text(text):
    """
    Extract number only if it appears at the very beginning or very end of the text.
    
    Args:
        text: String that may contain a number with additional text
        
    Returns:
        str: The extracted number as a string, or None if no number found at beginning/end
    """
    if not isinstance(text, str):
        return str(text) if text is not None else None
    
    # First try to convert directly (handles pure numbers)
    text_stripped = text.strip()
    try:
        int(text_stripped)
        return text_stripped
    except ValueError:
        pass
    
    # Look for numbers at the beginning or end only
    # Pattern for number at the beginning: starts with optional whitespace, then number, then non-digit
    beginning_pattern = r'^\s*(-?\d+)(?:\D|$)'
    # Pattern for number at the end: non-digit or start, then number, then optional whitespace at end
    end_pattern = r'(?:^|\D)(-?\d+)\s*$'
    
    # Check beginning first
    beginning_match = re.search(beginning_pattern, text)
    if beginning_match:
        return beginning_match.group(1)
    
    # Check end
    end_match = re.search(end_pattern, text)
    if end_match:
        return end_match.group(1)
    
    return None




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
    Only extracts numbers from text for preference_5score type.
    
    Args:
        response: The response to validate (string)
        judge_type: The type of judge to determine valid responses
        
    Returns:
        tuple: (is_valid, extracted_value) where is_valid is bool and extracted_value is the clean response
    """
    if judge_type == "reward":
        try:
            score = int(response)
            is_valid = 0 <= score <= 100
            return is_valid, str(score) if is_valid else None
        except:
            return False, None
    elif judge_type == "preference_binary":
        response_stripped = response.strip() if isinstance(response, str) else str(response)
        is_valid = response_stripped in ["A", "B"]
        return is_valid, response_stripped if is_valid else None
    elif judge_type == "preference_ternary":
        response_stripped = response.strip() if isinstance(response, str) else str(response)
        is_valid = response_stripped in ["A", "B", "Tie"]
        return is_valid, response_stripped if is_valid else None
    elif judge_type == "preference_score":
        try:
            score = int(response)
            is_valid = 0 <= score <= 10
            return is_valid, str(score) if is_valid else None
        except:
            return False, None
    elif judge_type == "preference_5score":
        # Only for preference_5score: extract numbers from text with additional content
        extracted = extract_number_from_text(response)
        if extracted is None:
            return False, None
        try:
            score = int(extracted)
            is_valid = -1 <= score <= 4
            return is_valid, extracted if is_valid else None
        except:
            return False, None
    else:
        # For unknown types, assume valid to avoid breaking
        return True, response


def filter_valid_responses(responses, judge_type):
    """
    Filter out invalid responses based on judge type and extract clean values.
    
    Args:
        responses: List of response strings
        judge_type: The type of judge to determine valid responses
        
    Returns:
        List of valid, extracted responses only
    """
    valid_responses = []
    for r in responses:
        is_valid, extracted_value = is_valid_response(r, judge_type)
        if is_valid and extracted_value is not None:
            valid_responses.append(extracted_value)
    return valid_responses


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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_type", type=str, default="preference_ternary", choices=["reward", "preference_binary", "preference_ternary", "preference_score", "preference_5score"])
    parser.add_argument("--input_repo", type=str, default="zjhhhh/human-scored-1.5B")
    parser.add_argument("--selection_pairs", type=int, default=2, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=0, help="number of pairs to use for gradient estimation")
    parser.add_argument("--max_tokens", type=int, default=20, help="max tokens to generate by the judge model")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--n_reward_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--judge_model", type=str, default="openai/o3")
    parser.add_argument("--switch_position", action="store_true", default=False, help="collect preferences in both directions to handle positional bias")

    return parser.parse_args()

def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def judge(
    client, judge_type, prompt_template, guided_decoding, dataset, total_pairs, 
    max_tokens, world_size, n_reward_samples, temperature, top_p, judge_model, switch_position
):
    if judge_type.startswith("reward"):
        # Process each response individually for reward
        for i in range(total_pairs):
            print(f'gathering reward for {i+1}th response')

            prompts = [
                get_message(
                    prompt_template.format(
                        prompt=row['prompt'], 
                        response=row[f'response_{i}'],
                        check=row['check']
                    )
                ) for row in tqdm(dataset)
            ]
            
            dataset = _process_judge_responses(client, prompts, dataset, judge_type, i, None, n_reward_samples, judge_model, False, prompt_template)
    
    elif judge_type.startswith("preference"):
        # Process all pairs for preference comparison
        for i in range(total_pairs):
            for j in range(i + 1, total_pairs):
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
                
                dataset = _process_judge_responses(client, prompts, dataset, judge_type, i, j, n_reward_samples, judge_model, switch_position, prompt_template)

    return _merge_dataset_results(dataset, judge_type, total_pairs)


def _process_judge_responses(client, prompts, dataset, judge_type, i, j, n_reward_samples, judge_model, switch_position=False, prompt_template=None):
    # start generate
    responses = []
    for prompt_idx, prompt in enumerate(prompts):
        # try:
        content = []
        for k in range(n_reward_samples):
            response = client.chat.completions.create(
            # model="qwen/qwen2.5-vl-72b-instruct",
            model=judge_model,
            messages=prompt)
            content.append(response.choices[0].message.content.strip())
        # Extract the actual content from the API response
        responses.append(content)
        print(f"Prompt {prompt_idx+1}/{len(prompts)}: {content}")
        # except Exception as e:
        #     print(f"Error processing prompt {i+1}: {e}")
        #     responses.append(None)

    # If switch_position is enabled and this is a preference task, also collect reversed preferences
    if switch_position and judge_type.startswith("preference") and j is not None:
        print(f'gathering preference for response {j+1} vs {i+1} (switched)')
        
        # Create switched prompts
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
        
        # Generate switched responses
        responses_switched = []
        for prompt_idx, prompt in enumerate(prompts_switched):
            content = []
            for k in range(n_reward_samples):
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=prompt)
                content.append(response.choices[0].message.content.strip())
            responses_switched.append(content)
            print(f"Switched Prompt {prompt_idx+1}/{len(prompts_switched)}: {content}")
        
        # Combine original and switched responses
        combined_responses = []
        for orig_content, switched_content in zip(responses, responses_switched):
            # Filter valid responses first
            orig_filtered = filter_valid_responses(orig_content, judge_type)
            switched_filtered = filter_valid_responses(switched_content, judge_type)
            # Reverse the switched responses and combine with original
            reversed_switched = [reverse_score(sample, judge_type) for sample in switched_filtered]
            combined_content = orig_filtered + reversed_switched
            combined_responses.append(combined_content)
        
        responses = combined_responses

    # add to dataset
    # Process responses to match dataset length
    output_majority = []
    output_mean = []
    for response_idx, response_list in enumerate(responses):
        if response_list is not None:
            # Filter valid responses first (unless already filtered for switch_position)
            if not (switch_position and judge_type.startswith("preference") and j is not None):
                response_list = filter_valid_responses(response_list, judge_type)
            
            if judge_type.startswith("reward") or judge_type == "preference_score":
                # Convert to numbers
                valid_responses = []
                for response in response_list:
                    try:
                        parsed_response = int(response)
                        valid_responses.append(parsed_response)
                    except:
                        continue
                
                if valid_responses:
                    # Calculate majority (most common response)
                    from collections import Counter
                    counts = Counter(valid_responses)
                    majority_value = counts.most_common(1)[0][0]
                    output_majority.append(majority_value)
                    
                    # Calculate mean excluding -1 values
                    valid_for_mean = [x for x in valid_responses if x != -1]
                    if valid_for_mean:
                        mean_value = sum(valid_for_mean) / len(valid_for_mean)
                        output_mean.append(mean_value)
                    else:
                        print(f"Warning: No valid responses for mean calculation (all -1) for prompt {response_idx+1}")
                        output_mean.append(None)
                else:
                    print(f"Warning: No valid numeric responses for prompt {response_idx+1}")
                    output_majority.append(None)
                    output_mean.append(None)
                    
            elif judge_type.startswith("preference"):
                if judge_type == "preference_5score":
                    # For preference_5score, treat as numeric and compute both majority and mean
                    valid_responses = []
                    for response in response_list:
                        try:
                            parsed_response = int(response)
                            valid_responses.append(parsed_response)
                        except:
                            continue
                    
                    if valid_responses:
                        # Calculate majority (most common response)
                        from collections import Counter
                        counts = Counter(valid_responses)
                        majority_value = counts.most_common(1)[0][0]
                        output_majority.append(majority_value)
                        
                        # Calculate mean excluding -1 values
                        valid_for_mean = [x for x in valid_responses if x != -1]
                        if valid_for_mean:
                            mean_value = sum(valid_for_mean) / len(valid_for_mean)
                            output_mean.append(mean_value)
                        else:
                            print(f"Warning: No valid responses for mean calculation (all -1) for prompt {response_idx+1}")
                            output_mean.append(None)
                    else:
                        print(f"Warning: No valid numeric responses for prompt {response_idx+1}")
                        output_majority.append(None)
                        output_mean.append(None)
                elif judge_type == "preference_ternary":
                    # For categorical preferences (A, B, Tie), already filtered
                    if response_list:
                        # Use the existing get_winner function to find majority
                        MAP = {'A': 1, 'B': 0, 'Tie': 0.5}
                        response_list = list(map(lambda x: MAP[x], response_list))
                        parsed_value = get_winner(response_list)
                        output_majority.append(parsed_value)
                        # For categorical preference types, mean doesn't apply
                        output_mean.append(np.mean(response_list))
                    else:
                        print(f"Warning: No valid preference responses for prompt {response_idx+1}")
                        output_majority.append(None)
                        output_mean.append(None)
        else:
            output_majority.append(None)
            output_mean.append(None)

    # Add columns to dataset based on judge type
    if judge_type.startswith("reward"):
        dataset = dataset.add_column(f"response_{i}_judged_reward_majority", output_majority)
        dataset = dataset.add_column(f"response_{i}_judged_reward_mean", output_mean)
        # filter out invalid judgements (None's) - filter based on majority column
        dataset = dataset.filter(lambda row: row[f"response_{i}_judged_reward_majority"] is not None)
    elif judge_type.startswith("preference"):
        dataset = dataset.add_column(f"response_{i}_{j}_judged_preference_majority", output_majority)
        dataset = dataset.add_column(f"response_{i}_{j}_judged_preference_mean", output_mean)
        # filter out invalid judgements (None's) - filter based on majority column
        dataset = dataset.filter(lambda row: row[f"response_{i}_{j}_judged_preference_majority"] is not None)
    
    return dataset


def _merge_dataset_results(dataset, judge_type, total_pairs):
    # Merge dataset by 'prompt': combine response reward columns into lists
    if judge_type.startswith("reward"):
        # Group by prompt and merge the judged reward columns
        merged_data = {}
        
        for row in dataset:
            prompt = row['prompt']
            if prompt not in merged_data:
                # Initialize with the first occurrence, keeping all non-response columns
                merged_data[prompt] = {k: v for k, v in row.items() if not k.startswith('response_') or not ('_judged_reward_majority' in k or '_judged_reward_mean' in k)}
                # Initialize response lists for both majority and mean
                for i in range(total_pairs):
                    if f'response_{i}' in row:
                        merged_data[prompt][f'response_{i}_majority'] = []
                        merged_data[prompt][f'response_{i}_mean'] = []
            
            # Append judged rewards (both majority and mean) to the corresponding response lists
            for i in range(total_pairs):
                if f'response_{i}_judged_reward_majority' in row and row[f'response_{i}_judged_reward_majority'] is not None:
                    if f'response_{i}_majority' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}_majority'] = []
                    merged_data[prompt][f'response_{i}_majority'].append(row[f'response_{i}_judged_reward_majority'])
                
                if f'response_{i}_judged_reward_mean' in row and row[f'response_{i}_judged_reward_mean'] is not None:
                    if f'response_{i}_mean' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}_mean'] = []
                    merged_data[prompt][f'response_{i}_mean'].append(row[f'response_{i}_judged_reward_mean'])

        # Convert back to dataset
        dataset = Dataset.from_list(list(merged_data.values()))

    elif judge_type.startswith("preference"):
        # Group by prompt and merge the judged preference columns
        merged_data = {}
        
        for row in dataset:
            prompt = row['prompt']
            if prompt not in merged_data:
                # Initialize with the first occurrence, keeping all non-response columns
                merged_data[prompt] = {k: v for k, v in row.items() if not k.startswith('response_') or not ('_judged_preference_majority' in k or '_judged_preference_mean' in k)}
                # Initialize preference lists for both majority and mean for all pairs
                for i in range(total_pairs):
                    for j in range(i + 1, total_pairs):
                        if f'response_{i}_{j}_judged_preference_majority' in row:
                            merged_data[prompt][f'response_{i}_{j}_judged_preference_majority'] = []
                            merged_data[prompt][f'response_{i}_{j}_judged_preference_mean'] = []
            
            # Append judged preferences (both majority and mean) to the corresponding lists
            for i in range(total_pairs):
                for j in range(i + 1, total_pairs):
                    if f'response_{i}_{j}_judged_preference_majority' in row and row[f'response_{i}_{j}_judged_preference_majority'] is not None:
                        if f'response_{i}_{j}_judged_preference_majority' not in merged_data[prompt]:
                            merged_data[prompt][f'response_{i}_{j}_judged_preference_majority'] = []
                        merged_data[prompt][f'response_{i}_{j}_judged_preference_majority'].append(row[f'response_{i}_{j}_judged_preference_majority'])
                    
                    if f'response_{i}_{j}_judged_preference_mean' in row and row[f'response_{i}_{j}_judged_preference_mean'] is not None:
                        if f'response_{i}_{j}_judged_preference_mean' not in merged_data[prompt]:
                            merged_data[prompt][f'response_{i}_{j}_judged_preference_mean'] = []
                        merged_data[prompt][f'response_{i}_{j}_judged_preference_mean'].append(row[f'response_{i}_{j}_judged_preference_mean'])
        
        # Convert back to dataset
        dataset = Dataset.from_list(list(merged_data.values()))

    return dataset


def main():
    # init
    st = time.time()
    args = parse_arguments()

    # prompt template
    if args.judge_type == "reward":
        filename = "prompt_reward.txt"
        guided_decoding = REWARD_GUIDED_DECODING
    elif args.judge_type == "preference_binary":
        filename = "prompt_preference_binary.txt"
        guided_decoding = PREFERENCE_BINARY_GUIDED_DECODING
    elif args.judge_type == "preference_ternary":
        filename = "prompt_preference_ternary.txt"
        guided_decoding = PREFERENCE_TERNARY_GUIDED_DECODING
    elif args.judge_type == "preference_score":
        filename = "prompt_preference_score.txt"
        guided_decoding = PREFERENCE_SCORE_GUIDED_DECODING
    elif args.judge_type == "preference_5score":
        filename = "prompt_preference_5score.txt"
        guided_decoding = PREFERENCE_5SCORE_GUIDED_DECODING
    with open(Path(__file__).parent / filename, "r") as f:
        prompt_template = f.read()

    # dataset
    # dataset = load_dataset(args.input_repo, split='train', download_mode="force_redownload")
    dataset = load_dataset(args.input_repo, split='train')
    dataset = dataset.select(range(10))
    
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
            temp = dict(row)
            new_row = {'requirements':temp['requirements'], 'prompt':temp['prompt']}
            for i in range(args.selection_pairs):
                new_row[f'response_{i}'] = temp[f'response_{i}']
            new_row['check'] = req.split('(importance:')[0].strip()
            expanded_data.append(new_row)

    # Create new dataset from expanded data
    dataset = Dataset.from_list(expanded_data)

    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="",
        )

    # load model
    total_pairs = args.selection_pairs + args.gradient_pairs
    dataset = judge(
        client,
        args.judge_type,
        prompt_template,
        guided_decoding,
        dataset,
        total_pairs,
        args.max_tokens,
        args.world_size,
        args.n_reward_samples,
        args.temperature,
        args.top_p,
        args.judge_model,
        args.switch_position,
    )
    
    dataset.push_to_hub('zjhhhh/' + f'claude_sonnet_4_ver1_switch')
    print(f'time taken: {time.time() - st}')


if __name__ == "__main__":
    main()
