import os
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI
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
import asyncio
from typing import List, Any


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
    parser.add_argument("--judge_model", type=str, default="qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--switch_position", action="store_true", default=False, help="collect preferences in both directions to handle positional bias")
    parser.add_argument("--batch_size", type=int, default=10, help="number of concurrent API calls to make")
    parser.add_argument("--max_concurrent", type=int, default=50, help="maximum number of concurrent API calls")

    return parser.parse_args()

def get_message(instruction):
    return [{"role": "user", "content": instruction}]


async def make_api_call_async(client: AsyncOpenAI, prompt: List[dict], judge_model: str, semaphore: asyncio.Semaphore) -> str:
    """Make a single async API call with semaphore for rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=prompt
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in API call: {e}")
            return None


async def batch_api_calls_async(client: AsyncOpenAI, prompts: List[List[dict]], judge_model: str, n_samples: int, max_concurrent: int) -> List[List[str]]:
    """Make batched async API calls for all prompts and samples."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create all tasks for all prompts and samples
    tasks = []
    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx in range(n_samples):
            task = make_api_call_async(client, prompt, judge_model, semaphore)
            tasks.append((prompt_idx, sample_idx, task))
    
    # Execute all tasks concurrently
    print(f"Making {len(tasks)} concurrent API calls...")
    results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
    # Organize results back into the expected format
    organized_results = [[] for _ in range(len(prompts))]
    for (prompt_idx, sample_idx, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"Exception for prompt {prompt_idx}, sample {sample_idx}: {result}")
            organized_results[prompt_idx].append(None)
        elif result is None:
            print(f"None result for prompt {prompt_idx}, sample {sample_idx}")
            organized_results[prompt_idx].append(None)
        else:
            organized_results[prompt_idx].append(result)
    
    return organized_results


async def judge(
    client, judge_type, prompt_template, guided_decoding, dataset, total_pairs, 
    max_tokens, world_size, n_reward_samples, temperature, top_p, judge_model, switch_position, max_concurrent
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
            
            dataset = await _process_judge_responses_async(client, prompts, dataset, judge_type, i, None, n_reward_samples, judge_model, False, prompt_template, max_concurrent)
    
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
                
                dataset = await _process_judge_responses_async(client, prompts, dataset, judge_type, i, j, n_reward_samples, judge_model, switch_position, prompt_template, max_concurrent)

    return _merge_dataset_results(dataset, judge_type, total_pairs)


async def _process_judge_responses_async(client, prompts, dataset, judge_type, i, j, n_reward_samples, judge_model, switch_position=False, prompt_template=None, max_concurrent=50):
    # Use async batch API calls instead of sequential calls
    responses = await batch_api_calls_async(client, prompts, judge_model, n_reward_samples, max_concurrent)
    
    # Print progress
    for prompt_idx, content in enumerate(responses):
        print(f"Prompt {prompt_idx+1}/{len(prompts)}: {content}")

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
        
        # Generate switched responses using async calls
        responses_switched = await batch_api_calls_async(client, prompts_switched, judge_model, n_reward_samples, max_concurrent)
        
        for prompt_idx, content in enumerate(responses_switched):
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
                        parsed_value = get_winner(response_list)
                        response_list = list(map(lambda x: MAP[x], response_list))
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
                        parsed_value = get_winner(response_list)
                        response_list = list(map(lambda x: MAP[x], response_list))
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


async def main():
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
    # dataset = dataset.select(range(10, 20))
    
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

    client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="",
        )

    # load model
    total_pairs = args.selection_pairs + args.gradient_pairs
    dataset = await judge(
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
        args.max_concurrent,
    )
    
    # dataset.push_to_hub('MisDrifter/' + f'_judge_{args.judge_type}_{args.selection_pairs}pairs_switch')
    dataset.push_to_hub('MisDrifter/try')

    print(f'time taken: {time.time() - st}')


if __name__ == "__main__":
    asyncio.run(main())
