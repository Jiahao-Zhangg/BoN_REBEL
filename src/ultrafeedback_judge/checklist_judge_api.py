import os
from collections import Counter
from pathlib import Path

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

    return parser.parse_args()

def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def judge(
    client, judge_type, prompt_template, guided_decoding, dataset, total_pairs, 
    max_tokens, world_size, n_reward_samples, temperature, top_p, judge_model
):
    for i in range(0, total_pairs, 1 if judge_type.startswith("reward") else 2):
        print(f'gathering reward for {i+1}th response')

        # prompts for judge
        if judge_type.startswith("reward"):
            prompts = [
                get_message(
                    prompt_template.format(
                        prompt=row['prompt'], 
                        response=row[f'response_{i}'],
                    )
                ) for row in tqdm(dataset)
            ]
        elif judge_type.startswith("preference"):
            prompts = [
                get_message(
                    prompt_template.format(
                        prompt=row['prompt'], 
                        response_a=row[f'response_{i}'], 
                        response_b=row[f'response_{i+1}'], 
                        check=row['check']
                    )
                ) for row in tqdm(dataset)
            ]
        # start generate
        responses = []
        for prompt_idx, prompt in enumerate(prompts):
            # try:
            content = []
            for j in range(n_reward_samples):
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

        # add to dataset
        # Process responses to match dataset length
        output_majority = []
        output_mean = []
        for j, response_list in enumerate(responses):
            if response_list is not None:
                if judge_type.startswith("reward") or judge_type == "preference_score" or judge_type == "preference_5score":
                    # Filter responses that can be converted to numbers
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
                            print(f"Warning: No valid responses for mean calculation (all -1) for prompt {j+1}")
                            output_mean.append(None)
                    else:
                        print(f"Warning: No valid numeric responses for prompt {j+1}")
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
                                print(f"Warning: No valid responses for mean calculation (all -1) for prompt {j+1}")
                                output_mean.append(None)
                        else:
                            print(f"Warning: No valid numeric responses for prompt {j+1}")
                            output_majority.append(None)
                            output_mean.append(None)
                    else:
                        # For categorical preferences (A, B, Tie), filter and find majority
                        valid_responses = []
                        for response in response_list:
                            if response in ["A", "B", "Tie"]:
                                valid_responses.append(response)
                        
                        if valid_responses:
                            # Use the existing get_winner function to find majority
                            parsed_value = get_winner(valid_responses)
                            output_majority.append(parsed_value)
                            # For categorical preference types, mean doesn't apply
                            output_mean.append(None)
                        else:
                            print(f"Warning: No valid preference responses for prompt {j+1}")
                            output_majority.append(None)
                            output_mean.append(None)
            else:
                output_majority.append(None)
                output_mean.append(None)

        if judge_type.startswith("reward"):
            dataset = dataset.add_column(f"response_{i}_judged_reward_majority", output_majority)
            dataset = dataset.add_column(f"response_{i}_judged_reward_mean", output_mean)
            # filter out invalid judgements (None's) - filter based on majority column
            dataset = dataset.filter(lambda row: row[f"response_{i}_judged_reward_majority"] is not None)
        elif judge_type.startswith("preference"):
            dataset = dataset.add_column(f"response_{i}_{i+1}_judged_preference_majority", output_majority)
            dataset = dataset.add_column(f"response_{i}_{i+1}_judged_preference_mean", output_mean)
            # filter out invalid judgements (None's) - filter based on majority column
            dataset = dataset.filter(lambda row: row[f"response_{i}_{i+1}_judged_preference_majority"] is not None)

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
                # Initialize preference lists for both majority and mean
                for i in range(0, total_pairs, 2):
                    if f'response_{i}_{i+1}_judged_preference_majority' in row:
                        merged_data[prompt][f'response_{i}_{i+1}_judged_preference_majority'] = []
                        merged_data[prompt][f'response_{i}_{i+1}_judged_preference_mean'] = []
            
            # Append judged preferences (both majority and mean) to the corresponding lists
            for i in range(0, total_pairs, 2):
                if f'response_{i}_{i+1}_judged_preference_majority' in row and row[f'response_{i}_{i+1}_judged_preference_majority'] is not None:
                    if f'response_{i}_{i+1}_judged_preference_majority' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}_{i+1}_judged_preference_majority'] = []
                    merged_data[prompt][f'response_{i}_{i+1}_judged_preference_majority'].append(row[f'response_{i}_{i+1}_judged_preference_majority'])
                
                if f'response_{i}_{i+1}_judged_preference_mean' in row and row[f'response_{i}_{i+1}_judged_preference_mean'] is not None:
                    if f'response_{i}_{i+1}_judged_preference_mean' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}_{i+1}_judged_preference_mean'] = []
                    merged_data[prompt][f'response_{i}_{i+1}_judged_preference_mean'].append(row[f'response_{i}_{i+1}_judged_preference_mean'])
        
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
            new_row = dict(row)
            new_row['check'] = req.split('(importance:')[0].strip()
            new_row['response_0'] = row['response_0']
            new_row['response_1'] = row['response_1']
            expanded_data.append(new_row)

    # Create new dataset from expanded data
    dataset = Dataset.from_list(expanded_data)

    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            # api_key="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",
            api_key="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",
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
    )
    
    dataset.push_to_hub(args.input_repo + f'_judge_{args.judge_type}'+'_o3_ver1')
    print(f'time taken: {time.time() - st}')


if __name__ == "__main__":
    main()
