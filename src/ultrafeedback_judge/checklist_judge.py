import os
from collections import Counter
from pathlib import Path

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
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--judge_type", type=str, default="reward", choices=["reward", "preference_binary", "preference_ternary", "preference_score"])
    parser.add_argument("--input_repo", type=str, default="viswavi/wildchecklists")
    parser.add_argument("--selection_pairs", type=int, default=2, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=0, help="number of pairs to use for gradient estimation")
    parser.add_argument("--max_tokens", type=int, default=20, help="max tokens to generate by the judge model")
    parser.add_argument("--world_size", type=int, default=2)

    parser.add_argument("--n_reward_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--switch_position", action="store_true", default=False, help="collect preferences in both directions to handle positional bias")

    return parser.parse_args()


def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def judge(
    llm, tokenizer, judge_type, prompt_template, guided_decoding, dataset, total_pairs, 
    max_tokens, world_size, n_reward_samples, temperature, top_p, switch_position
):
    if judge_type.startswith("reward"):
        # Process each response individually for reward
        for i in range(total_pairs):
            print(f'gathering reward for {i+1}th response')

            prompts = [
                get_message(
                    prompt_template.format(
                        prompt=row['prompt'], 
                        response=row[f'response_{i}'][1]["content"], 
                        check=row['check']
                    )
                ) for row in tqdm(dataset)
            ]
            prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

            # start generate
            set_seed(0)
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                n=n_reward_samples,
                max_tokens=max_tokens,
                seed=0,
                guided_decoding=guided_decoding,
            )
            response = llm.generate(prompts, sampling_params)

            # add to dataset
            output = list(map(lambda x: [r.text for r in x.outputs], response))    # Get responses
            output = list(map(lambda x: [r for r in x if r is not None], output))  # Filter out None's
            output = list(map(lambda x: filter_valid_responses(x, judge_type), output))  # Filter invalid responses

            if judge_type == "preference_score":
                output = list(map(lambda x: [int(r) for r in x], output))
                output = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output))  # Average the scored preferences
            else:
                output = list(map(lambda x: [int(r) for r in x], output))
                output = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output))  # Average the rewards

            dataset = dataset.add_column(f"response_{i}_judged_reward", output)
            # filter out invalid judgements (None's)
            dataset = dataset.filter(lambda row: row[f"response_{i}_judged_reward"] is not None)

    elif judge_type.startswith("preference"):
        # Process all pairs for preference comparison
        for i in range(total_pairs):
            for j in range(i + 1, total_pairs):
                # Collect preferences in original direction (i vs j)
                print(f'gathering preference for response {i+1} vs {j+1}')
                
                prompts = [
                    get_message(
                        prompt_template.format(
                            prompt=row['prompt'], 
                            response_a=row[f'response_{i}'][1]["content"], 
                            response_b=row[f'response_{j}'][1]["content"], 
                            check=row['check']
                        )
                    ) for row in tqdm(dataset)
                ]
                prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

                # start generate
                set_seed(0)
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    n=n_reward_samples,
                    max_tokens=max_tokens,
                    seed=0,
                    guided_decoding=guided_decoding,
                )
                response = llm.generate(prompts, sampling_params)

                # add to dataset
                output = list(map(lambda x: [r.text for r in x.outputs], response))    # Get responses
                output = list(map(lambda x: [r for r in x if r is not None], output))  # Filter out None's
                output = list(map(lambda x: filter_valid_responses(x, judge_type), output))  # Filter invalid responses

                if judge_type == "preference_score":
                    output = list(map(lambda x: [int(r) for r in x], output))
                    output = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output))  # Average the scored preferences
                else:
                    output = list(map(lambda x: get_winner(x) if len(x) > 0 else None, output))  # Pick most common answer

                # If switch_position is enabled, also collect preferences in reversed direction
                if switch_position:
                    print(f'gathering preference for response {j+1} vs {i+1} (switched)')
                    
                    prompts_switched = [
                        get_message(
                            prompt_template.format(
                                prompt=row['prompt'], 
                                response_a=row[f'response_{j}'][1]["content"], 
                                response_b=row[f'response_{i}'][1]["content"], 
                                check=row['check']
                            )
                        ) for row in tqdm(dataset)
                    ]
                    prompts_switched = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts_switched]

                    # Generate switched responses
                    response_switched = llm.generate(prompts_switched, sampling_params)
                    
                    # Process switched responses
                    output_switched = list(map(lambda x: [r.text for r in x.outputs], response_switched))
                    output_switched = list(map(lambda x: [r for r in x if r is not None], output_switched))
                    output_switched = list(map(lambda x: filter_valid_responses(x, judge_type), output_switched))

                    if judge_type == "preference_score":
                        output_switched = list(map(lambda x: [int(r) for r in x], output_switched))
                        output_switched = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output_switched))
                    else:
                        output_switched = list(map(lambda x: get_winner(x) if len(x) > 0 else None, output_switched))

                    # Reverse the switched scores and combine with original
                    output_switched_reversed = [reverse_score(score, judge_type) if score is not None else None for score in output_switched]
                    
                    # Combine original and reversed scores (double the samples)
                    if judge_type == "preference_score":
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
                        for orig_samples, switched_samples in zip(
                            list(map(lambda x: [r.text for r in x.outputs], response)),
                            list(map(lambda x: [r.text for r in x.outputs], response_switched))
                        ):
                            # Filter and reverse each switched sample and combine with original
                            orig_filtered = filter_valid_responses([s for s in orig_samples if s is not None], judge_type)
                            switched_filtered = filter_valid_responses([s for s in switched_samples if s is not None], judge_type)
                            reversed_switched = [reverse_score(sample, judge_type) for sample in switched_filtered]
                            combined_samples = orig_filtered + reversed_switched
                            extended_samples.append(combined_samples)
                        
                        # Recompute winner with doubled samples
                        output = list(map(lambda x: get_winner(x) if len(x) > 0 else None, extended_samples))

                dataset = dataset.add_column(f"response_{i}_{j}_judged_preference", output)
                # filter out invalid judgements (None's)
                dataset = dataset.filter(lambda row: row[f"response_{i}_{j}_judged_preference"] is not None)

    # Merge dataset by 'prompt': combine response reward columns into lists
    if judge_type.startswith("reward"):
        # Group by prompt and merge the judged reward columns
        merged_data = {}
        
        for row in dataset:
            prompt = row['prompt']
            if prompt not in merged_data:
                # Initialize with the first occurrence, keeping all non-response columns
                merged_data[prompt] = {k: v for k, v in row.items() if not k.startswith('response_') or not k.endswith('_judged_reward')}
                # Initialize response lists
                for i in range(total_pairs):
                    if f'response_{i}' in row:
                        merged_data[prompt][f'response_{i}'] = []
            
            # Append judged rewards and importance to the corresponding response lists
            for i in range(total_pairs):
                if f'response_{i}_judged_reward' in row and row[f'response_{i}_judged_reward'] is not None:
                    if f'response_{i}' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}'] = []
                    merged_data[prompt][f'response_{i}'].append(row[f'response_{i}_judged_reward'])

        # Convert back to dataset
        dataset = Dataset.from_list(list(merged_data.values()))

    elif judge_type.startswith("preference"):
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
                        if f'response_{i}_{j}_judged_preference' in row:
                            merged_data[prompt][f'response_{i}_{j}_judged_preference'] = []
            
            # Append judged preferences to the corresponding lists
            for i in range(total_pairs):
                for j in range(i + 1, total_pairs):
                    if f'response_{i}_{j}_judged_preference' in row and row[f'response_{i}_{j}_judged_preference'] is not None:
                        if f'response_{i}_{j}_judged_preference' not in merged_data[prompt]:
                            merged_data[prompt][f'response_{i}_{j}_judged_preference'] = []
                        merged_data[prompt][f'response_{i}_{j}_judged_preference'].append(row[f'response_{i}_{j}_judged_preference'])
        
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
    with open(Path(__file__).parent / filename, "r") as f:
        prompt_template = f.read()

    # dataset
    dataset = load_dataset(args.input_repo, split='train', download_mode="force_redownload")
    
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
            new_row['response_0'] = row['chosen']
            new_row['response_1'] = row['rejected']
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
        args.n_reward_samples,
        args.temperature,
        args.top_p,
        args.switch_position,
    )

    dataset.push_to_hub(args.input_repo + f'_judge_{args.judge_type}')
    print(f'time taken: {time.time() - st}')


if __name__ == "__main__":
    main()
