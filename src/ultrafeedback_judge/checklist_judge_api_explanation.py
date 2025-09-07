import os
from collections import Counter
from pathlib import Path
from typing import Literal, List, Optional

from datasets import load_dataset, Dataset
from pydantic import BaseModel, Field
from tqdm import tqdm
from openai import OpenAI

import argparse
import time
import random
import numpy as np


# Pydantic models for structured outputs
class RewardOutput(BaseModel):
    score: int = Field(ge=0, le=100, description="Reward score from 0 to 100")

class PreferenceBinaryOutput(BaseModel):
    choice: Literal["A", "B"] = Field(description="Preferred response: A or B")

class PreferenceTernaryOutput(BaseModel):
    choice: Literal["A", "B", "Tie"] = Field(description="Preferred response: A, B, or Tie")

class PreferenceScoreOutput(BaseModel):
    score: int = Field(ge=0, le=10, description="Preference score from 0 to 10")

class Preference5ScoreOutput(BaseModel):
    score: int = Field(ge=-1, le=4, description="Preference score from -1 to 4")

class ExplanationOutput(BaseModel):
    explanation: str = Field(description="Detailed explanation of the judgment")
    verdict: str = Field(description="Final verdict or score")


## You can also do more complex guided decoding with Pydantic. E.g.:
# from pydantic import BaseModel, Field
# from typing import Literal, List, Optional
# class JudgeOutput(BaseModel):
#     verdict: Literal["A", "B"]
#     confidence: float = Field(ge=0, le=1)
#     reasons: Optional[List[str]] = None
# guided = GuidedDecodingParams(json=JudgeOutput.model_json_schema())


def get_response_format(judge_type):
    """Get the appropriate response format for structured output based on judge type."""
    if judge_type == "reward":
        return RewardOutput
    elif judge_type == "preference_binary":
        return PreferenceBinaryOutput
    elif judge_type == "preference_ternary":
        return PreferenceTernaryOutput
    elif judge_type == "preference_score":
        return PreferenceScoreOutput
    elif judge_type == "preference_5score":
        return Preference5ScoreOutput
    else:
        return None


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)


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
    parser.add_argument("--judge_model", type=str, default="openai/o3")
    parser.add_argument("--judge_type", type=str, default="reward", choices=["reward", "preference_binary", "preference_ternary", "preference_score", "preference_5score"])
    parser.add_argument("--input_repo", type=str, default="viswavi/wildchecklists")
    parser.add_argument("--selection_pairs", type=int, default=2, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=0, help="number of pairs to use for gradient estimation")
    parser.add_argument("--max_tokens", type=int, default=20, help="max tokens to generate by the judge model")
    parser.add_argument("--world_size", type=int, default=2)

    parser.add_argument("--n_reward_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--switch_position", action="store_true", default=False, help="collect preferences in both directions to handle positional bias")

    parser.add_argument("--explanation", action="store_true", default=False, help="generate an explanation before judgement")
    parser.add_argument("--explanation_max_tokens", type=int, default=256, help="max tokens to generate for the explanation")

    return parser.parse_args()


def get_message(instruction):
    return [{"role": "user", "content": instruction}]


def cleanup_explanation(text):
    """Clean up incomplete JSON from explanation generation and prepare for verdict generation."""
    import re
    
    # Replace different ending patterns and standardize whitespace
    if re.search(r'"\s*,\s*"\s*$', text):
        # Ends with `","` (with any amount of whitespace) - replace with standardized version
        return re.sub(r'"\s*,\s*"\s*$', '", "verdict":', text)
    elif re.search(r'"\s*,\s*$', text):
        # Ends with `",` - replace with standardized version
        return re.sub(r'"\s*,\s*$', '", "verdict":', text)
    elif re.search(r'"\s*$', text):
        # Ends with `"` (with any amount of whitespace) - replace with standardized version
        return re.sub(r'"\s*$', '", "verdict":', text)
    else:
        # Fallback: add the full structure
        return text + '", "verdict":'

    
def judge(
    client, judge_type, prompt_template, guided_decoding, dataset, total_pairs, 
    max_tokens, world_size, n_reward_samples, temperature, top_p, judge_model, switch_position,
    explanation, explanation_max_tokens
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

            # Get response format for structured output
            response_format = get_response_format(judge_type)
            
            # Process each prompt individually with API calls
            responses = []
            for prompt_idx, prompt in enumerate(prompts):
                content = []
                for k in range(n_reward_samples):
                    if explanation:
                        # First generate explanation
                        explanation_response = client.chat.completions.parse(
                            model=judge_model,
                            messages=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=explanation_max_tokens,
                            response_format=ExplanationOutput
                        )
                        explanation_data = explanation_response.choices[0].message.parsed
                        content.append(explanation_data.verdict)
                    else:
                        # Direct verdict generation without explanation
                        response = client.chat.completions.parse(
                            model=judge_model,
                            messages=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            response_format=response_format
                        )
                        response_data = response.choices[0].message.parsed
                        
                        # Extract the actual value from structured response
                        if judge_type == "reward":
                            content.append(str(response_data.score))
                        elif judge_type in ["preference_score", "preference_5score"]:
                            content.append(str(response_data.score))
                        else:
                            content.append(response_data.choice)
                
                responses.append(content)
                print(f"Prompt {prompt_idx+1}/{len(prompts)}: {content}")

            # Process responses (already in the right format from API calls)
            output = list(map(lambda x: filter_valid_responses(x, judge_type), responses))  # Filter invalid responses

            if judge_type in ["preference_score", "preference_5score"]:
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
                            response_a=row[f'response_{i}'], 
                            response_b=row[f'response_{j}'], 
                            check=row['check']
                        )
                    ) for row in tqdm(dataset)
                ]

                # Get response format for structured output
                response_format = get_response_format(judge_type)
                
                # Process each prompt individually with API calls
                responses = []
                for prompt_idx, prompt in enumerate(prompts):
                    content = []
                    for k in range(n_reward_samples):
                        if explanation:
                            # First generate explanation
                            explanation_response = client.chat.completions.parse(
                                model=judge_model,
                                messages=prompt,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=explanation_max_tokens,
                                response_format=ExplanationOutput
                            )
                            explanation_data = explanation_response.choices[0].message.parsed
                            content.append(explanation_data.verdict)
                        else:
                            # Direct verdict generation without explanation
                            response = client.chat.completions.parse(
                                model=judge_model,
                                messages=prompt,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=max_tokens,
                                response_format=response_format
                            )
                            response_data = response.choices[0].message.parsed
                            
                            # Extract the actual value from structured response
                            if judge_type in ["reward", "preference_score", "preference_5score"]:
                                content.append(str(response_data.score))
                            else:
                                content.append(response_data.choice)
                    
                    responses.append(content)
                    print(f"Prompt {prompt_idx+1}/{len(prompts)}: {content}")

                # Process responses (already in the right format from API calls)
                output = list(map(lambda x: filter_valid_responses(x, judge_type), responses))  # Filter invalid responses

                if judge_type in ["preference_score", "preference_5score"]:
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
                                response_a=row[f'response_{j}'], 
                                response_b=row[f'response_{i}'], 
                                check=row['check']
                            )
                        ) for row in tqdm(dataset)
                    ]

                    # Generate switched responses with API calls
                    responses_switched = []
                    for prompt_idx, prompt in enumerate(prompts_switched):
                        content = []
                        for k in range(n_reward_samples):
                            if explanation:
                                # First generate explanation
                                explanation_response = client.chat.completions.parse(
                                    model=judge_model,
                                    messages=prompt,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=explanation_max_tokens,
                                    response_format=ExplanationOutput
                                )
                                explanation_data = explanation_response.choices[0].message.parsed
                                content.append(explanation_data.verdict)
                            else:
                                # Direct verdict generation without explanation
                                response = client.chat.completions.parse(
                                    model=judge_model,
                                    messages=prompt,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_tokens,
                                    response_format=response_format
                                )
                                response_data = response.choices[0].message.parsed
                                
                                # Extract the actual value from structured response
                                if judge_type in ["reward", "preference_score", "preference_5score"]:
                                    content.append(str(response_data.score))
                                else:
                                    content.append(response_data.choice)
                        
                        responses_switched.append(content)
                        print(f"Switched Prompt {prompt_idx+1}/{len(prompts_switched)}: {content}")
                    
                    # Process switched responses (already in the right format from API calls)
                    output_switched = list(map(lambda x: filter_valid_responses(x, judge_type), responses_switched))  # Filter invalid responses

                    if judge_type in ["preference_score", "preference_5score"]:
                        output_switched = list(map(lambda x: [int(r) for r in x], output_switched))
                        output_switched = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else None, output_switched))
                    else:
                        output_switched = list(map(lambda x: get_winner(x) if len(x) > 0 else None, output_switched))

                    # Reverse the switched scores and combine with original
                    output_switched_reversed = [reverse_score(score, judge_type) if score is not None else None for score in output_switched]
                    
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
                        for orig_samples, switched_samples in zip(responses, responses_switched):
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
    elif args.judge_type == "preference_binary":
        filename = "prompt_preference_binary.txt"
    elif args.judge_type == "preference_ternary":
        filename = "prompt_preference_ternary.txt"
    elif args.judge_type == "preference_score":
        filename = "prompt_preference_score.txt"
    elif args.judge_type == "preference_5score":
        filename = "prompt_preference_5score_explanation.txt"
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
            # Keep existing response_0 and response_1 fields as they are
            expanded_data.append(new_row)

    # Create new dataset from expanded data
    dataset = Dataset.from_list(expanded_data)

    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    total_pairs = args.selection_pairs + args.gradient_pairs
    dataset = judge(
        client,
        args.judge_type,
        prompt_template,
        None,  # guided_decoding not used with API
        dataset,
        total_pairs,
        args.max_tokens,
        args.world_size,
        args.n_reward_samples,
        args.temperature,
        args.top_p,
        args.judge_model,
        args.switch_position,
        args.explanation,
        args.explanation_max_tokens,
    )

    dataset.push_to_hub(args.input_repo + f'_judge_{args.judge_type}_with_explanation')
    print(f'time taken: {time.time() - st}')


if __name__ == "__main__":
    main()