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


REWARD_PROMPT = """Please examine the following prompt-response pair and provide a score from 0 to 100 for the quality of the response. Please check the key criteria question to determine the score.

Prompt:
'''
{prompt}
'''

Key Criteria Question:
'''
{check}
'''

Response:
'''
{response}
'''

Respond *only* with the numerical score.
"""

REWARD_GUIDED_DECODING = GuidedDecodingParams(choice=[str(i) for i in range(101)])

def parse_reward(text):
    try:
        result = int(text)
        if result < 0 or result > 100:
            return None
        return result
    except:
        return None


PREFERENCE_PROMPT = """Please examine the following prompt and two potential responses (A and B). Select the response with the highest quality. Please check the key criteria question to determine the choice.

Prompt:
'''
{prompt}
'''

Key Criteria Question:
'''
{check}
'''

Response A:
'''
{response_a}
'''

Response B:
'''
{response_b}
'''

Respond *only* with the letter of the response with the highest quality. Ties are not allowed.
"""

PREFERENCE_GUIDED_DECODING = GuidedDecodingParams(choice=["A", "B"])

def parse_preference(text):
    if text.lower() == "a":
        return "A"
    elif text.lower() == "b":
        return "B"
    else:
        return None

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    # parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--judge_type", type=str, default="reward", choices=["reward", "preference"])
    parser.add_argument("--input_repo", type=str, default="viswavi/rlcf")
    parser.add_argument("--selection_pairs", type=int, default=2, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=0, help="number of pairs to use for gradient estimation")
    parser.add_argument("--max_tokens", type=int, default=20, help="max tokens to generate by the judge model")
    parser.add_argument("--world_size", type=int, default=2)
    return parser.parse_args()


def get_message(instruction):
    message = [
        {"role": "user", "content": instruction},
    ]
    return message


def main():
    # init
    st = time.time()
    args = parse_arguments()


    # dataset
    dataset = load_dataset(args.input_repo, split='train')
    dataset = dataset.select(range(10))
    
    # Split requirements and create new rows
    expanded_data = []
    for row in dataset:
        requirements = list(map(lambda x: x[3:].strip(), row['requirements'].split('\n')))
        for req in requirements:
            if req.startswith('Does the response satisfy the following two criteria:'):
                continue
            new_row = dict(row)
            new_row['check'] = req.split('(importance:')[0].strip()
            new_row['response_0'] = dataset['chosen']
            new_row['response_1'] = dataset['rejected']
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
    for i in range(0, total_pairs, 1 if args.judge_type == "reward" else 2):
        print(f'gathering reward for {i+1}th response')

        # prompts for judge
        if args.judge_type == "reward":
            prompts = [
                get_message(
                    REWARD_PROMPT.format(prompt=row['prompt'], response=row[f'response_{i}'], check=row['check'])
                ) for row in tqdm(dataset)
            ]
        elif args.judge_type == "preference":
            prompts = [
                get_message(
                    PREFERENCE_PROMPT.format(prompt=row['prompt'], response_a=row[f'response_{i}'], response_b=row[f'response_{i+1}'], check=row['check'])
                ) for row in tqdm(dataset)
            ]
        prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

        # start generate
        set_seed(0)
        guided = REWARD_GUIDED_DECODING if args.judge_type == "reward" else PREFERENCE_GUIDED_DECODING
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=0,
            guided_decoding=guided,
        )
        response = llm.generate(prompts, sampling_params)

        # add to dataset
        if args.judge_type == "reward":
            output = list(map(lambda x: x.outputs[0].text, response))
            output = list(map(parse_reward, output))
            dataset = dataset.add_column(f"response_{i}_judged_reward", output)
            # filter out invalid judgements (None's)
            dataset = dataset.filter(lambda row: row[f"response_{i}_judged_reward"] is not None)
        elif args.judge_type == "preference":
            output = list(map(lambda x: x.outputs[0].text, response))
            output = list(map(parse_preference, output))
            dataset = dataset.add_column(f"response_{i}_{i+1}_judged_preference", output)
            # filter out invalid judgements (None's)
            dataset = dataset.filter(lambda row: row[f"response_{i}_{i+1}_judged_preference"] is not None)

    # Merge dataset by 'prompt': combine response reward columns into lists
    if args.judge_type == "reward":
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
            
            # Append judged rewards to the corresponding response lists
            for i in range(total_pairs):
                if f'response_{i}_judged_reward' in row and row[f'response_{i}_judged_reward'] is not None:
                    if f'response_{i}' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}'] = []
                    merged_data[prompt][f'response_{i}'].append(row[f'response_{i}_judged_reward'])
        
        # Convert back to dataset
        dataset = Dataset.from_list(list(merged_data.values()))
    
    elif args.judge_type == "preference":
        # Group by prompt and merge the judged preference columns
        merged_data = {}
        
        for row in dataset:
            prompt = row['prompt']
            if prompt not in merged_data:
                # Initialize with the first occurrence, keeping all non-response columns
                merged_data[prompt] = {k: v for k, v in row.items() if not k.startswith('response_') or not '_judged_preference' in k}
                # Initialize preference lists
                for i in range(0, total_pairs, 2):
                    if f'response_{i}_{i+1}_judged_preference' in row:
                        merged_data[prompt][f'response_{i}_{i+1}_judged_preference'] = []
            
            # Append judged preferences to the corresponding lists
            for i in range(0, total_pairs, 2):
                if f'response_{i}_{i+1}_judged_preference' in row and row[f'response_{i}_{i+1}_judged_preference'] is not None:
                    if f'response_{i}_{i+1}_judged_preference' not in merged_data[prompt]:
                        merged_data[prompt][f'response_{i}_{i+1}_judged_preference'] = []
                    merged_data[prompt][f'response_{i}_{i+1}_judged_preference'].append(row[f'response_{i}_{i+1}_judged_preference'])
        
        # Convert back to dataset
        dataset = Dataset.from_list(list(merged_data.values()))
    
    # dataset.push_to_hub(args.input_repo + f'_judge_{args.judge_type}')
    dataset.push_to_hub(f"MisDrifter/play_with_checklist_{args.judge_type}")
    print(f'time taken: {time.time() - st}')

if __name__ == "__main__":
    main()
