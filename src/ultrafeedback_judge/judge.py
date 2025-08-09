from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

import argparse
import torch
import random
import numpy as np


REWARD_PROMPT = """Please examine the following prompt-response pair and provide a score from 0 to 100 for the quality of the response:

Prompt:
'''
{prompt}
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


PREFERENCE_PROMPT = """Please examine the following prompt and two potential responses (A and B). Select the response with the highest quality. 

Prompt:
'''
{prompt}
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
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--judge_type", type=str, default="reward", choices=["reward", "preference"])
    parser.add_argument("--input_repo", type=str, default="zjhhhh/Whole-Data-Llama-3.2-3B-Instruct-20_armo_tokenized_Llama-3.2-3B-Instruct_slice30")
    parser.add_argument("--selection_pairs", type=int, default=10, help="number of pairs to use for selecting chosen/reject responses")
    parser.add_argument("--gradient_pairs", type=int, default=10, help="number of pairs to use for gradient estimation")
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
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.world_size,
    )

    # dataset
    dataset = load_dataset(args.input_repo, split='train')

    total_pairs = args.selection_pairs + args.gradient_pairs
    for i in range(0, total_pairs, 1 if args.judge_type == "reward" else 2):
        print(f'gathering reward for {i+1}th response')

        # prompts for judge
        if args.judge_type == "reward":
            prompts = [
                get_message(
                    REWARD_PROMPT.format(prompt=row['prompt'], response=row[f'response_{i}'])
                ) for row in tqdm(dataset)
            ]
        elif args.judge_type == "preference":
            prompts = [
                get_message(
                    PREFERENCE_PROMPT.format(prompt=row['prompt'], response_a=row[f'response_{i}'], response_b=row[f'response_{i+1}'])
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

    dataset.push_to_hub(args.input_repo + f'_judge_{args.judge_type}')


if __name__ == "__main__":
    main()
