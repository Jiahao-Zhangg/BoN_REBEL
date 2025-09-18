import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List

torch.set_printoptions(threshold=10_000)

# WARNING: Magic number, make sure it works for your model
SYS_PROMPT_LEN = 24

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--input_repo", type=str, required=True, help="output repo from rank_by_score.py")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--maxlen_prompt", type=int, default=1024)
    return parser.parse_args()


def get_message(instruction=None, response=None):

    assert instruction != None or response != None

    if response == None:
        message = [
            {"role": "user", "content": instruction},
        ]
    elif instruction == None:
        message = [
            {"role": "assistant", "content": response}
        ]
    else:
        message = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]

    return message


def filter_same_responses(row):
    return row['chosen'] != row['reject']


def main():

    # init
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer_left = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer_left.add_special_tokens({"pad_token": "[PAD]"})

    dataset = load_dataset(args.input_repo, split='train')
    
    # process dataset
    print('initial length:', len(dataset))

    # filter dataset with long prompt or response
    dataset = dataset.filter(lambda row: tokenizer.apply_chat_template(get_message(row['prompt']), tokenize=True, add_generation_prompt=True, return_tensors='pt').shape[-1] <= args.maxlen_prompt)
    print('filtered long prompts:', len(dataset))

    # discover response columns from rank_by_score output: any column X with accompanying X_reward
    column_names = set(dataset.column_names)
    response_cols = []
    for name in column_names:
        if name.endswith('_reward'):
            base = name[:-7]
            if base in column_names:
                response_cols.append(base)
    response_cols.sort()
    print('discovered response columns:', response_cols)

    def responses_within_limit(row):
        for rc in response_cols:
            resp = row[rc]
            if resp is None:
                return False
            toks = tokenizer.apply_chat_template(
                get_message(response=resp),
                tokenize=True,
                add_generation_prompt=False,
                return_tensors='pt',
            )[:, SYS_PROMPT_LEN:]
            if toks.shape[-1] > args.maxlen:
                return False
        return True

    dataset = dataset.filter(responses_within_limit)
    print('filtered long responses:', len(dataset))

    # filter responses that don't end with PAD or EOS token after tokenization
    def responses_end_properly(row):
        for rc in response_cols:
            resp = row[rc]
            if resp is None:
                return False
            # tokenize the response the same way as in the main loop
            response_token = tokenizer.apply_chat_template(
                get_message(response=resp),
                add_generation_prompt=False,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen+SYS_PROMPT_LEN,
            )[SYS_PROMPT_LEN:]
            
            # check if last token is PAD or EOS
            if "Qwen" in args.model:
                last_id = int(response_token[-1])
                pid = tokenizer.pad_token_id
                eid = tokenizer.eos_token_id
                if not ((pid is not None and last_id == pid) or (eid is not None and last_id == eid)):
                    return False
        return True

    dataset = dataset.filter(responses_end_properly)
    print('filtered responses not ending with PAD/EOS:', len(dataset))

    # add prompt tokens
    qwen_prompts = []
    qwen_prompt_tokens = []
    for row in tqdm(dataset):
        qwen_prompt_token = tokenizer_left.apply_chat_template(
                get_message(row['prompt']), 
                add_generation_prompt=True,
                tokenize=True,
                padding='max_length',
                max_length=args.maxlen_prompt,
        )
        qwen_prompt = tokenizer_left.decode(qwen_prompt_token, skip_special_tokens=False)
        assert len(qwen_prompt_token) == args.maxlen_prompt
        if "Qwen" in args.model:
            assert ("<|start_header_id|>" in qwen_prompt or "<|im_start|>" in qwen_prompt), "Qwen prompt missing chat header markers"
        qwen_prompts.append(qwen_prompt)
        qwen_prompt_tokens.append(qwen_prompt_token)
    dataset = dataset.add_column("qwen_prompt", qwen_prompts)
    dataset = dataset.add_column("qwen_prompt_tokens", qwen_prompt_tokens)

    # select chosen and reject across ALL discovered response columns
    chosen, reject, qwen_chosen, qwen_reject, qwen_chosen_tokens, qwen_reject_tokens, chosen_reward, reject_reward,g_chosen, g_reject = [], [], [], [], [], [], [], [], [], []

    for row in tqdm(dataset):
        # gather responses and rewards across all response columns
        responses = []
        rewards = []
        for rc in response_cols:
            responses.append(row[rc])
            rewards.append(row[f"{rc}_reward"])        
        if len(rewards) == 0:
            # skip rows without any reward
            continue
        chosen_idx = int(np.argmax(rewards))
        reject_idx = int(np.argmin(rewards))

        chosen_text = responses[chosen_idx]
        reject_text = responses[reject_idx]
        _chosen_reward = rewards[chosen_idx]
        _reject_reward = rewards[reject_idx]

        chosen.append(chosen_text)
        reject.append(reject_text)
        chosen_reward.append(_chosen_reward)
        reject_reward.append(_reject_reward)
        g_chosen.append(chosen_text)
        g_reject.append(reject_text)

        qwen_chosen_token = tokenizer.apply_chat_template(
            get_message(response=chosen_text),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen+SYS_PROMPT_LEN,
        )[SYS_PROMPT_LEN:]
        qwen_chosen_tokens.append(qwen_chosen_token)
        chosen_text_decoded = tokenizer.decode(qwen_chosen_token, skip_special_tokens=False)
        qwen_chosen.append(chosen_text_decoded)
        assert len(qwen_chosen_token) == args.maxlen
        if "Qwen" in args.model:
            assert not chosen_text_decoded.lstrip().startswith("<|im_start|>assistant"), "Qwen chosen should not include assistant header"
            assert ("<|eot_id|>" in chosen_text_decoded) or ("<|im_end|>" in chosen_text_decoded), "Qwen chosen text should include end-of-turn marker"
            last_id = int(qwen_chosen_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen chosen last token should be PAD or EOS"

        qwen_reject_token = tokenizer.apply_chat_template(
            get_message(response=reject_text),
            add_generation_prompt=False,
            tokenize=True,
            padding='max_length',
            max_length=args.maxlen+SYS_PROMPT_LEN,
        )[SYS_PROMPT_LEN:]
        qwen_reject_tokens.append(qwen_reject_token)
        reject_text_decoded = tokenizer.decode(qwen_reject_token, skip_special_tokens=False)
        qwen_reject.append(reject_text_decoded)
        assert len(qwen_reject_token) == args.maxlen
        if "Qwen" in args.model:
            assert not reject_text_decoded.lstrip().startswith("<|im_start|>assistant"), "Qwen reject should not include assistant header"
            assert ("<|eot_id|>" in reject_text_decoded) or ("<|im_end|>" in reject_text_decoded), "Qwen reject text should include end-of-turn marker"
            last_id = int(qwen_reject_token[-1])
            pid = tokenizer.pad_token_id
            eid = tokenizer.eos_token_id
            assert (pid is None or last_id == pid) or (eid is None or last_id == eid), "Qwen reject last token should be PAD or EOS"

    dataset = dataset.add_column("chosen", chosen)
    dataset = dataset.add_column("chosen_reward", chosen_reward)
    dataset = dataset.add_column("qwen_chosen", qwen_chosen)
    dataset = dataset.add_column("qwen_chosen_tokens", qwen_chosen_tokens)
    dataset = dataset.add_column("reject", reject)
    dataset = dataset.add_column("reject_reward", reject_reward)
    dataset = dataset.add_column("qwen_reject", qwen_reject)
    dataset = dataset.add_column("qwen_reject_tokens", qwen_reject_tokens)
    dataset = dataset.add_column("g_chosen", g_chosen)
    dataset = dataset.add_column("g_reject", g_reject)

    # filter prompts with exactly same responses
    dataset = dataset.filter(lambda row: filter_same_responses(row))
    print('filtered same responses:', len(dataset))

    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    dataset.push_to_hub(args.input_repo + '_tokenized')


if __name__ == "__main__":
    main()