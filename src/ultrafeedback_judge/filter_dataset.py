import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Filter dataset based on prompt length when applied to preference judging templates")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/Qwen2.5_3B_generation_488_1", help="Input dataset repository")
    parser.add_argument("--n_sample", type=int, default=72000, help="Number of samples to randomly select")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum allowed prompt length in tokens")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B", help="Judge model for tokenization")
    parser.add_argument("--selection_pairs", type=int, default=4, help="Number of selection responses")
    parser.add_argument("--base_pairs", type=int, default=2, help="Number of base responses")
    parser.add_argument("--current_pairs", type=int, default=2, help="Number of current responses")
    parser.add_argument("--adversary_pairs", type=int, default=2, help="Number of adversary responses")
    parser.add_argument("--output_repo", type=str, default=None, help="Output dataset repository")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.output_repo is None:
        args.output_repo = f"{args.input_repo}_filtered"
    return args

def load_prompt_templates():
    """Load all prompt templates used in the judging process."""
    templates = {}
    template_files = {
        "preference_binary": "prompt_preference_binary.txt",
        "preference_ternary": "prompt_preference_ternary.txt", 
        "preference_score": "prompt_preference_score.txt",
        "preference_5score": "prompt_preference_5score_explanation.txt"
    }
    
    script_dir = Path(__file__).parent
    for template_name, filename in template_files.items():
        template_path = script_dir / filename
        if template_path.exists():
            with open(template_path, "r") as f:
                templates[template_name] = f.read()
        else:
            print(f"Warning: Template file {filename} not found")
    
    return templates

def expand_dataset_requirements(dataset):
    """Expand dataset by splitting requirements into individual rows."""
    expanded_data = []
    for row in dataset:
        # Parse requirements string
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

        # Create new row for each requirement
        for req in requirements:
            new_row = dict(row)
            new_row['check'] = req.split('(importance:')[0].strip()
            new_row['importance'] = int(req.split('(importance:')[1].split('/')[0].strip())
            expanded_data.append(new_row)

    return Dataset.from_list(expanded_data)

def parse_requirements_checks(requirements_str: str) -> List[str]:
    """Parse a requirements string and return a list of 'check' texts.

    The parser is tolerant to formatting variations, including:
    - Different line endings (\r\n or \n)
    - Leading spaces before numbering
    - Missing trailing "/100)" or even missing (importance: ...) segments
    - Entire string without numbering treated as a single requirement
    """
    if not isinstance(requirements_str, str):
        return []

    text = requirements_str.replace("\r\n", "\n").strip()
    if not text:
        return []

    matches = list(re.finditer(r"(?m)^\s*(\d+)\)\s*", text))
    segments: List[str] = []

    if matches:
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            segment = text[start:end].strip()
            if not segment:
                continue
            if '(importance:' in segment:
                check_text = segment.split('(importance:')[0].strip()
            else:
                check_text = segment.strip()
            if check_text:
                segments.append(check_text)
    else:
        segment = text
        if '(importance:' in segment:
            check_text = segment.split('(importance:')[0].strip()
        else:
            check_text = segment.strip()
        if check_text:
            segments.append(check_text)

    return segments

def calculate_prompt_length(tokenizer, template, row, response_a_col, response_b_col):
    """Calculate the token length of a prompt when formatted with the template."""
    try:
        formatted_prompt = template.format(
            prompt=row['prompt'],
            response_a=row[response_a_col],
            response_b=row[response_b_col],
            check=row['check']
        )
        
        # Apply chat template (similar to what's done in the judge script)
        message = [{"role": "user", "content": formatted_prompt}]
        chat_formatted = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and get length
        tokens = tokenizer.encode(chat_formatted)
        return len(tokens)
    except Exception as e:
        print(f"Error calculating length for {response_a_col} vs {response_b_col}: {e}")
        return float('inf')  # Return very large number to filter out problematic rows

def filter_by_length(dataset, tokenizer, templates, max_length, selection_pairs, base_pairs, current_pairs, adversary_pairs):
    """Efficiently filter dataset at the prompt level using batched tokenization.

    A prompt row is kept if and only if, for all of its requirements, all
    selection/base, current/base, selection/adversary, and current/adversary pairs
    (when the corresponding columns exist) format within the max_length.
    """
    # Use the 5score explanation template as it's the longest/most complex
    template = templates.get("preference_5score", templates.get("preference_binary", ""))
    if not template:
        raise ValueError("No suitable template found for length calculation")
    
    print(f"Using template: preference_5score_explanation")
    print(f"Template length: {len(template)} characters")
    
    # Define response column names
    selection_responses = [f'selection_response_{i+1}' for i in range(selection_pairs)]
    base_responses = [f'base_response_{j+1}' for j in range(base_pairs)]
    current_responses = [f'current_response_{k+1}' for k in range(current_pairs)]
    adversary_responses = [f'adversary_response_{t+1}' for t in range(adversary_pairs)]
    
    valid_indices = []  # indices into the original (non-expanded) dataset
    max_lengths = []    # track the max prompt length per kept prompt for stats

    print("Preparing formatted prompts for all pairs (batched tokenization)...")

    all_formatted_prompts: List[str] = []
    prompt_row_indices: List[int] = []
    row_prompt_count = [0] * len(dataset)

    for idx, row in enumerate(tqdm(dataset)):
        try:
            checks = parse_requirements_checks(row.get('requirements', ''))
        except Exception as e:
            print(f"Error parsing requirements for row {idx}: {e}")
            checks = []

        for check_text in checks:
            temp_row = dict(row)
            temp_row['check'] = check_text

            # selection vs base
            for selection_col in selection_responses:
                for base_col in base_responses:
                    if selection_col in temp_row and base_col in temp_row:
                        try:
                            formatted_prompt = template.format(
                                prompt=temp_row['prompt'],
                                response_a=temp_row[selection_col],
                                response_b=temp_row[base_col],
                                check=temp_row['check']
                            )
                            message = [{"role": "user", "content": formatted_prompt}]
                            chat_formatted = tokenizer.apply_chat_template(
                                message, tokenize=False, add_generation_prompt=True
                            )
                            all_formatted_prompts.append(chat_formatted)
                            prompt_row_indices.append(idx)
                            row_prompt_count[idx] += 1
                        except Exception:
                            continue

            # current vs base
            for current_col in current_responses:
                for base_col in base_responses:
                    if current_col in temp_row and base_col in temp_row:
                        try:
                            formatted_prompt = template.format(
                                prompt=temp_row['prompt'],
                                response_a=temp_row[current_col],
                                response_b=temp_row[base_col],
                                check=temp_row['check']
                            )
                            message = [{"role": "user", "content": formatted_prompt}]
                            chat_formatted = tokenizer.apply_chat_template(
                                message, tokenize=False, add_generation_prompt=True
                            )
                            all_formatted_prompts.append(chat_formatted)
                            prompt_row_indices.append(idx)
                            row_prompt_count[idx] += 1
                        except Exception:
                            continue

            # selection vs adversary
            for selection_col in selection_responses:
                for adv_col in adversary_responses:
                    if selection_col in temp_row and adv_col in temp_row:
                        try:
                            formatted_prompt = template.format(
                                prompt=temp_row['prompt'],
                                response_a=temp_row[selection_col],
                                response_b=temp_row[adv_col],
                                check=temp_row['check']
                            )
                            message = [{"role": "user", "content": formatted_prompt}]
                            chat_formatted = tokenizer.apply_chat_template(
                                message, tokenize=False, add_generation_prompt=True
                            )
                            all_formatted_prompts.append(chat_formatted)
                            prompt_row_indices.append(idx)
                            row_prompt_count[idx] += 1
                        except Exception:
                            continue

            # current vs adversary
            for current_col in current_responses:
                for adv_col in adversary_responses:
                    if current_col in temp_row and adv_col in temp_row:
                        try:
                            formatted_prompt = template.format(
                                prompt=temp_row['prompt'],
                                response_a=temp_row[current_col],
                                response_b=temp_row[adv_col],
                                check=temp_row['check']
                            )
                            message = [{"role": "user", "content": formatted_prompt}]
                            chat_formatted = tokenizer.apply_chat_template(
                                message, tokenize=False, add_generation_prompt=True
                            )
                            all_formatted_prompts.append(chat_formatted)
                            prompt_row_indices.append(idx)
                            row_prompt_count[idx] += 1
                        except Exception:
                            continue

    print(f"Total formatted prompts prepared: {len(all_formatted_prompts)}")

    # Batched tokenization to get lengths
    row_max_length = [0] * len(dataset)
    row_valid = [True] * len(dataset)

    if all_formatted_prompts:
        chunk_size = 2048
        total = len(all_formatted_prompts)
        num_chunks = (total + chunk_size - 1) // chunk_size
        for start in tqdm(range(0, total, chunk_size), total=num_chunks, desc="Tokenizing prompts", dynamic_ncols=True):
            end = min(start + chunk_size, total)
            batch_texts = all_formatted_prompts[start:end]
            batch_indices = prompt_row_indices[start:end]

            enc = tokenizer(
                batch_texts,
                add_special_tokens=True,
                return_length=True,
                padding=False,
                truncation=False,
            )

            lengths = enc.get('length')
            if lengths is None:
                input_ids = enc.get('input_ids', [])
                lengths = [len(x) for x in input_ids]

            for r_idx, l in zip(batch_indices, lengths):
                if l is None:
                    continue
                if l > row_max_length[r_idx]:
                    row_max_length[r_idx] = l
                if l > max_length:
                    row_valid[r_idx] = False

    # Rows with no evaluated prompts are considered invalid
    for i in range(len(dataset)):
        if row_prompt_count[i] == 0:
            row_valid[i] = False

    for i in range(len(dataset)):
        if row_valid[i]:
            valid_indices.append(i)
            max_lengths.append(row_max_length[i])
    
    print(f"Filtered {len(valid_indices)} valid rows out of {len(dataset)} total rows")
    
    # Filter dataset (prompt-level)
    filtered_dataset = dataset.select(valid_indices)

    return filtered_dataset, max_lengths

def main():
    args = parse_arguments()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Loading dataset from {args.input_repo}...")
    # Load dataset
    dataset = load_dataset(args.input_repo, split='train')
    print(f"Original dataset size: {len(dataset)}")
    
    # Randomly sample n_sample examples
    if len(dataset) > args.n_sample:
        indices = random.sample(range(len(dataset)), args.n_sample)
        dataset = dataset.select(indices)
        print(f"Randomly sampled {args.n_sample} examples")
    
    # Load prompt templates
    print("Loading prompt templates...")
    templates = load_prompt_templates()
    
    # Initialize tokenizer
    print(f"Loading tokenizer for {args.judge_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    
    # Filter by length
    print(f"Filtering by max length {args.max_length} tokens...")
    filtered_dataset, max_lengths = filter_by_length(
        dataset, tokenizer, templates, args.max_length,
        args.selection_pairs, args.base_pairs, args.current_pairs, args.adversary_pairs
    )
    
    # Output statistics
    print("\n" + "="*50)
    print("FILTERING RESULTS")
    print("="*50)
    print(f"Original sampled dataset size: {args.n_sample}")
    print(f"After length filtering (prompt-level): {len(filtered_dataset)}")
    print(f"Filtering ratio: {len(filtered_dataset)/len(dataset)*100:.2f}%")
    
    if max_lengths:
        print(f"\nPrompt Length Statistics (tokens):")
        print(f"Max length in filtered dataset: {max(max_lengths)}")
        print(f"Min length in filtered dataset: {min(max_lengths)}")
        print(f"Average length: {sum(max_lengths)/len(max_lengths):.1f}")
        print(f"Length threshold used: {args.max_length}")
        
        # Show distribution
        length_ranges = [
            (0, 2000), (2000, 4000), (4000, 6000), 
            (6000, 8000), (8000, 10000), (10000, float('inf'))
        ]
        print(f"\nLength Distribution:")
        for min_len, max_len in length_ranges:
            count = sum(1 for l in max_lengths if min_len <= l < max_len)
            if count > 0:
                range_str = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
                print(f"  {range_str:>10} tokens: {count:>6} samples ({count/len(max_lengths)*100:.1f}%)")
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset to {args.output_repo}...")
    filtered_dataset.push_to_hub(args.output_repo)
    print("Done!")

if __name__ == "__main__":
    main()
