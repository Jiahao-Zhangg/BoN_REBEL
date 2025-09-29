import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Filter dataset based on prompt length when applied to preference judging templates")
    parser.add_argument("--input_repo", type=str, default="zjhhhh/Qwen2.5_3B_generation", help="Input dataset repository")
    parser.add_argument("--n_sample", type=int, default=12000, help="Number of samples to randomly select")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum allowed prompt length in tokens")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B", help="Judge model for tokenization")
    parser.add_argument("--selection_pairs", type=int, default=3, help="Number of selection responses")
    parser.add_argument("--base_pairs", type=int, default=2, help="Number of base responses")
    parser.add_argument("--current_pairs", type=int, default=2, help="Number of current responses")
    parser.add_argument("--output_repo", type=str, default="MisDrifter/filtered_dataset", help="Output dataset repository")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

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

def filter_by_length(dataset, tokenizer, templates, max_length, selection_pairs, base_pairs, current_pairs):
    """Filter dataset based on prompt length constraints."""
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
    
    valid_indices = []
    max_lengths = []
    
    print("Calculating prompt lengths for all pairs...")
    
    for idx, row in enumerate(tqdm(dataset)):
        row_max_length = 0
        valid_row = True
        
        # Check selection vs base pairs
        for selection_col in selection_responses:
            for base_col in base_responses:
                if selection_col in row and base_col in row:
                    length = calculate_prompt_length(tokenizer, template, row, selection_col, base_col)
                    row_max_length = max(row_max_length, length)
                    if length > max_length:
                        valid_row = False
                        break
            if not valid_row:
                break
        
        # Check current vs base pairs if row is still valid
        if valid_row:
            for current_col in current_responses:
                for base_col in base_responses:
                    if current_col in row and base_col in row:
                        length = calculate_prompt_length(tokenizer, template, row, current_col, base_col)
                        row_max_length = max(row_max_length, length)
                        if length > max_length:
                            valid_row = False
                            break
                if not valid_row:
                    break
        
        if valid_row:
            valid_indices.append(idx)
            max_lengths.append(row_max_length)
    
    print(f"Filtered {len(valid_indices)} valid rows out of {len(dataset)} total rows")
    
    # Filter dataset
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
    
    # Expand dataset by splitting requirements
    print("Expanding dataset by splitting requirements...")
    dataset = expand_dataset_requirements(dataset)
    print(f"Expanded dataset size: {len(dataset)}")
    
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
        args.selection_pairs, args.base_pairs, args.current_pairs
    )
    
    # Output statistics
    print("\n" + "="*50)
    print("FILTERING RESULTS")
    print("="*50)
    print(f"Original sampled dataset size: {args.n_sample}")
    print(f"After expanding requirements: {len(dataset)}")
    print(f"After length filtering: {len(filtered_dataset)}")
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
