#!/usr/bin/env python3
"""
Script to download zjhhhh/Qwen3b and viswavi/wildchecklists datasets,
merge them, and create a new dataset with shuffled response columns.
"""

import os
import random
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
import numpy as np
from typing import List, Dict, Any
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_datasets(requirements_source: str = "wild", qwen_repo_id: str = "zjhhhh/Qwen3b"):
    """Download necessary datasets from HuggingFace Hub.

    qwen_repo_id controls which Qwen-like dataset to use.
    If requirements_source == 'qwen', only download Qwen dataset and skip Wildchecklists.
    If requirements_source == 'wild', download both datasets.
    """
    logger.info(f"Downloading {qwen_repo_id} dataset...")
    qwen_dataset = load_dataset(qwen_repo_id, split="train")

    wildchecklists_dataset = None
    if requirements_source == "wild":
        logger.info("Downloading viswavi/wildchecklists dataset...")
        try:
            wildchecklists_dataset = load_dataset("viswavi/wildchecklists", split="train")
        except ValueError as e:
            if "Feature type 'List' not found" in str(e):
                logger.warning("Feature type compatibility issue detected. Trying alternative loading method...")
                try:
                    # Try loading without specifying split first, then select train
                    wildchecklists_full = load_dataset("viswavi/wildchecklists")
                    wildchecklists_dataset = wildchecklists_full["train"]
                except Exception as e2:
                    logger.warning(f"Alternative method failed: {e2}")
                    # Try loading with trust_remote_code
                    try:
                        wildchecklists_dataset = load_dataset("viswavi/wildchecklists", split="train", trust_remote_code=True)
                    except Exception as e3:
                        logger.error(f"All loading methods failed. Last error: {e3}")
                        # Try to download and convert manually
                        wildchecklists_dataset = load_dataset_manually("viswavi/wildchecklists")
            else:
                raise e
    else:
        logger.info("Skipping wildchecklists download (using requirements from Qwen dataset)")

    return qwen_dataset, wildchecklists_dataset

def load_dataset_manually(dataset_name):
    """Manually load dataset when automatic loading fails."""
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        logger.info(f"Attempting manual download of {dataset_name}...")
        
        # Try to download the dataset files manually
        try:
            # Common dataset file patterns
            possible_files = [
                "train.jsonl", 
                "train.json", 
                "data/train.jsonl",
                "data/train.json",
                "dataset.jsonl",
                "dataset.json"
            ]
            
            dataset_file = None
            for filename in possible_files:
                try:
                    dataset_file = hf_hub_download(repo_id=dataset_name, filename=filename)
                    logger.info(f"Found dataset file: {filename}")
                    break
                except:
                    continue
            
            if dataset_file is None:
                raise ValueError(f"Could not find dataset files for {dataset_name}")
            
            # Load the file
            if dataset_file.endswith('.jsonl'):
                data = []
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Convert to Dataset
            from datasets import Dataset
            return Dataset.from_list(data)
            
        except Exception as e:
            logger.error(f"Manual download failed: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Could not load dataset {dataset_name}: {e}")
        raise

def verify_dataset_structure(qwen_dataset, wildchecklists_dataset, requirements_source: str):
    """Verify the structure of datasets needed for the chosen requirements source."""
    logger.info("Verifying dataset structures...")
    
    # Check Qwen dataset columns
    qwen_columns = list(qwen_dataset.features.keys())
    logger.info(f"Qwen dataset columns: {qwen_columns}")
    
    # Check for prompt and response_i columns
    response_columns = [col for col in qwen_columns if col.startswith('response_')]
    logger.info(f"Found {len(response_columns)} response columns: {response_columns}")
    
    if 'prompt' not in qwen_columns:
        raise ValueError("Qwen dataset missing 'prompt' column")
    
    if len(response_columns) != 20:
        logger.warning(f"Expected 20 response columns, found {len(response_columns)}")
    
    if requirements_source == 'qwen':
        if 'requirements' not in qwen_columns:
            raise ValueError("Qwen dataset missing 'requirements' column, but requirements_source is 'qwen'")
        logger.info("Using requirements from Qwen dataset; skipping wildchecklists validation")
    else:
        if wildchecklists_dataset is None:
            raise ValueError("requirements_source is 'wild' but wildchecklists_dataset is None")
        # Check wildchecklists dataset columns
        wildcheck_columns = list(wildchecklists_dataset.features.keys())
        logger.info(f"Wildchecklists dataset columns: {wildcheck_columns}")
        
        if 'prompt' not in wildcheck_columns:
            raise ValueError("Wildchecklists dataset missing 'prompt' column")
        
        if 'requirements' not in wildcheck_columns:
            raise ValueError("Wildchecklists dataset missing 'requirements' column")
        
        # Check if prompts match
        logger.info("Checking if prompts match between datasets...")
        qwen_prompts = set(qwen_dataset['prompt'])
        wildcheck_prompts = set(wildchecklists_dataset['prompt'])
        
        if len(qwen_dataset) != len(wildchecklists_dataset):
            logger.warning(f"Dataset lengths differ: Qwen={len(qwen_dataset)}, Wildchecklists={len(wildchecklists_dataset)}")
        
        # Check for prompt overlap
        common_prompts = qwen_prompts.intersection(wildcheck_prompts)
        logger.info(f"Common prompts: {len(common_prompts)} out of {len(qwen_prompts)} Qwen prompts and {len(wildcheck_prompts)} wildcheck prompts")
    
    return response_columns

def shuffle_and_select_responses(
    response_columns: List[str],
    seed: int = 42,
    num_selection: int = 3,
    num_base: int = 2,
    num_current: int = 2,
) -> Dict[str, str]:
    """
    Shuffle the response columns and select configurable counts for each group:
    - num_selection for selection_response_i (i=1..num_selection)
    - num_base for base_response_i (i=1..num_base)
    - num_current for current_response_i (i=1..num_current)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Ensure we have exactly 20 response columns (dataset contract)
    if len(response_columns) != 20:
        raise ValueError(f"Expected 20 response columns, got {len(response_columns)}")

    if any(x < 0 for x in (num_selection, num_base, num_current)):
        raise ValueError("num_selection, num_base, and num_current must be >= 0")
    total_requested = num_selection + num_base + num_current
    if total_requested > len(response_columns):
        raise ValueError(
            f"Requested {total_requested} columns (selection={num_selection}, base={num_base}, current={num_current}) "
            f"but only {len(response_columns)} available"
        )
    
    # Shuffle the response columns
    shuffled_columns = response_columns.copy()
    random.shuffle(shuffled_columns)
    
    # Select columns for each category
    selection_mapping = {}
    base_mapping = {}
    current_mapping = {}
    
    # First num_selection for selection_response
    for i in range(num_selection):
        selection_mapping[f'selection_response_{i+1}'] = shuffled_columns[i]

    # Next num_base for base_response
    base_start = num_selection
    for i in range(num_base):
        base_mapping[f'base_response_{i+1}'] = shuffled_columns[base_start + i]

    # Next num_current for current_response
    current_start = num_selection + num_base
    for i in range(num_current):
        current_mapping[f'current_response_{i+1}'] = shuffled_columns[current_start + i]
    
    # Combine all mappings
    all_mappings = {**selection_mapping, **base_mapping, **current_mapping}
    
    logger.info("Response column mappings:")
    for new_col, old_col in all_mappings.items():
        logger.info(f"  {new_col} <- {old_col}")
    
    return all_mappings

def create_merged_dataset(qwen_dataset, wildchecklists_dataset, response_mappings: Dict[str, str], requirements_source: str):
    """Create the final dataset, sourcing 'requirements' per requirements_source."""
    logger.info("Creating merged dataset...")
    
    # Convert datasets to pandas for easier manipulation
    qwen_df = qwen_dataset.to_pandas()
    final_data = {}
    if requirements_source == 'wild':
        wildcheck_df = wildchecklists_dataset.to_pandas()
        # Merge on prompt (assuming prompts are in the same order)
        logger.info("Merging datasets on prompt...")
        merged_df = pd.merge(qwen_df, wildcheck_df[['prompt', 'requirements']], 
                            on='prompt', how='inner')
        logger.info(f"Merged dataset size: {len(merged_df)} rows")
        # Create the final dataset with selected columns
        final_data = {
            'prompt': merged_df['prompt'].tolist(),
            'requirements': merged_df['requirements'].tolist()
        }
        # Add the selected response columns with new names from merged_df (which includes qwen columns)
        for new_col_name, old_col_name in response_mappings.items():
            final_data[new_col_name] = merged_df[old_col_name].tolist()
    else:
        # Use requirements directly from qwen_df; no merge
        logger.info("Using requirements from Qwen dataset; no merge performed")
        final_data = {
            'prompt': qwen_df['prompt'].tolist(),
            'requirements': qwen_df['requirements'].tolist()
        }
        for new_col_name, old_col_name in response_mappings.items():
            final_data[new_col_name] = qwen_df[old_col_name].tolist()
    
    # Create HuggingFace dataset
    final_dataset = Dataset.from_dict(final_data)
    
    logger.info(f"Final dataset columns: {list(final_dataset.features.keys())}")
    logger.info(f"Final dataset size: {len(final_dataset)}")
    
    return final_dataset

def upload_dataset(dataset: Dataset, dataset_name: str, hf_token: str = None):
    """Upload the dataset to HuggingFace Hub."""
    logger.info(f"Uploading dataset as {dataset_name}...")
    
    try:
        # Push to hub
        dataset.push_to_hub(
            dataset_name,
            token=hf_token,
            private=False  # Set to True if you want a private dataset
        )
        logger.info(f"Successfully uploaded dataset to {dataset_name}")
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Create Qwen2.5_3B_generation dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--dataset-name", type=str, default="Qwen2.5_3B_generation", 
                       help="Name for the uploaded dataset")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace token for uploading (if not set in environment)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Don't upload, just create and save locally")
    parser.add_argument("--requirements-source", type=str, choices=["wild", "qwen"], default="wild",
                       help="Source of 'requirements' field: 'wild' (merge with wildchecklists) or 'qwen' (use Qwen dataset)")
    parser.add_argument("--qwen-dataset", type=str, default="zjhhhh/Qwen3b",
                       help="Repo id of the Qwen-like dataset to load (e.g., 'zjhhhh/Qwen3b')")
    parser.add_argument("--num-selection", type=int, default=3,
                       help="Number of selection responses to include (default: 3)")
    parser.add_argument("--num-base", type=int, default=2,
                       help="Number of base responses to include (default: 2)")
    parser.add_argument("--num-current", type=int, default=2,
                       help="Number of current responses to include (default: 2)")
    
    args = parser.parse_args()
    
    try:
        # Download datasets
        qwen_dataset, wildchecklists_dataset = download_datasets(requirements_source=args.requirements_source, qwen_repo_id=args.qwen_dataset)
        
        # Verify structure
        response_columns = verify_dataset_structure(qwen_dataset, wildchecklists_dataset, requirements_source=args.requirements_source)
        
        # Shuffle and select response columns
        response_mappings = shuffle_and_select_responses(
            response_columns,
            seed=args.seed,
            num_selection=args.num_selection,
            num_base=args.num_base,
            num_current=args.num_current,
        )
        
        # Create merged dataset
        final_dataset = create_merged_dataset(qwen_dataset, wildchecklists_dataset, response_mappings, requirements_source=args.requirements_source)
        
        # Save locally first
        # local_path = f"{args.dataset_name}_local"
        # final_dataset.save_to_disk(local_path)
        # logger.info(f"Saved dataset locally to {local_path}")
        
        if not args.dry_run:
            # Upload to HuggingFace Hub
            hf_token = args.hf_token or os.getenv("HF_TOKEN")
            upload_dataset(final_dataset, args.dataset_name, hf_token)
        else:
            logger.info("Dry run mode - skipping upload")
            
        logger.info("Process completed successfully!")
        
        # Print sample of the final dataset
        logger.info("\nSample from final dataset:")
        sample = final_dataset.select(range(min(3, len(final_dataset))))
        for i, example in enumerate(sample):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"  Prompt: {example['prompt'][:100]}...")
            logger.info(f"  Requirements: {example['requirements'][:100]}...")
            for key in example.keys():
                if key.startswith(('selection_response', 'base_response', 'current_response')):
                    logger.info(f"  {key}: {str(example[key])[:100]}...")
                    
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
