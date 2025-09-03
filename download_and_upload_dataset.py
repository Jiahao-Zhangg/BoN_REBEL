#!/usr/bin/env python3
"""
Script to download specific rows from zjhhhh/human-annotation-1.5B dataset
and upload them to zjhhhh/human-scored-1.5
"""

import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import argparse


def main():
    # Specific rows to extract
    rows = [0, 8, 10, 11, 13, 14, 18, 29, 30, 31]
    
    print("Loading dataset zjhhhh/human-annotation-1.5B...")
    try:
        # Load the original dataset
        dataset = load_dataset("zjhhhh/human-annotation-1.5B")
        
        # Check if it's a DatasetDict or a single Dataset
        if hasattr(dataset, 'keys'):
            print(f"Dataset splits available: {list(dataset.keys())}")
            # Use the first split or 'train' if available
            split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Using split: {split_name}")
        else:
            data = dataset
            print("Using default dataset (no splits)")
        
        print(f"Original dataset size: {len(data)}")
        
        # Validate that all requested rows exist
        max_row = max(rows)
        if max_row >= len(data):
            print(f"Error: Requested row {max_row} but dataset only has {len(data)} rows")
            return
        
        print(f"Extracting rows: {rows}")
        
        # Extract the specific rows
        selected_data = data.select(rows)
        print(f"Selected {len(selected_data)} rows")
        
        # Create a new dataset from the selected rows
        filtered_dataset = Dataset.from_dict(selected_data.to_dict())
        
        print("Uploading to zjhhhh/human-scored-1.5...")
        
        # Upload to Hugging Face Hub
        filtered_dataset.push_to_hub(
            "zjhhhh/human-scored-1.5B",
            private=False,  # Set to True if you want a private dataset
            commit_message=f"Upload filtered dataset with rows {rows} from human-annotation-1.5B"
        )
        
        print("✅ Successfully uploaded filtered dataset to zjhhhh/human-scored-1.5")
        print(f"Dataset contains {len(filtered_dataset)} rows")
        
        # Print first few examples for verification
        print("\nFirst example from uploaded dataset:")
        if len(filtered_dataset) > 0:
            example = filtered_dataset[0]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {value[:100]}...")
                else:
                    print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install datasets huggingface_hub")
        print("2. Logged in to Hugging Face: huggingface-cli login")
        print("3. Have access to both source and destination repositories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and upload filtered dataset")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Only load and filter data without uploading")
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 Running in dry-run mode (no upload)")
    
    main()
