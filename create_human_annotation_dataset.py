#!/usr/bin/env python3
"""
Script to create human annotation dataset from viswavi/wildchecklists
- Randomly sample 100 entries
- Shuffle chosen/rejected responses and rename as response_0/response_1
- Upload to HuggingFace as 'human-annotation'
"""

from datasets import load_dataset, Dataset
import random
import numpy as np
from huggingface_hub import HfApi
import argparse

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def create_human_annotation_dataset(
    source_dataset="viswavi/wildchecklists",
    num_samples=100,
    output_repo="human-annotation",
    seed=42
):
    """
    Create human annotation dataset by sampling and shuffling responses
    
    Args:
        source_dataset: Source dataset name
        num_samples: Number of samples to draw
        output_repo: Output repository name for HuggingFace
        seed: Random seed for reproducibility
    """
    
    # Set seed for reproducibility
    set_seed(seed)
    
    print(f"Loading dataset: {source_dataset}")
    # Load the source dataset
    dataset = load_dataset(source_dataset, split='train')
    print(f"Original dataset size: {len(dataset)}")
    
    # Randomly sample entries
    total_size = len(dataset)
    random_indices = random.sample(range(total_size), min(num_samples, total_size))
    sampled_dataset = dataset.select(random_indices)
    print(f"Sampled {len(sampled_dataset)} entries")
    
    # Process each entry to shuffle chosen/rejected responses
    processed_entries = []
    
    for i, entry in enumerate(sampled_dataset):
        # Extract the required fields
        prompt = entry['prompt']
        chosen = entry['chosen']
        rejected = entry['rejected']
        requirements = entry['requirements']
        
        # Randomly shuffle chosen and rejected
        responses = [chosen, rejected]
        random.shuffle(responses)
        
        # Create new entry with shuffled responses
        new_entry = {
            'prompt': prompt,
            'response_0': responses[0],
            'response_1': responses[1],
            'requirements': requirements
        }
        
        processed_entries.append(new_entry)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sampled_dataset)} entries")
    
    # Create new dataset
    new_dataset = Dataset.from_list(processed_entries)
    print(f"Created new dataset with {len(new_dataset)} entries")
    print(f"Dataset columns: {new_dataset.column_names}")
    
    # Show sample entry
    print("\nSample entry:")
    sample = new_dataset[0]
    for key, value in sample.items():
        print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    try:
        # Upload to HuggingFace Hub
        print(f"\nUploading to HuggingFace Hub: {output_repo}")
        new_dataset.push_to_hub(output_repo, private=False)
        print(f"✅ Successfully uploaded to {output_repo}")
        
    except Exception as e:
        print(f"❌ Failed to upload to HuggingFace Hub: {e}")
        print("💾 Saving locally as backup...")
        
        # Save locally as backup
        backup_path = f"./{output_repo}_backup"
        new_dataset.save_to_disk(backup_path)
        print(f"Dataset saved locally to {backup_path}")
        
        # Also save as JSON for easy inspection
        json_path = f"./{output_repo}_backup.json"
        new_dataset.to_json(json_path)
        print(f"Dataset also saved as JSON to {json_path}")
    
    return new_dataset

def main():
    parser = argparse.ArgumentParser(description="Create human annotation dataset")
    parser.add_argument("--source_dataset", default="viswavi/wildchecklists", 
                       help="Source dataset name")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to draw")
    parser.add_argument("--output_repo", default="human-annotation",
                       help="Output repository name for HuggingFace")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("Creating human annotation dataset...")
    print(f"Source: {args.source_dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output_repo}")
    print(f"Seed: {args.seed}")
    print("-" * 50)
    
    dataset = create_human_annotation_dataset(
        source_dataset=args.source_dataset,
        num_samples=args.num_samples,
        output_repo=args.output_repo,
        seed=args.seed
    )
    
    print("\n🎉 Process completed!")

if __name__ == "__main__":
    main()
