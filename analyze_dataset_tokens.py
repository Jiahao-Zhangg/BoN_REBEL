#!/usr/bin/env python3
"""
Analyze token lengths in your dataset to determine appropriate max_tokens setting.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset_tokens(dataset_name="viswavi/wildchecklists", model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """Analyze token lengths in the dataset"""
    
    # Load dataset and tokenizer
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split='train')
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Sample a subset if dataset is very large
    sample_size = min(1000, len(dataset))
    if len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        print(f"Analyzing {sample_size} samples from {len(dataset)} total")
    
    # Analyze prompt lengths
    prompt_lengths = []
    for item in dataset:
        # Tokenize the prompt
        tokens = tokenizer.encode(item['prompt'])
        prompt_lengths.append(len(tokens))
    
    # Calculate statistics
    prompt_lengths = np.array(prompt_lengths)
    
    print("\n=== PROMPT TOKEN ANALYSIS ===")
    print(f"Mean prompt length: {np.mean(prompt_lengths):.1f} tokens")
    print(f"Median prompt length: {np.median(prompt_lengths):.1f} tokens")
    print(f"Max prompt length: {np.max(prompt_lengths)} tokens")
    print(f"95th percentile: {np.percentile(prompt_lengths, 95):.1f} tokens")
    print(f"99th percentile: {np.percentile(prompt_lengths, 99):.1f} tokens")
    
    # Estimate response lengths (rough heuristic)
    print("\n=== RESPONSE LENGTH ESTIMATES ===")
    print("Typical response/prompt ratios:")
    print("- Short answers: 0.5-1x prompt length")
    print("- Medium answers: 1-3x prompt length") 
    print("- Long answers: 3-10x prompt length")
    
    max_prompt = np.max(prompt_lengths)
    
    print(f"\nWith max prompt of {max_prompt} tokens:")
    print(f"- For short responses: need ~{max_prompt + max_prompt * 1:.0f} tokens")
    print(f"- For medium responses: need ~{max_prompt + max_prompt * 3:.0f} tokens")
    print(f"- For long responses: need ~{max_prompt + max_prompt * 10:.0f} tokens")
    
    print(f"\n=== RECOMMENDATION ===")
    safe_limit = max_prompt + max_prompt * 5  # 5x response/prompt ratio
    if safe_limit <= 2048:
        print(f"✅ Your current max_tokens=2048 should be sufficient!")
        print(f"   (Estimated need: ~{safe_limit:.0f} tokens)")
    else:
        print(f"⚠️  Consider increasing max_tokens to {safe_limit:.0f}")
        print(f"   (Current: 2048 may be too small)")
    
    # Show distribution
    plt.figure(figsize=(10, 6))
    plt.hist(prompt_lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(prompt_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(prompt_lengths):.1f}')
    plt.axvline(np.percentile(prompt_lengths, 95), color='orange', linestyle='--', label=f'95th percentile: {np.percentile(prompt_lengths, 95):.1f}')
    plt.xlabel('Prompt Length (tokens)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prompt Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prompt_length_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved plot to: prompt_length_distribution.png")

if __name__ == "__main__":
    analyze_dataset_tokens()








