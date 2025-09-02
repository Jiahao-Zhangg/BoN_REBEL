#!/usr/bin/env python3
"""
Script to compute Pearson correlation coefficient between human annotations
and judge preferences from Hugging Face dataset.
"""

import os
import ast
import numpy as np
from scipy.stats import pearsonr
from datasets import load_dataset
import argparse


def read_human_annotations(file_path):
    """Read human annotation arrays from the text file."""
    annotations = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the first two lines (rows info and empty line)
    for line in lines[2:]:
        line = line.strip()
        if line and line.startswith('[') and line.endswith(']'):
            try:
                # Parse the array from string
                array = ast.literal_eval(line)
                annotations.append(array)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse line '{line}': {e}")
    
    return annotations


def load_judge_preferences(dataset_name):
    """Load judge preferences from Hugging Face dataset."""
    print(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name)
        
        # Handle different dataset formats
        if hasattr(dataset, 'keys'):
            print(f"Dataset splits available: {list(dataset.keys())}")
            split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Using split: {split_name}")
        else:
            data = dataset
            print("Using default dataset (no splits)")
        
        print(f"Dataset size: {len(data)}")
        
        # Extract the judge preferences
        judge_preferences = []
        for i, row in enumerate(data):
            if 'response_0_1_judged_preference' in row:
                pref = row['response_0_1_judged_preference']
                judge_preferences.append(pref)
                print(f"Row {i}: {pref}")
            else:
                print(f"Warning: Row {i} missing 'response_0_1_judged_preference' column")
                print(f"Available columns: {list(row.keys())}")
        
        return judge_preferences
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def flatten_arrays(arrays):
    """Flatten a list of arrays into a single list."""
    flattened = []
    for array in arrays:
        if isinstance(array, list):
            flattened.extend(array)
        else:
            flattened.append(array)
    return flattened


def compute_pearson_correlation(x_data, y_data, method='flattened'):
    """
    Compute Pearson correlation coefficient between two datasets.
    
    Args:
        x_data: List of arrays (human annotations)
        y_data: List of arrays (judge preferences)
        method: 'flattened' (concatenate all arrays) or 'average' (compute correlation for each array pair, then average)
    """
    print(f"\nComputing correlation using method: {method}")
    print(f"X data: {len(x_data)} arrays")
    print(f"Y data: {len(y_data)} arrays")
    
    if len(x_data) != len(y_data):
        print(f"Warning: Mismatched number of arrays. X: {len(x_data)}, Y: {len(y_data)}")
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        print(f"Using first {min_len} arrays from each dataset")
    
    if method == 'flattened':
        # Flatten all arrays into single lists
        x_flat = flatten_arrays(x_data)
        y_flat = flatten_arrays(y_data)
        
        print(f"Flattened X: {len(x_flat)} values")
        print(f"Flattened Y: {len(y_flat)} values")
        
        if len(x_flat) != len(y_flat):
            print(f"Warning: Different number of values after flattening")
            min_len = min(len(x_flat), len(y_flat))
            x_flat = x_flat[:min_len]
            y_flat = y_flat[:min_len]
            print(f"Using first {min_len} values from each")
        
        correlation, p_value = pearsonr(x_flat, y_flat)
        
        print(f"\nFlattened data correlation:")
        print(f"X values (first 10): {x_flat[:10]}")
        print(f"Y values (first 10): {y_flat[:10]}")
        
    elif method == 'average':
        # Compute correlation for each array pair, then average the correlations
        correlations = []
        p_values = []
        
        print(f"\nComputing correlation for each array pair:")
        
        for i, (x_arr, y_arr) in enumerate(zip(x_data, y_data)):
            # Ensure arrays have same length
            min_len = min(len(x_arr), len(y_arr))
            if min_len < 2:
                print(f"  Array pair {i}: Skipping (too few data points: {min_len})")
                continue
                
            x_trimmed = x_arr[:min_len]
            y_trimmed = y_arr[:min_len]
            
            try:
                corr, p_val = pearsonr(x_trimmed, y_trimmed)
                correlations.append(corr)
                p_values.append(p_val)
                print(f"  Array pair {i}: r={corr:.4f}, p={p_val:.4f} | X={x_trimmed} | Y={y_trimmed}")
            except Exception as e:
                print(f"  Array pair {i}: Error computing correlation: {e}")
                continue
        
        if not correlations:
            print("No valid correlations computed!")
            return np.nan, np.nan
        
        # Average the correlations
        correlation = np.mean(correlations)
        p_value = np.mean(p_values)  # Note: This is not statistically rigorous for p-values
        
        print(f"\nIndividual correlations: {[f'{r:.4f}' for r in correlations]}")
        print(f"Average correlation: {correlation:.4f}")
        print(f"Average p-value: {p_value:.4f} (Note: averaging p-values is not statistically rigorous)")
        print(f"Valid array pairs: {len(correlations)}/{len(x_data)}")
    
    return correlation, p_value


def main():
    parser = argparse.ArgumentParser(description="Compute Pearson correlation between annotations")
    parser.add_argument("--human-file", default="src/ultrafeedback_judge/human_annotation.txt",
                       help="Path to human annotation file")
    parser.add_argument("--dataset", default="zjhhhh/human-scored-1.5B_judge_preference_5score",
                       help="Hugging Face dataset name")
    parser.add_argument("--method", choices=['flattened', 'average', 'both'], default='both',
                       help="Correlation method")
    
    args = parser.parse_args()
    
    print("=== Pearson Correlation Analysis ===")
    
    # Read human annotations (X)
    print(f"\n1. Reading human annotations from: {args.human_file}")
    x_data = read_human_annotations(args.human_file)
    print(f"Loaded {len(x_data)} annotation arrays:")
    for i, arr in enumerate(x_data):
        print(f"  Array {i}: {arr}")
    
    # Load judge preferences (Y)
    print(f"\n2. Loading judge preferences from: {args.dataset}")
    y_data = load_judge_preferences(args.dataset)
    
    if y_data is None:
        print("Failed to load judge preferences. Exiting.")
        return
    
    print(f"Loaded {len(y_data)} judge preference arrays:")
    for i, arr in enumerate(y_data):
        print(f"  Array {i}: {arr}")
    
    # Compute correlations
    print(f"\n3. Computing Pearson correlation coefficient(s)")
    
    if args.method in ['flattened', 'both']:
        corr, p_val = compute_pearson_correlation(x_data, y_data, 'flattened')
        print(f"\n📊 FLATTENED METHOD RESULTS:")
        print(f"Pearson correlation coefficient: {corr:.4f}")
        print(f"P-value: {p_val:.6f}")
        print(f"Significance: {'Significant' if p_val < 0.05 else 'Not significant'} (α = 0.05)")
    
    if args.method in ['average', 'both']:
        corr, p_val = compute_pearson_correlation(x_data, y_data, 'average')
        print(f"\n📊 AVERAGE METHOD RESULTS:")
        print(f"Pearson correlation coefficient: {corr:.4f}")
        print(f"P-value: {p_val:.6f}")
        print(f"Significance: {'Significant' if p_val < 0.05 else 'Not significant'} (α = 0.05)")


if __name__ == "__main__":
    main()
