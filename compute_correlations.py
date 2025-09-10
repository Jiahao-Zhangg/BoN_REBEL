#!/usr/bin/env python3
"""
Script to compute Pearson, Spearman, and Kendall's Tau correlation coefficients 
between human annotations and judge preferences from a single Hugging Face dataset.
"""

import os
import ast
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from datasets import load_dataset
import argparse
import pandas as pd

MAP = {'A': 4, 'B': 0, 'Tie': 2, -1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
MAP_TERNARY = {0:0, 1:0, 2:2, 3:4, 4:4}


def read_human_annotations(file_path, ternary=False):
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
                if ternary:
                    array = list(map(lambda x: MAP_TERNARY[x], array))
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
            # Use preference judgments - try different column names
            if 'response_0_1_judged_preference_mean' in row:
                pref = row['response_0_1_judged_preference_mean']
                judge_preferences.append(pref)
                print(f"Row {i}: {pref}")
            elif 'response_0_1_judged_preference' in row:
                pref = row['response_0_1_judged_preference']
                judge_preferences.append(pref)
                print(f"Row {i}: {pref}")
            elif 'response_0_1_judged_preference_majority' in row:
                pref = row['response_0_1_judged_preference_majority']
                # pref = list(map(lambda x: MAP[x], pref))
                judge_preferences.append(pref)
                print(f"Row {i}: {pref}")
            else:
                print(f"Warning: Row {i} missing 'response_0_1_judged_preference' column")
                print(f"Available columns: {list(row.keys())}")
        
        return judge_preferences
        
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()
        return None

def map_to_ternary(value):
    """Map a value to {-1, 0, 1} based on comparison to 2."""
    if value == 2:
        return 2
    elif value < 2:
        return 0
    else:  # value > 2
        return 4

def apply_ternary_mapping(arrays):
    """Apply ternary mapping to all values in arrays."""
    mapped_arrays = []
    for array in arrays:
        if isinstance(array, list):
            mapped_array = [map_to_ternary(x) for x in array]
        else:
            mapped_array = map_to_ternary(array)
        mapped_arrays.append(mapped_array)
    return mapped_arrays

def flatten_arrays(arrays):
    """Flatten a list of arrays into a single list."""
    flattened = []
    for array in arrays:
        if isinstance(array, list):
            flattened.extend(array)
        else:
            flattened.append(array)
    return flattened

def compute_correlations(x_data, y_data, use_ternary_mapping=False):
    """
    Compute Pearson, Spearman, and Kendall's Tau correlations between two datasets.
    
    Args:
        x_data: List of arrays (human annotations)
        y_data: List of arrays (judge preferences)
        use_ternary_mapping: If True, map all values to {-1, 0, 1}
    
    Returns:
        Dictionary with correlation results
    """
    print(f"\nComputing correlations using flattened method")
    print(f"X data: {len(x_data)} arrays")
    print(f"Y data: {len(y_data)} arrays")
    
    if len(x_data) != len(y_data):
        print(f"Warning: Mismatched number of arrays. X: {len(x_data)}, Y: {len(y_data)}")
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        print(f"Using first {min_len} arrays from each dataset")
    
    # Apply ternary mapping if requested
    if use_ternary_mapping:
        print("Applying ternary mapping: 2→0, <2→-1, >2→1")
        x_data_orig = [arr.copy() if isinstance(arr, list) else arr for arr in x_data]
        y_data_orig = [arr.copy() if isinstance(arr, list) else arr for arr in y_data]
        
        x_data = apply_ternary_mapping(x_data)
        y_data = apply_ternary_mapping(y_data)
        
        print("Original vs Mapped examples:")
        for i in range(min(3, len(x_data))):
            print(f"  X array {i}: {x_data_orig[i]} → {x_data[i]}")
            print(f"  Y array {i}: {y_data_orig[i]} → {y_data[i]}")
    
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
    
    # Compute all three correlations
    pearson_r, pearson_p = pearsonr(x_flat, y_flat)
    spearman_rho, spearman_p = spearmanr(x_flat, y_flat)
    kendall_tau, kendall_p = kendalltau(x_flat, y_flat)
    
    results = {
        'pearson': {'r': pearson_r, 'p': pearson_p},
        'spearman': {'rho': spearman_rho, 'p': spearman_p},
        'kendall': {'tau': kendall_tau, 'p': kendall_p}
    }
    
    print(f"\nFlattened data correlations:")
    print(f"X values (first 10): {x_flat[:10]}")
    print(f"Y values (first 10): {y_flat[:10]}")
    
    return results

def interpret_correlation(correlation, correlation_type):
    """
    Provide interpretation of correlation strength.
    
    Args:
        correlation: Correlation coefficient value
        correlation_type: Type of correlation ('pearson', 'spearman', 'kendall')
    """
    abs_corr = abs(correlation)
    
    if abs_corr >= 0.9:
        strength = "very strong"
    elif abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.5:
        strength = "moderate"
    elif abs_corr >= 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    direction = "positive" if correlation > 0 else "negative"
    
    # Type-specific interpretations
    if correlation_type == 'pearson':
        description = f"Linear relationship is {strength} and {direction}"
    elif correlation_type == 'spearman':
        description = f"Monotonic relationship is {strength} and {direction}"
    elif correlation_type == 'kendall':
        description = f"Rank concordance is {strength} and {direction}"
    
    return strength, direction, description

def display_results(results):
    """Display correlation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    for corr_type, values in results.items():
        if corr_type == 'pearson':
            coeff_name = 'r'
            full_name = "Pearson correlation coefficient"
            coeff_value = values['r']
        elif corr_type == 'spearman':
            coeff_name = 'ρ (rho)'
            full_name = "Spearman's rank correlation coefficient"
            coeff_value = values['rho']
        elif corr_type == 'kendall':
            coeff_name = 'τ (tau)'
            full_name = "Kendall's Tau correlation coefficient"
            coeff_value = values['tau']
        
        p_value = values['p']
        strength, direction, description = interpret_correlation(coeff_value, corr_type)
        
        # Significance level
        if p_value < 0.001:
            significance = "*** (p < 0.001)"
        elif p_value < 0.01:
            significance = "** (p < 0.01)"
        elif p_value < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "ns (not significant)"
        
        print(f"\n{full_name} ({coeff_name}):")
        print(f"  Coefficient: {coeff_value:.4f}")
        print(f"  P-value: {p_value:.2e}")
        print(f"  Significance: {significance}")
        print(f"  Interpretation: {description}")
        
        # Additional explanations
        if corr_type == 'pearson':
            print(f"  → Measures linear relationship between variables")
            print(f"  → Sensitive to outliers and assumes normal distribution")
        elif corr_type == 'spearman':
            print(f"  → Non-parametric, based on rank ordering")
            print(f"  → More robust to outliers than Pearson")
        elif corr_type == 'kendall':
            print(f"  → Based on concordant/discordant pairs")
            print(f"  → More interpretable in terms of probability of agreement")
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print("• Pearson: Measures linear relationships (assumes normality)")
    print("• Spearman: Measures monotonic relationships (rank-based, robust)")
    print("• Kendall: Measures rank concordance (probability-based interpretation)")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Compute Pearson, Spearman, and Kendall correlations")
    parser.add_argument("--human-file", default="src/ultrafeedback_judge/human_annotation.txt",
                       help="Path to human annotation file")
    parser.add_argument("--dataset", required=True,
                       help="Hugging Face dataset name")
    parser.add_argument("--ternary", action='store_true', help="Use ternary preference")
    parser.add_argument("--map-ternary", action='store_true', 
                       help="Map all values to {-1,0,1}: 2→0, <2→-1, >2→1")
    args = parser.parse_args()
    
    print("=== Multi-Correlation Analysis ===")
    print("Computing Pearson, Spearman, and Kendall's Tau correlations")
    
    # Read human annotations
    print(f"\n1. Reading human annotations from: {args.human_file}")
    human_data = read_human_annotations(args.human_file, args.ternary)
    print(f"Loaded {len(human_data)} annotation arrays:")
    for i, arr in enumerate(human_data):
        print(f"  Array {i}: {arr}")
    
    # Load judge preferences
    print(f"\n2. Loading judge preferences from: {args.dataset}")
    judge_data = load_judge_preferences(args.dataset)
    
    if judge_data is None:
        print("Failed to load judge data. Exiting.")
        return
    
    print(f"Loaded {len(judge_data)} judge preference arrays:")
    for i, arr in enumerate(judge_data):
        print(f"  Array {i}: {arr}")
    
    # Verify data alignment
    if len(human_data) != len(judge_data):
        print(f"\nWarning: Different number of arrays. Human: {len(human_data)}, Judge: {len(judge_data)}")
        min_len = min(len(human_data), len(judge_data))
        print(f"Truncating both datasets to {min_len} arrays")
        human_data = human_data[:min_len]
        judge_data = judge_data[:min_len]
    
    # Compute correlations
    print(f"\n3. Computing correlations")
    
    results = compute_correlations(human_data, judge_data, args.map_ternary)
    if results:
        display_results(results)

if __name__ == "__main__":
    main()
