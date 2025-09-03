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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
            if 'response_0_1_judged_preference_majority' in row:
                pref = row['response_0_1_judged_preference_majority']
                judge_preferences.append(pref)
                print(f"Row {i}: {pref}")
            elif 'response_0_1_judged_preference' in row:
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


def compute_correlation_matrix(datasets_data, labels, method='flattened'):
    """
    Compute correlation matrix between multiple datasets.
    
    Args:
        datasets_data: List of datasets [X, Y, Z, W, ...]
        labels: List of labels ['X', 'Y', 'Z', 'W', ...]
        method: Correlation method
    
    Returns:
        correlation_matrix: DataFrame with correlation coefficients
        p_value_matrix: DataFrame with p-values
    """
    n_datasets = len(datasets_data)
    corr_matrix = np.zeros((n_datasets, n_datasets))
    p_matrix = np.zeros((n_datasets, n_datasets))
    
    print(f"\nComputing {n_datasets}x{n_datasets} correlation matrix using method: {method}")
    
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i == j:
                # Diagonal elements (self-correlation)
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                # Compute correlation between datasets i and j
                corr, p_val = compute_pearson_correlation(datasets_data[i], datasets_data[j], method)
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
                print(f"  {labels[i]} vs {labels[j]}: r={corr:.4f}, p={p_val:.2e}")
    
    # Create DataFrames for better visualization
    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    p_df = pd.DataFrame(p_matrix, index=labels, columns=labels)
    
    return corr_df, p_df


def plot_correlation_heatmap(corr_matrix, p_matrix, method, save_path=None):
    """
    Plot correlation matrix as heatmap with significance annotations.
    
    Args:
        corr_matrix: DataFrame with correlation coefficients
        p_matrix: DataFrame with p-values
        method: Method used for correlation computation
        save_path: Optional path to save the plot
    """
    # Set up the matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot correlation heatmap
    mask_corr = np.zeros_like(corr_matrix, dtype=bool)
    mask_corr[np.triu_indices_from(mask_corr, k=1)] = True  # Mask upper triangle
    
    sns.heatmap(corr_matrix, mask=mask_corr, annot=True, fmt='.3f', 
                cmap='RdBu_r', center=0, square=True, ax=ax1,
                cbar_kws={"shrink": .8}, vmin=-1, vmax=1)
    ax1.set_title(f'Correlation Matrix ({method.title()} Method)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Datasets', fontweight='bold')
    ax1.set_ylabel('Datasets', fontweight='bold')
    
    # Create significance annotation matrix
    sig_matrix = p_matrix.copy()
    for i in range(len(sig_matrix)):
        for j in range(len(sig_matrix.columns)):
            p_val = p_matrix.iloc[i, j]
            if i == j:
                sig_matrix.iloc[i, j] = ''
            elif p_val < 0.001:
                sig_matrix.iloc[i, j] = '***'
            elif p_val < 0.01:
                sig_matrix.iloc[i, j] = '**'
            elif p_val < 0.05:
                sig_matrix.iloc[i, j] = '*'
            else:
                sig_matrix.iloc[i, j] = 'ns'
    
    # Plot p-value heatmap with significance annotations
    mask_p = np.zeros_like(p_matrix, dtype=bool)
    mask_p[np.triu_indices_from(mask_p, k=1)] = True  # Mask upper triangle
    
    sns.heatmap(p_matrix, mask=mask_p, annot=sig_matrix, fmt='s',
                cmap='viridis_r', square=True, ax=ax2,
                cbar_kws={"shrink": .8, "label": "P-value"})
    ax2.set_title(f'P-values with Significance ({method.title()} Method)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Datasets', fontweight='bold')
    ax2.set_ylabel('Datasets', fontweight='bold')
    
    # Add legend for significance levels
    legend_text = "Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compute Pearson correlation matrix between annotations")
    parser.add_argument("--human-file", default="src/ultrafeedback_judge/human_annotation.txt",
                       help="Path to human annotation file")
    parser.add_argument("--datasets", nargs='+', 
                       default=["zjhhhh/human-scored-1.5B_judge_preference_5score_qwen2.5_72b_ver2_run2"],
                       help="List of Hugging Face dataset names (Y, Z, W, ...)")
    parser.add_argument("--dataset-labels", nargs='+',
                       default=["Y"],
                       help="Labels for the datasets (Y, Z, W, ...)")
    parser.add_argument("--method", choices=['flattened', 'average', 'both'], default='both',
                       help="Correlation method")
    parser.add_argument("--plot", action='store_true',
                       help="Generate heatmap visualization")
    parser.add_argument("--save-plot", type=str, default="correlation_heatmap.png",
                       help="Path to save the heatmap plot (e.g., 'correlation_heatmap.png')")
    
    args = parser.parse_args()
    
    # Ensure we have labels for all datasets
    if len(args.dataset_labels) != len(args.datasets):
        print("Warning: Number of dataset labels doesn't match number of datasets. Using default labels.")
        args.dataset_labels = [f"Dataset_{i+1}" for i in range(len(args.datasets))]
    
    print("=== Pearson Correlation Matrix Analysis ===")
    
    # Read human annotations (X)
    print(f"\n1. Reading human annotations from: {args.human_file}")
    x_data = read_human_annotations(args.human_file)
    print(f"Loaded {len(x_data)} annotation arrays:")
    for i, arr in enumerate(x_data):
        print(f"  Array {i}: {arr}")
    
    # Load all judge preference datasets (Y, Z, W, ...)
    all_datasets = [x_data]  # Start with human data (X)
    all_labels = ['human'] + args.dataset_labels
    
    print(f"\n2. Loading {len(args.datasets)} judge preference datasets:")
    for i, dataset_name in enumerate(args.datasets):
        print(f"  Loading {args.dataset_labels[i]} from: {dataset_name}")
        judge_data = load_judge_preferences(dataset_name)
        
        if judge_data is None:
            print(f"Failed to load {args.dataset_labels[i]}. Skipping.")
            continue
        
        print(f"  Loaded {len(judge_data)} arrays for {args.dataset_labels[i]}:")
        for j, arr in enumerate(judge_data):
            print(f"    Array {j}: {arr}")
        
        all_datasets.append(judge_data)
    
    # Verify all datasets have the same number of arrays
    if len(set(len(dataset) for dataset in all_datasets)) > 1:
        print("\nWarning: Datasets have different numbers of arrays:")
        for i, dataset in enumerate(all_datasets):
            print(f"  {all_labels[i]}: {len(dataset)} arrays")
        
        # Truncate to minimum length
        min_len = min(len(dataset) for dataset in all_datasets)
        print(f"Truncating all datasets to {min_len} arrays")
        all_datasets = [dataset[:min_len] for dataset in all_datasets]
        all_labels = all_labels[:len(all_datasets)]
    
    # Compute correlation matrices
    print(f"\n3. Computing Pearson correlation matrix")
    
    if args.method in ['flattened', 'both']:
        corr_matrix, p_matrix = compute_correlation_matrix(all_datasets, all_labels, 'flattened')
        print(f"\n📊 FLATTENED METHOD CORRELATION MATRIX:")
        print("Correlation Coefficients:")
        print(corr_matrix.round(4))
        print("\nP-values:")
        print(p_matrix)
        
        print(f"\nSignificance Matrix (α = 0.05):")
        significance = (p_matrix < 0.05).astype(str)
        significance = significance.replace({'True': 'Significant', 'False': 'Not Significant'})
        print(significance)
        
        # Generate heatmap for flattened method
        if args.plot:
            save_path = None
            if args.save_plot:
                save_path = args.save_plot.replace('.png', '_flattened.png')
            plot_correlation_heatmap(corr_matrix, p_matrix, 'flattened', save_path)
    
    if args.method in ['average', 'both']:
        corr_matrix, p_matrix = compute_correlation_matrix(all_datasets, all_labels, 'average')
        print(f"\n📊 AVERAGE METHOD CORRELATION MATRIX:")
        print("Correlation Coefficients:")
        print(corr_matrix.round(4))
        print("\nP-values:")
        print(p_matrix)
        
        print(f"\nSignificance Matrix (α = 0.05):")
        significance = (p_matrix < 0.05).astype(str)
        significance = significance.replace({'True': 'Significant', 'False': 'Not Significant'})
        print(significance)
        
        # Generate heatmap for average method
        if args.plot:
            save_path = None
            if args.save_plot:
                save_path = args.save_plot.replace('.png', '_average.png')
            plot_correlation_heatmap(corr_matrix, p_matrix, 'average', save_path)


if __name__ == "__main__":
    main()
