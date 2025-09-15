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

MAP = {'A': 4, 'B': 0, 'Tie': 2, -1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
MAP_TERNARY = {0:0, 1:0, 2:2, 3:4, 4:4}

def round_to_discrete_values(value, discrete_values=[0, 0.25, 0.5, 0.75, 1]):
    """Round a value to the closest value in the discrete set."""
    return min(discrete_values, key=lambda x: abs(x - value))

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


def load_judge_preferences(dataset_name, measure='majority', reward_mode=0):
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
            if reward_mode > 0:
                # Use reward scores
                if measure == 'mean' and 'response_0_mean' in row and 'response_1_mean' in row:
                    response_0 = row['response_0_mean']
                    response_1 = row['response_1_mean']
                    
                    if reward_mode == 1:
                        # Mode 1: Use r_0 - r_1
                        pref = [r0 - r1 for r0, r1 in zip(response_0, response_1)]
                        print(f"Row {i} (reward mean, diff): response_0={response_0}, response_1={response_1}, diff={pref}")
                    elif reward_mode == 2:
                        # Mode 2: Use Bradley-Terry e(r_0)/(e(r_0)+e(r_1)) rounded to {0, 0.25, 0.5, 0.75, 1}
                        import math
                        pref_raw = [math.exp(r0) / (math.exp(r0) + math.exp(r1)) for r0, r1 in zip(response_0, response_1)]
                        pref = [round_to_discrete_values(p) for p in pref_raw]
                        print(f"Row {i} (reward mean, Bradley-Terry): response_0={response_0}, response_1={response_1}, raw_bradley_terry={pref_raw}, rounded_bradley_terry={pref}")
                    
                    judge_preferences.append(pref)
                elif measure == 'majority' and 'response_0_majority' in row and 'response_1_majority' in row:
                    response_0 = row['response_0_majority']
                    response_1 = row['response_1_majority']
                    
                    if reward_mode == 1:
                        # Mode 1: Use r_0 - r_1
                        pref = [r0 - r1 for r0, r1 in zip(response_0, response_1)]
                        print(f"Row {i} (reward majority, diff): response_0={response_0}, response_1={response_1}, diff={pref}")
                    elif reward_mode == 2:
                        # Mode 2: Use Bradley-Terry e(r_0)/(e(r_0)+e(r_1)) rounded to {0, 0.25, 0.5, 0.75, 1}
                        import math
                        beta = 25
                        pref_raw = [math.exp(r0/beta) / (math.exp(r0/beta) + math.exp(r1/beta)) for r0, r1 in zip(response_0, response_1)]
                        pref = [round_to_discrete_values(p) for p in pref_raw]
                        print(f"Row {i} (reward majority, Bradley-Terry): response_0={response_0}, response_1={response_1}, raw_bradley_terry={pref_raw}, rounded_bradley_terry={pref}")
                    
                    judge_preferences.append(pref)
                else:
                    print(f"Warning: Row {i} missing reward columns for measure '{measure}'")
                    print(f"Available columns: {list(row.keys())}")
                # Skip preference judgment logic when in reward mode
                continue
            else:
                # Use preference judgments (original logic)
                if measure == 'majority' and 'response_0_1_judged_preference_majority' in row:
                    pref = row['response_0_1_judged_preference_majority']
                    pref = list(map(lambda x: MAP[x], pref))
                    judge_preferences.append(pref)
                elif measure == 'mean' and 'response_0_1_judged_preference_mean' in row:
                    pref = row['response_0_1_judged_preference_mean']
                    # pref = list(map(lambda x: MAP[x], pref))
                    judge_preferences.append(pref)
                    print(f"Row {i}: {pref}")
                elif measure == 'mean' and 'response_0_1_judged_preference' in row:
                    pref = row['response_0_1_judged_preference']
                    # pref = list(map(lambda x: MAP[x], pref))
                    judge_preferences.append(pref)
                    print(f"Row {i}: {pref}")
                elif 'response_0_1_judged_preference' in row:
                    pref = row['response_0_1_judged_preference']
                    pref = list(map(lambda x: MAP[x], pref))
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


def flatten_arrays(arrays):
    """Flatten a list of arrays into a single list."""
    flattened = []
    for array in arrays:
        if isinstance(array, list):
            flattened.extend(array)
        else:
            flattened.append(array)
    return flattened


def compute_l2_loss(x_data, y_data, method='flattened'):
    """
    Compute L2 loss (Mean Squared Error) between two datasets.
    
    Args:
        x_data: List of arrays (human annotations)
        y_data: List of arrays (judge preferences)
        method: 'flattened' (concatenate all arrays) or 'average' (compute loss for each array pair, then average)
    
    Returns:
        l2_loss: L2 loss value
        mse: Mean squared error (same as L2 loss)
    """
    print(f"\nComputing L2 loss using method: {method}")
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
        
        # Compute L2 loss (MSE)
        x_array = np.array(x_flat)
        y_array = np.array(y_flat)
        squared_errors = (x_array - y_array) ** 2
        l2_loss = np.mean(squared_errors)
        
        print(f"\nFlattened data L2 loss:")
        print(f"X values (first 10): {x_flat[:10]}")
        print(f"Y values (first 10): {y_flat[:10]}")
        print(f"Squared errors (first 10): {squared_errors[:10]}")
        
    elif method == 'average':
        # Compute L2 loss for each array pair, then average the losses
        losses = []
        
        print(f"\nComputing L2 loss for each array pair:")
        
        for i, (x_arr, y_arr) in enumerate(zip(x_data, y_data)):
            # Ensure arrays have same length
            min_len = min(len(x_arr), len(y_arr))
            if min_len < 1:
                print(f"  Array pair {i}: Skipping (no data points)")
                continue
                
            x_trimmed = np.array(x_arr[:min_len])
            y_trimmed = np.array(y_arr[:min_len])
            
            try:
                squared_errors = (x_trimmed - y_trimmed) ** 2
                mse = np.mean(squared_errors)
                losses.append(mse)
                print(f"  Array pair {i}: L2 loss={mse:.4f} | X={x_trimmed} | Y={y_trimmed}")
            except Exception as e:
                print(f"  Array pair {i}: Error computing L2 loss: {e}")
                continue
        
        if not losses:
            print("No valid L2 losses computed!")
            return np.nan, np.nan
        
        # Average the losses
        l2_loss = np.mean(losses)
        
        print(f"\nIndividual L2 losses: {[f'{loss:.4f}' for loss in losses]}")
        print(f"Average L2 loss: {l2_loss:.4f}")
        print(f"Valid array pairs: {len(losses)}/{len(x_data)}")
    
    return l2_loss, l2_loss  # Return same value twice for consistency with correlation function


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


def compute_l2_loss_matrix(datasets_data, labels, method='flattened'):
    """
    Compute L2 loss matrix between multiple datasets.
    
    Args:
        datasets_data: List of datasets [X, Y, Z, W, ...]
        labels: List of labels ['X', 'Y', 'Z', 'W', ...]
        method: L2 loss method
    
    Returns:
        l2_loss_matrix: DataFrame with L2 loss values
    """
    n_datasets = len(datasets_data)
    l2_matrix = np.zeros((n_datasets, n_datasets))
    
    print(f"\nComputing {n_datasets}x{n_datasets} L2 loss matrix using method: {method}")
    
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i == j:
                # Diagonal elements (self-loss should be 0)
                l2_matrix[i, j] = 0.0
            else:
                # Compute L2 loss between datasets i and j
                l2_loss, _ = compute_l2_loss(datasets_data[i], datasets_data[j], method)
                l2_matrix[i, j] = l2_loss
                print(f"  {labels[i]} vs {labels[j]}: L2 loss={l2_loss:.4f}")
    
    # Create DataFrame for better visualization
    l2_df = pd.DataFrame(l2_matrix, index=labels, columns=labels)
    
    return l2_df


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


def plot_l2_loss_heatmap(l2_matrix, method, save_path=None):
    """
    Plot L2 loss matrix as heatmap.
    
    Args:
        l2_matrix: DataFrame with L2 loss values
        method: Method used for L2 loss computation
        save_path: Optional path to save the plot
    """
    # Set up the matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot L2 loss heatmap
    mask = np.zeros_like(l2_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # Mask upper triangle
    
    sns.heatmap(l2_matrix, mask=mask, annot=True, fmt='.4f', 
                cmap='viridis', square=True, ax=ax,
                cbar_kws={"shrink": .8, "label": "L2 Loss"})
    ax.set_title(f'L2 Loss Matrix ({method.title()} Method)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Datasets', fontweight='bold')
    ax.set_ylabel('Datasets', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"L2 loss heatmap saved to: {save_path}")
    
    plt.show()


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
    parser = argparse.ArgumentParser(description="Compute Pearson correlation matrix and/or L2 loss matrix between annotations")
    parser.add_argument("--human-file", default="src/ultrafeedback_judge/human_annotation.txt",
                       help="Path to human annotation file")
    parser.add_argument("--datasets", nargs='+', 
                       default=["zjhhhh/human-scored-1.5B_judge_preference_5score_qwen2.5_72b_ver2_run2"],
                       help="List of Hugging Face dataset names (Y, Z, W, ...)")
    parser.add_argument("--dataset-labels", nargs='+',
                       default=["Y"],
                       help="Labels for the datasets (Y, Z, W, ...)")
    parser.add_argument("--method", choices=['flattened', 'average', 'both'], default='flattened',
                       help="Correlation method")
    parser.add_argument("--plot", action='store_true',
                       help="Generate heatmap visualization")
    parser.add_argument("--save-plot", type=str, default="correlation_heatmap.png",
                       help="Path to save the heatmap plot (e.g., 'correlation_heatmap.png')")
    parser.add_argument("--ternary", action='store_true', help="Use ternary preference")
    parser.add_argument("--measure", choices=['majority', 'mean'], default='majority',
                       help="Measure to use for preference")
    parser.add_argument("--reward", type=int, choices=[0, 1, 2], default=0,
                       help="Reward mode: 0=no use, 1=use r_0-r_1, 2=use Bradley-Terry e(r_0)/(e(r_0)+e(r_1))")
    parser.add_argument("--compute-l2-loss", action='store_true',
                       help="Compute L2 loss matrix in addition to correlation matrix")
    parser.add_argument("--l2-only", action='store_true',
                       help="Compute only L2 loss matrix (skip correlation analysis)")
    args = parser.parse_args()
    
    # Ensure we have labels for all datasets
    if len(args.dataset_labels) != len(args.datasets):
        print("Warning: Number of dataset labels doesn't match number of datasets. Using default labels.")
        args.dataset_labels = [f"Dataset_{i+1}" for i in range(len(args.datasets))]
    
    print("=== Pearson Correlation Matrix Analysis ===")
    
    # Read human annotations (X)
    print(f"\n1. Reading human annotations from: {args.human_file}")
    x_data = read_human_annotations(args.human_file, args.ternary)
    print(f"Loaded {len(x_data)} annotation arrays:")
    for i, arr in enumerate(x_data):
        print(f"  Array {i}: {arr}")
    
    # Load all judge preference datasets (Y, Z, W, ...)
    all_datasets = [x_data]  # Start with human data (X)
    all_labels = ['human']  # Start with human label
    
    print(f"\n2. Loading {len(args.datasets)} judge preference datasets:")
    for i, dataset_name in enumerate(args.datasets):
        print(f"  Loading {args.dataset_labels[i]} from: {dataset_name}")
        judge_data = load_judge_preferences(dataset_name, args.measure, args.reward)
        
        if judge_data is None:
            print(f"Failed to load {args.dataset_labels[i]}. Skipping.")
            continue
        
        print(f"  Loaded {len(judge_data)} arrays for {args.dataset_labels[i]}:")
        for j, arr in enumerate(judge_data):
            print(f"    Array {j}: {arr}")
        
        all_datasets.append(judge_data)
        all_labels.append(args.dataset_labels[i])  # Only add label if dataset loaded successfully
    
    # Verify all datasets have the same number of arrays
    if len(set(len(dataset) for dataset in all_datasets)) > 1:
        print("\nWarning: Datasets have different numbers of arrays:")
        for i, dataset in enumerate(all_datasets):
            print(f"  {all_labels[i]}: {len(dataset)} arrays")
        
        # Truncate to minimum length
        min_len = min(len(dataset) for dataset in all_datasets)
        print(f"Truncating all datasets to {min_len} arrays")
        all_datasets = [dataset[:min_len] for dataset in all_datasets]
    
    # Compute L2 loss matrices if requested
    if args.compute_l2_loss or args.l2_only:
        print(f"\n3. Computing L2 loss matrix")
        
        if args.method in ['flattened', 'both']:
            l2_matrix = compute_l2_loss_matrix(all_datasets, all_labels, 'flattened')
            print(f"\n📊 FLATTENED METHOD L2 LOSS MATRIX:")
            print("L2 Loss Values:")
            print(l2_matrix.round(4))
            
            # Generate heatmap for flattened method L2 loss
            if args.plot:
                save_path = None
                if args.save_plot:
                    save_path = args.save_plot.replace('.png', '_l2_loss_flattened.png')
                plot_l2_loss_heatmap(l2_matrix, 'flattened', save_path)
        
        if args.method in ['average', 'both']:
            l2_matrix = compute_l2_loss_matrix(all_datasets, all_labels, 'average')
            print(f"\n📊 AVERAGE METHOD L2 LOSS MATRIX:")
            print("L2 Loss Values:")
            print(l2_matrix.round(4))
            
            # Generate heatmap for average method L2 loss
            if args.plot:
                save_path = None
                if args.save_plot:
                    save_path = args.save_plot.replace('.png', '_l2_loss_average.png')
                plot_l2_loss_heatmap(l2_matrix, 'average', save_path)
    
    # Compute correlation matrices (skip if l2-only is specified)
    if not args.l2_only:
        print(f"\n{'4' if args.compute_l2_loss else '3'}. Computing Pearson correlation matrix")
        
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
                    save_path = args.save_plot.replace('.png', '_correlation_flattened.png')
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
                    save_path = args.save_plot.replace('.png', '_correlation_average.png')
                plot_correlation_heatmap(corr_matrix, p_matrix, 'average', save_path)


if __name__ == "__main__":
    main()
