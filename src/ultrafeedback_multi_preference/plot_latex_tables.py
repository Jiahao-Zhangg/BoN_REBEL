#!/usr/bin/env python3
"""Create LaTeX-ready visualizations from pairwise comparison results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_results_file(filepath):
    """Parse the pairwise tables text file and extract the three tables."""
    tables = {}
    current_table = None
    current_data = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if "Table 1: Helpfulness Comparison" in line:
            current_table = "helpfulness"
            current_data = []
        elif "Table 2: Correctness Comparison" in line:
            current_table = "correctness"
            current_data = []
        elif "Table 3: Min(Helpfulness, Correctness)" in line:
            current_table = "worst_dimension"
            current_data = []
        elif current_table and line and not line.startswith("1[") and not line.startswith("min_i") and not line.startswith("Rows:"):
            # Parse data lines like "Base       0.500   0.437   0.327"
            parts = line.split()
            if len(parts) == 4:  # model name + 3 values
                values = [float(x) for x in parts[1:]]
                current_data.append(values)
                
                # If we have 3 rows, save the table
                if len(current_data) == 3:
                    tables[current_table] = np.array(current_data)
    
    return tables

def create_latex_visualization(tables, output_path):
    """Create a LaTeX-ready visualization with the specified formatting."""
    
    # Set up the plot with LaTeX-style formatting
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    # Model names
    model_names = ['Base', 'REBEL', 'Ours']
    
    # Create the figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Table titles as requested
    titles = [
        'Win Rate on Helpfulness',
        'Win Rate on Correctness', 
        'Win Rate on Worst Dimension'
    ]
    
    # Table data
    table_data = [
        tables['helpfulness'],
        tables['correctness'],
        tables['worst_dimension']
    ]
    
    # Create heatmaps
    for i, (table, title) in enumerate(zip(table_data, titles)):
        # Create heatmap without individual colorbars
        im = axes[i].imshow(table, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
        
        # Set title
        axes[i].set_title(title, fontsize=18, pad=20)
        
        # Set tick labels
        axes[i].set_xticks(range(3))
        axes[i].set_yticks(range(3))
        axes[i].set_xticklabels(model_names, fontsize=14)
        axes[i].set_yticklabels(model_names, fontsize=14)
        
        # Add value annotations
        for j in range(3):
            for k in range(3):
                text = axes[i].text(k, j, f'{table[j, k]:.3f}', 
                                  ha="center", va="center", 
                                  color="black", fontsize=12, fontweight='bold')
        
        # Remove axis labels as requested (no "Row Model" and "Column Model")
        # Just clean up the appearance
        axes[i].tick_params(axis='both', which='major', labelsize=14)
    
    # Add a single colorbar for all plots
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"LaTeX-ready visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create LaTeX-ready visualizations from pairwise comparison results')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to the pairwise tables text file')
    parser.add_argument('--output', '-o', 
                       help='Output path for the visualization (default: same directory as input)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{input_name}_latex.png")
    
    # Parse the results file
    print(f"Reading results from: {args.input}")
    tables = parse_results_file(args.input)
    
    if len(tables) != 3:
        print(f"Error: Expected 3 tables, found {len(tables)}")
        return
    
    print("Found tables:", list(tables.keys()))
    
    # Create the visualization
    create_latex_visualization(tables, args.output)

if __name__ == "__main__":
    main()
