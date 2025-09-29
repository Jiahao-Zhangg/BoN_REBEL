#!/usr/bin/env python3
"""
Script to recreate the results table with mean±std format and color coding.
"""

import numpy as np
import pandas as pd
from colorama import Fore, Back, Style
import colorama
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns

# Initialize colorama for cross-platform colored output
colorama.init()

def calculate_mean_std(values):
    """Calculate mean and standard deviation from a list of values."""
    arr = np.array(values)
    return arr.mean(), arr.std()

def color_value(value, min_val, max_val, is_kl=False):
    """Color code values based on their range. For KL divergence, lower is better."""
    if is_kl:
        # For KL divergence, lower (more negative) is better - color green
        # Higher (less negative) is worse - color red
        if value <= min_val + 0.2 * (max_val - min_val):
            return f"{Fore.GREEN}{value:.4f}{Style.RESET_ALL}"
        elif value >= max_val - 0.2 * (max_val - min_val):
            return f"{Fore.RED}{value:.4f}{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{value:.4f}{Style.RESET_ALL}"
    else:
        # For win rates and preferences, higher is better - color green
        # Lower is worse - color red
        if value >= max_val - 0.2 * (max_val - min_val):
            return f"{Fore.GREEN}{value:.4f}{Style.RESET_ALL}"
        elif value <= min_val + 0.2 * (max_val - min_val):
            return f"{Fore.RED}{value:.4f}{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{value:.4f}{Style.RESET_ALL}"

def format_mean_std(mean, std, min_val, max_val, is_kl=False):
    """Format mean±std with color coding."""
    colored_mean = color_value(mean, min_val, max_val, is_kl)
    return f"{colored_mean}±{std:.4f}"

def format_mean_std_plain(mean, std):
    """Format mean±std without color coding for saving to file."""
    return f"{mean:.4f}±{std:.4f}"

def get_color_class(value, min_val, max_val, is_kl=False):
    """Get CSS class for color coding in HTML."""
    if is_kl:
        if value <= min_val + 0.2 * (max_val - min_val):
            return "best"
        elif value >= max_val - 0.2 * (max_val - min_val):
            return "worst"
        else:
            return "medium"
    else:
        if value >= max_val - 0.2 * (max_val - min_val):
            return "best"
        elif value <= min_val + 0.2 * (max_val - min_val):
            return "worst"
        else:
            return "medium"

def save_plain_text(data, winrate_stats, preference_stats, filename="results_table.txt"):
    """Save table as plain text without colors."""
    with open(filename, 'w') as f:
        f.write("Results Table: Mean ± Standard Deviation\n")
        f.write("=" * 80 + "\n\n")
        
        # Header
        f.write(f"{'Configuration':<30} | {'Winrate vs Qwen3-14B':<20} | {'Mean preference vs base':<25} | {'KL':<15}\n")
        f.write("-" * 95 + "\n")
        
        # Data rows
        for i, config in enumerate(data['Configuration']):
            wr_mean, wr_std = winrate_stats[i]
            pref_mean, pref_std = preference_stats[i]
            kl_value = data['KL'][i]
            
            wr_formatted = format_mean_std_plain(wr_mean, wr_std)
            pref_formatted = format_mean_std_plain(pref_mean, pref_std)
            
            f.write(f"{config:<30} | {wr_formatted:<20} | {pref_formatted:<25} | {kl_value:<15.4f}\n")
        
        f.write("=" * 80 + "\n")

def save_html(data, winrate_stats, preference_stats, wr_min, wr_max, pref_min, pref_max, kl_min, kl_max, filename="results_table.html"):
    """Save table as HTML with color coding."""
    css = """
    <style>
        table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .best { color: #008000; font-weight: bold; }
        .medium { color: #FFA500; }
        .worst { color: #FF0000; }
        .header { text-align: center; margin: 20px 0; }
        .legend { margin: 10px 0; }
    </style>
    """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Results Table</title>
        {css}
    </head>
    <body>
        <div class="header">
            <h1>Results Table: Mean ± Standard Deviation with Color Coding</h1>
            <div class="legend">
                <span class="best">Green: Best performance</span> | 
                <span class="medium">Orange: Medium performance</span> | 
                <span class="worst">Red: Worst performance</span>
                <br><em>Note: For KL divergence, lower (more negative) values are better</em>
            </div>
        </div>
        
        <table>
            <tr>
                <th>Configuration</th>
                <th>Winrate vs Qwen3-14B</th>
                <th>Mean preference vs base [0,4]</th>
                <th>KL</th>
            </tr>
    """
    
    for i, config in enumerate(data['Configuration']):
        wr_mean, wr_std = winrate_stats[i]
        pref_mean, pref_std = preference_stats[i]
        kl_value = data['KL'][i]
        
        wr_class = get_color_class(wr_mean, wr_min, wr_max)
        pref_class = get_color_class(pref_mean, pref_min, pref_max)
        kl_class = get_color_class(kl_value, kl_min, kl_max, is_kl=True)
        
        wr_formatted = format_mean_std_plain(wr_mean, wr_std)
        pref_formatted = format_mean_std_plain(pref_mean, pref_std)
        
        html += f"""
            <tr>
                <td>{config}</td>
                <td><span class="{wr_class}">{wr_formatted}</span></td>
                <td><span class="{pref_class}">{pref_formatted}</span></td>
                <td><span class="{kl_class}">{kl_value:.4f}</span></td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html)

def get_color_for_value(value, min_val, max_val, is_kl=False):
    """Get RGB color for a value based on performance (for text coloring)."""
    # Normalize to 0-1
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    
    # For KL, reverse since lower is better
    if is_kl:
        normalized = 1 - normalized
    
    # Use matplotlib colormap to get RGB values
    cmap = plt.cm.RdYlGn
    rgb = cmap(normalized)[:3]  # Get RGB, ignore alpha
    return rgb

def save_figure_heatmap(data, winrate_stats, preference_stats, filename="results_heatmap.png"):
    """Save table as a figure with number coloring instead of background."""
    # Prepare data for heatmap
    configs = [config.replace('β', 'beta').replace('η', 'eta') for config in data['Configuration']]
    
    # Create data matrix
    wr_means = [stats[0] for stats in winrate_stats]
    wr_stds = [stats[1] for stats in winrate_stats]
    pref_means = [stats[0] for stats in preference_stats]
    pref_stds = [stats[1] for stats in preference_stats]
    kl_values = data['KL']
    
    # Get min/max for color scaling
    wr_min, wr_max = min(wr_means), max(wr_means)
    pref_min, pref_max = min(pref_means), max(pref_means)
    kl_min, kl_max = min(kl_values), max(kl_values)
    
    # Create figure with one subplot for table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data with colored text
    table_data = []
    headers = ['Configuration', 'Winrate vs Qwen3-14B', 'Mean preference vs base', 'KL']
    
    for i, config in enumerate(configs):
        row = [
            config,
            f'{wr_means[i]:.4f}±{wr_stds[i]:.4f}',
            f'{pref_means[i]:.4f}±{pref_stds[i]:.4f}',
            f'{kl_values[i]:.4f}'
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center', 
                     colWidths=[0.35, 0.22, 0.22, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color code text based on performance (keep background white)
    for i in range(len(configs)):
        # Set all cells to white background
        table[(i+1, 0)].set_facecolor('white')
        table[(i+1, 1)].set_facecolor('white')
        table[(i+1, 2)].set_facecolor('white')
        table[(i+1, 3)].set_facecolor('white')
        
        # Color the numbers based on performance
        # Winrate column (index 1)
        wr_color = get_color_for_value(wr_means[i], wr_min, wr_max)
        table[(i+1, 1)].set_text_props(color=wr_color, weight='bold')
        
        # Preference column (index 2)
        pref_color = get_color_for_value(pref_means[i], pref_min, pref_max)
        table[(i+1, 2)].set_text_props(color=pref_color, weight='bold')
        
        # KL column (index 3)
        kl_color = get_color_for_value(kl_values[i], kl_min, kl_max, is_kl=True)
        table[(i+1, 3)].set_text_props(color=kl_color, weight='bold')
    
    # Style headers
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Add title and legend
    ax.set_title('Results Table with Colored Numbers\n(Green = Better Performance, Red = Worse Performance)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add a legend explaining the color coding
    legend_text = (
        "Color Legend:\n"
        "• Green numbers: Better performance\n"
        "• Yellow numbers: Medium performance\n"
        "• Red numbers: Worse performance\n"
        "• For KL divergence: Lower (more negative) values are better"
    )
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def save_separate_heatmaps(data, winrate_stats, preference_stats, filename_prefix="metric_heatmap"):
    """Save separate heatmaps for each metric."""
    configs = [config.replace('β', 'beta').replace('η', 'eta') for config in data['Configuration']]
    
    # Extract means
    wr_means = [stats[0] for stats in winrate_stats]
    pref_means = [stats[0] for stats in preference_stats]
    kl_values = data['KL']
    
    metrics = {
        'Winrate by Qwen3-14B': wr_means,
        'Mean Preference vs Base': pref_means,
        'KL Divergence': kl_values
    }
    
    for metric_name, values in metrics.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap data (single column)
        heatmap_data = np.array(values).reshape(-1, 1)
        
        # For KL, use reversed colormap since lower is better
        cmap = 'RdYlGn_r' if metric_name == 'KL Divergence' else 'RdYlGn'
        
        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto')
        
        # Set labels
        ax.set_xticks([0])
        ax.set_xticklabels([metric_name], fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=11)
        
        # Add value annotations
        for i, value in enumerate(values):
            ax.text(0, i, f'{value:.4f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(metric_name, rotation=270, labelpad=20)
        
        ax.set_title(f'{metric_name} Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = f"{filename_prefix}_{metric_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved {metric_name} heatmap to: {Fore.CYAN}{filename}{Style.RESET_ALL}")

# Data from the table
data = {
    'Configuration': [
        'fixed β = 10, η = 1e4',
        'fixed β = 0.1, η = 1e4', 
        'fixed β = 0.01, η = 1e4',
        'fixed β = 1, η = 1e4',
        'fixed β = 1, η = 2.5e3',
        'fixed β = 1, η = 1e5',
        'fixed β = 1, η = 1e6',
        'pooling β = 1, η = 1e4',
        'pooling β = 1, η = 1e5',
        'pooling β = 1, η = 1e6',
        'expand β = 1, η = 1e4',
        'bn β = 1, η = 1e4',
        'reward η = 1e4',
        'gap β = 1, η = 1e4'
    ],
    'Winrate_vs_Qwen3-14B': [
        [0.6351, 0.6602, 0.6521],
        [0.6327, 0.6412, 0.6439],
        [0.6134, 0.6125, 0.6214],
        [0.6190, 0.6327, 0.6439],
        [0.6280, 0.6318, 0.6186],
        [0.6467, 0.6277, 0.6137],
        [0.6408, 0.6573, 0.6538],
        [0.6769, 0.6340, 0.6637],
        [0.6421, 0.6875, 0.6839],
        [0.6773, 0.6708, 0.6613],
        [0.6449, 0.6670, 0.6324],
        [0.6327, 0.6460, 0.6154],
        [0.6940, 0.6727, 0.6659],
        [0.6974, 0.6995, 0.7093]
    ],
    'Mean_preference_vs_base': [
        [2.1404, 2.1707, 2.1686],
        [2.1500, 2.1583, 2.1683],
        [2.1319, 2.1261, 2.1437],
        [2.1661, 2.1488, 2.1563],
        [2.1483, 2.1601, 2.1557],
        [2.1834, 2.1589, 2.1490],
        [2.1889, 2.1921, 2.1810],
        [2.1981, 2.1732, 2.1924],
        [2.1681, 2.2124, 2.1799],
        [2.1702, 2.1825, 2.1646],
        [2.1999, 2.2058, 2.1879],
        [2.1553, 2.1650, 2.1649],
        [2.2617, 2.2385, 2.2531],
        [2.2519, 2.2434, 2.2573]
    ],
    'KL': [
        -13.5947,
        -16.0463,
        -14.7535,
        -14.2502,
        -15.6881,
        -18.9315,
        -17.1991,
        -43.7073,
        -44.5459,
        -44.7011,
        -758.4074,
        -17.1357,
        -107.2789,
        -131.4244
    ]
}

def main():
    print("=" * 100)
    print(f"{Style.BRIGHT}Results Table: Mean ± Standard Deviation with Color Coding{Style.RESET_ALL}")
    print("=" * 100)
    print(f"{Fore.GREEN}Green: Best performance{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Yellow: Medium performance{Style.RESET_ALL}")
    print(f"{Fore.RED}Red: Worst performance{Style.RESET_ALL}")
    print("Note: For KL divergence, lower (more negative) values are better")
    print("=" * 100)
    
    # Calculate mean and std for each metric
    winrate_stats = []
    preference_stats = []
    
    # Get all mean values for min/max calculation
    winrate_means = []
    preference_means = []
    
    for i in range(len(data['Configuration'])):
        wr_mean, wr_std = calculate_mean_std(data['Winrate_vs_Qwen3-14B'][i])
        pref_mean, pref_std = calculate_mean_std(data['Mean_preference_vs_base'][i])
        
        winrate_stats.append((wr_mean, wr_std))
        preference_stats.append((pref_mean, pref_std))
        
        winrate_means.append(wr_mean)
        preference_means.append(pref_mean)
    
    # Get min/max for color coding
    wr_min, wr_max = min(winrate_means), max(winrate_means)
    pref_min, pref_max = min(preference_means), max(preference_means)
    kl_min, kl_max = min(data['KL']), max(data['KL'])
    
    # Print table header
    print(f"{'Configuration':<30} | {'Winrate vs Qwen3-14B':<25} | {'Mean preference vs base [0,4]':<30} | {'KL':<15}")
    print("-" * 105)
    
    # Print each row
    for i, config in enumerate(data['Configuration']):
        wr_mean, wr_std = winrate_stats[i]
        pref_mean, pref_std = preference_stats[i]
        kl_value = data['KL'][i]
        
        # Format with colors
        wr_formatted = format_mean_std(wr_mean, wr_std, wr_min, wr_max)
        pref_formatted = format_mean_std(pref_mean, pref_std, pref_min, pref_max)
        kl_colored = color_value(kl_value, kl_min, kl_max, is_kl=True)
        
        print(f"{config:<30} | {wr_formatted:<35} | {pref_formatted:<40} | {kl_colored}")
    
    print("=" * 100)
    
    # Summary statistics
    print(f"\n{Style.BRIGHT}Summary Statistics:{Style.RESET_ALL}")
    print(f"Winrate - Best: {color_value(wr_max, wr_min, wr_max)}, Worst: {color_value(wr_min, wr_min, wr_max)}")
    print(f"Preference - Best: {color_value(pref_max, pref_min, pref_max)}, Worst: {color_value(pref_min, pref_min, pref_max)}")
    print(f"KL - Best: {color_value(kl_min, kl_min, kl_max, is_kl=True)}, Worst: {color_value(kl_max, kl_min, kl_max, is_kl=True)}")
    
    # Save files
    print(f"\n{Style.BRIGHT}Saving files...{Style.RESET_ALL}")
    save_plain_text(data, winrate_stats, preference_stats)
    save_html(data, winrate_stats, preference_stats, wr_min, wr_max, pref_min, pref_max, kl_min, kl_max)
    save_figure_heatmap(data, winrate_stats, preference_stats)
    save_separate_heatmaps(data, winrate_stats, preference_stats)
    
    print(f"✓ Saved plain text table to: {Fore.CYAN}results_table.txt{Style.RESET_ALL}")
    print(f"✓ Saved HTML table to: {Fore.CYAN}results_table.html{Style.RESET_ALL}")
    print(f"✓ Saved combined heatmap figure to: {Fore.CYAN}results_heatmap.png{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
