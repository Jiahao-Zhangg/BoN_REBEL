"""Plot evaluation results from text file with custom labels and larger text."""

import matplotlib.pyplot as plt
import numpy as np
import re
import os

def parse_results_file(file_path):
    """Parse the evaluation results text file to extract model scores."""
    results = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract model results using regex
    model_pattern = r'Model: (.*?)\n\s*Helpfulness: ([\d.]+)\n\s*Correctness: ([\d.]+)'
    matches = re.findall(model_pattern, content)
    
    for model_name, helpfulness, correctness in matches:
        results[model_name] = {
            'helpfulness': float(helpfulness),
            'correctness': float(correctness)
        }
    
    return results

def create_plot(results, output_dir):
    """Create scatter plot with custom labels."""
    
    # Define label mapping
    label_mapping = {
        'meta-llama/Llama-3.2-3B-Instruct': 'Base',
        'zjhhhh/REBEL_1e4_ver2': 'REBEL',
        'zjhhhh/Multi_Preference_REBEL_1e4': 'Ours',
        'zjhhhh/BoN_REBEL_1e4': None  # This will be dropped
    }
    
    # Filter and prepare data
    filtered_results = {}
    for model_name, scores in results.items():
        if model_name in label_mapping and label_mapping[model_name] is not None:
            filtered_results[label_mapping[model_name]] = scores
    
    print("Filtered results:")
    for label, scores in filtered_results.items():
        print(f"  {label}: Helpfulness={scores['helpfulness']:.4f}, Correctness={scores['correctness']:.4f}")
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    model_labels = []
    helpfulness_scores = []
    correctness_scores = []
    
    for label, scores in filtered_results.items():
        model_labels.append(label)
        helpfulness_scores.append(scores['helpfulness'])
        correctness_scores.append(scores['correctness'])
    
    # Create scatter plot with different colors for each model
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_labels)))
    
    for i, (label, help_score, corr_score) in enumerate(zip(model_labels, helpfulness_scores, correctness_scores)):
        plt.scatter(help_score, corr_score, c=[colors[i]], s=200, alpha=0.8, label=label, edgecolors='black', linewidth=1.5)
        
        # Add model label as text annotation with larger font
        plt.annotate(label, (help_score, corr_score), 
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=16, fontweight='bold', alpha=0.9)
    
    # Customize plot appearance
    plt.xlabel("Helpfulness Score", fontsize=16, fontweight='bold')
    plt.ylabel("Correctness Score", fontsize=16, fontweight='bold')
    plt.title("Model Performance: Helpfulness vs Correctness", fontsize=18, fontweight='bold', pad=20)
    
    # Make legend larger
    plt.legend(fontsize=14, loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Customize grid and ticks
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Set axis limits with some padding
    x_margin = (max(helpfulness_scores) - min(helpfulness_scores)) * 0.1
    y_margin = (max(correctness_scores) - min(correctness_scores)) * 0.1
    plt.xlim(min(helpfulness_scores) - x_margin, max(helpfulness_scores) + x_margin)
    plt.ylim(min(correctness_scores) - y_margin, max(correctness_scores) + y_margin)
    
    # Save plot
    plot_filename = os.path.join(output_dir, "helpfulness_vs_correctness_replotted.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nPlot saved to: {plot_filename}")
    
    return plot_filename

def main():
    # File paths
    results_file = "BoN_REBEL/evaluation_results/evaluation_results_20250820_013714.txt"
    output_dir = "BoN_REBEL/evaluation_results"
    
    # Parse results
    print("Parsing evaluation results...")
    results = parse_results_file(results_file)
    
    # Create plot
    print("Creating plot...")
    plot_filename = create_plot(results, output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()



