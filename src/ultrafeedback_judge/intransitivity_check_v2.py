import argparse
import numpy as np
import networkx as nx
from datasets import load_dataset
from itertools import combinations
import matplotlib.pyplot as plt


def transform_digit_to_preference(type, digit_scores):
    """Transform raw scores to preference values."""
    if type == "5score":
        return digit_scores/4
    elif type == "ternary":
        MAP = {'A': 1, 'B': 0, 'Tie': 0.5}
        return MAP[digit_scores]


def process_dataset(dataset, n_response):
    """
    Process dataset to extract and flatten preference columns for all pairs.
    Returns a dictionary with keys as (i,j) tuples and values as flattened lists of preferences.
    """
    preference_data = {}
    
    # Initialize all possible pairs
    for i in range(n_response):
        for j in range(i+1, n_response):
            preference_data[(i, j)] = []
    
    for item in dataset:
        # Extract and flatten preferences for all pairs
        for i in range(n_response):
            for j in range(i+1, n_response):
                column_name = f'response_{i}_{j}_judged_preference_majority'
                if column_name in item:
                    # Flatten the list - extend adds each element individually
                    preference_data[(i, j)].extend(item[column_name])
                else:
                    print(f"Warning: Column {column_name} not found in dataset")
    
    return preference_data


def build_directed_graph(preference_values, n_response, data_type):
    """
    Build a directed graph from preference values.
    If transformed value is 1, build edge from i to j.
    If transformed value is 0, build edge from j to i.
    If transformed value is 0.5 (tie), merge nodes i and j.
    """
    G = nx.DiGraph()
    
    # Add all nodes
    for i in range(n_response):
        G.add_node(i)
    
    # Track nodes that should be merged (tied nodes)
    merged_nodes = {}  # maps original node -> representative node
    
    # First pass: identify tied nodes and create merge mapping
    for (i, j), values in preference_values.items():
        for value in values:
            pref = transform_digit_to_preference(data_type, value)
            if pref == 0.5:  # tie case - merge nodes
                # Find representatives for both nodes
                rep_i = merged_nodes.get(i, i)
                rep_j = merged_nodes.get(j, j)
                
                # Merge to the smaller representative (for consistency)
                if rep_i != rep_j:
                    min_rep = min(rep_i, rep_j)
                    max_rep = max(rep_i, rep_j)
                    
                    # Update all nodes that point to max_rep to point to min_rep
                    for node in list(merged_nodes.keys()):
                        if merged_nodes[node] == max_rep:
                            merged_nodes[node] = min_rep
                    
                    # Update the mapping for the current nodes
                    merged_nodes[i] = min_rep
                    merged_nodes[j] = min_rep
                    merged_nodes[max_rep] = min_rep
    
    # Second pass: add edges using merged node mapping
    for (i, j), values in preference_values.items():
        for value in values:
            pref = transform_digit_to_preference(data_type, value)
            
            # Get representative nodes
            node_i = merged_nodes.get(i, i)
            node_j = merged_nodes.get(j, j)
            
            # Skip self-loops (merged nodes)
            if node_i == node_j:
                continue
                
            if pref > 0.5:  # i is preferred over j
                G.add_edge(node_i, node_j)
            elif pref < 0.5:  # j is preferred over i
                G.add_edge(node_j, node_i)
    
    return G


def count_cycles(G):
    """
    Count the number of simple cycles in a directed graph.
    Returns the total number of cycles.
    """
    try:
        cycles = list(nx.simple_cycles(G))
        return len(cycles), cycles
    except Exception as e:
        print(f"Error counting cycles: {e}")
        return 0, []


def find_dominant_responses_for_single_judgment(preference_values, judgment_idx, n_response, data_type):
    """
    Find responses that are dominant (better than or tied to all others) for a single judgment.
    A response i is dominant if for all other responses j, response i is preferred over j or tied with j.
    
    Returns:
        dominant_responses: list of response indices that are dominant
        preference_matrix: n_response x n_response matrix showing pairwise preferences
    """
    # Initialize preference matrix with None (no comparison available)
    preference_matrix = [[None for _ in range(n_response)] for _ in range(n_response)]
    
    # Fill diagonal with ties (response compared to itself)
    for i in range(n_response):
        preference_matrix[i][i] = 0.5
    
    # Fill preference matrix from pairwise judgments
    for (i, j), values in preference_values.items():
        if judgment_idx < len(values):
            value = values[judgment_idx]
            pref = transform_digit_to_preference(data_type, value)
            
            # Store preference from i's perspective towards j
            preference_matrix[i][j] = pref
            # Store reverse preference from j's perspective towards i
            preference_matrix[j][i] = 1 - pref
    
    # Find dominant responses
    dominant_responses = []
    
    for i in range(n_response):
        is_dominant = True
        for j in range(n_response):
            if i != j and preference_matrix[i][j] is not None:
                # Response i is dominant over j if pref >= 0.5 (preferred or tied)
                if preference_matrix[i][j] < 0.5:
                    is_dominant = False
                    break
        
        if is_dominant:
            dominant_responses.append(i)
    
    return dominant_responses, preference_matrix


def calculate_intransitivity_for_single_judgment(preference_values, judgment_idx, n_response, data_type):
    """
    Calculate intransitivity for a single judgment (at index judgment_idx) by building a directed graph
    and counting cycles. When pref=0.5 (tie), merge nodes i and j.
    """
    G = nx.DiGraph()
    
    # Add all nodes
    for i in range(n_response):
        G.add_node(i)
    
    # Track nodes that should be merged (tied nodes)
    merged_nodes = {}  # maps original node -> representative node
    
    # First pass: identify tied nodes and create merge mapping
    for (i, j), values in preference_values.items():
        if judgment_idx < len(values):
            value = values[judgment_idx]
            pref = transform_digit_to_preference(data_type, value)
            
            if pref == 0.5:  # tie case - merge nodes
                # Find representatives for both nodes
                rep_i = merged_nodes.get(i, i)
                rep_j = merged_nodes.get(j, j)
                
                # Merge to the smaller representative (for consistency)
                if rep_i != rep_j:
                    min_rep = min(rep_i, rep_j)
                    max_rep = max(rep_i, rep_j)
                    
                    # Update all nodes that point to max_rep to point to min_rep
                    for node in list(merged_nodes.keys()):
                        if merged_nodes[node] == max_rep:
                            merged_nodes[node] = min_rep
                    
                    # Update the mapping for the current nodes
                    merged_nodes[i] = min_rep
                    merged_nodes[j] = min_rep
                    merged_nodes[max_rep] = min_rep
    
    # Second pass: add edges using merged node mapping
    for (i, j), values in preference_values.items():
        if judgment_idx < len(values):
            value = values[judgment_idx]
            pref = transform_digit_to_preference(data_type, value)
            
            # Get representative nodes
            node_i = merged_nodes.get(i, i)
            node_j = merged_nodes.get(j, j)
            
            # Skip self-loops (merged nodes)
            # if node_i == node_j:
            #     continue
            
            if pref > 0.5:  # i is preferred over j
                G.add_edge(node_i, node_j)
            elif pref < 0.5:  # j is preferred over i
                G.add_edge(node_j, node_i)
    
    # Count cycles
    cycle_count, cycles = count_cycles(G)
    return cycle_count, cycles, G, merged_nodes


def analyze_for_n_responses(preference_data, n_response, data_type, verbose=True):
    """
    Run intransitivity and dominance analysis for a specific number of responses.
    
    Returns:
        dict: Analysis results including intransitivity and dominance statistics
    """
    # Filter preference data to only include pairs within n_response range
    filtered_preference_data = {}
    for (i, j), values in preference_data.items():
        if i < n_response and j < n_response:
            filtered_preference_data[(i, j)] = values
    
    if not filtered_preference_data:
        return None
    
    # Find the maximum number of judgments across all pairs
    max_judgments = max(len(values) for values in filtered_preference_data.values())
    
    # Calculate intransitivity and dominance for each individual judgment
    intransitivity_counts = []
    total_cycles = 0
    dominance_data = []
    
    for judgment_idx in range(max_judgments):
        # Check if this judgment index exists for all pairs
        valid_judgment = all(judgment_idx < len(values) for values in filtered_preference_data.values())
        
        if valid_judgment:
            # Calculate intransitivity (cycles)
            cycle_count, cycles, graph, merged_nodes = calculate_intransitivity_for_single_judgment(
                filtered_preference_data, judgment_idx, n_response, data_type
            )
            intransitivity_counts.append(cycle_count)
            total_cycles += cycle_count
            
            # Find dominant responses
            dominant_responses, preference_matrix = find_dominant_responses_for_single_judgment(
                filtered_preference_data, judgment_idx, n_response, data_type
            )
            dominance_data.append(dominant_responses)
    
    # Calculate statistics
    results = {
        'n_response': n_response,
        'total_judgments': len(intransitivity_counts),
        'total_cycles': total_cycles,
        'avg_cycles': np.mean(intransitivity_counts) if intransitivity_counts else 0,
        'judgments_with_cycles': np.sum(np.array(intransitivity_counts) > 0) if intransitivity_counts else 0,
        'intransitivity_percentage': 100 * np.mean(np.array(intransitivity_counts) > 0) if intransitivity_counts else 0,
    }
    
    if dominance_data:
        judgments_with_dominant = [len(dom_list) for dom_list in dominance_data]
        judgments_with_no_dominant = sum(1 for count in judgments_with_dominant if count == 0)
        judgments_with_one_dominant = sum(1 for count in judgments_with_dominant if count == 1)
        judgments_with_multiple_dominant = sum(1 for count in judgments_with_dominant if count > 1)
        
        results.update({
            'judgments_with_no_dominant': judgments_with_no_dominant,
            'judgments_with_one_dominant': judgments_with_one_dominant,
            'judgments_with_multiple_dominant': judgments_with_multiple_dominant,
            'no_dominant_percentage': 100 * judgments_with_no_dominant / len(dominance_data),
            'one_dominant_percentage': 100 * judgments_with_one_dominant / len(dominance_data),
            'multiple_dominant_percentage': 100 * judgments_with_multiple_dominant / len(dominance_data),
            'avg_dominant': np.mean(judgments_with_dominant)
        })
    
    if verbose:
        print(f"\nResults for {n_response} responses:")
        print(f"Total judgments processed: {results['total_judgments']}")
        print(f"Total cycles found: {results['total_cycles']}")
        print(f"Average cycles per judgment: {results['avg_cycles']:.4f}")
        print(f"Judgments with cycles: {results['judgments_with_cycles']}")
        print(f"Percentage of judgments with intransitivity: {results['intransitivity_percentage']:.2f}%")
        
        if dominance_data:
            print(f"\n--- Condorcet Winner Analysis ---")
            print(f"Judgments with no Condorcet winner: {results['judgments_with_no_dominant']} ({results['no_dominant_percentage']:.2f}%)")
            print(f"Judgments with exactly one Condorcet winner: {results['judgments_with_one_dominant']} ({results['one_dominant_percentage']:.2f}%)")
            print(f"Judgments with multiple Condorcet winners: {results['judgments_with_multiple_dominant']} ({results['multiple_dominant_percentage']:.2f}%)")
            print(f"Average number of Condorcet winners per judgment: {results['avg_dominant']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Check intransitivity using directed graphs')
    parser.add_argument('--input_repo', type=str, required=True, 
                       help='HuggingFace repository name to load dataset from')
    parser.add_argument('--type', type=str, required=True, choices=['5score', 'ternary'],
                       help='Type of preference data (5score or ternary)')
    parser.add_argument('--n_response', type=int, default=3,
                       help='Maximum number of responses to compare. When --plot is used, analyzes from 2 to n_response (default: 3)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plot showing trends vs number of responses (analyzes from 2 to n_response)')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.input_repo}")
    print(f"Number of responses: {args.n_response}")
    print(f"Data type: {args.type}")
    
    dataset = load_dataset(args.input_repo)
    
    # Use the train split by default, adjust if needed
    if 'train' in dataset:
        dataset = dataset['train']
    else:
        # Use the first available split
        dataset = dataset[list(dataset.keys())[0]]
    
    print("Processing dataset...")
    # Process dataset with the requested number of responses
    preference_data = process_dataset(dataset, args.n_response)
    
    if args.plot:
        print(f"Running analysis for n_response from 2 to {args.n_response}...")
        
        # Collect results for different numbers of responses
        plot_results = []
        
        for n_resp in range(2, args.n_response + 1):
            print(f"\nAnalyzing with {n_resp} responses...")
            result = analyze_for_n_responses(preference_data, n_resp, args.type, verbose=False)
            if result:
                plot_results.append(result)
                print(f"  {n_resp} responses: {result['no_dominant_percentage']:.1f}% no Condorcet winner, {result['intransitivity_percentage']:.1f}% intransitive")
        
        if plot_results:
            # Extract data for plotting
            n_responses = [r['n_response'] for r in plot_results]
            no_dominant_percentages = [r['no_dominant_percentage'] for r in plot_results]
            intransitivity_percentages = [r['intransitivity_percentage'] for r in plot_results]
            
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot no-dominant percentage
            color = 'tab:red'
            ax1.set_xlabel('Number of Responses')
            ax1.set_ylabel('No Condorcet Winner (%)', color=color)
            line1 = ax1.plot(n_responses, no_dominant_percentages, 'o-', color=color, linewidth=2, markersize=6, label='No Condorcet Winner')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for intransitivity percentage
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Intransitivity (%)', color=color)
            line2 = ax2.plot(n_responses, intransitivity_percentages, 's-', color=color, linewidth=2, markersize=6, label='Intransitivity')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            # Set title and format
            plt.title('No Condorcet Winner vs Intransitivity by Number of Responses', fontsize=14, pad=20)
            ax1.set_xticks(n_responses)
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"dominance_intransitivity_plot_{args.type}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved as: {plot_filename}")
            
            # Show the plot
            plt.show()
            
            # Print summary table
            print(f"\n{'N_Resp':<8}{'No-CW %':<10}{'Intrans %':<12}{'Judgments':<12}")
            print("-" * 42)
            for r in plot_results:
                print(f"{r['n_response']:<8}{r['no_dominant_percentage']:<10.1f}{r['intransitivity_percentage']:<12.1f}{r['total_judgments']:<12}")
        else:
            print("No valid results found for plotting")
    
    else:
        # Run single analysis as before
        result = analyze_for_n_responses(preference_data, args.n_response, args.type, verbose=True)
        
        if result and 'avg_cycles' in result:
            print(f"Max cycles in a single judgment: {result.get('max_cycles', 'N/A')}")
            print(f"Standard deviation of cycle counts: {result.get('std_cycles', 'N/A')}")


if __name__ == "__main__":
    main()
