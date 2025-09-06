import argparse
import numpy as np
import networkx as nx
from datasets import load_dataset
from itertools import combinations


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


def main():
    parser = argparse.ArgumentParser(description='Check intransitivity using directed graphs')
    parser.add_argument('--input_repo', type=str, required=True, 
                       help='HuggingFace repository name to load dataset from')
    parser.add_argument('--type', type=str, required=True, choices=['5score', 'ternary'],
                       help='Type of preference data (5score or ternary)')
    parser.add_argument('--n_response', type=int, default=3,
                       help='Number of responses to compare (default: 3)')
    
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
    preference_data = process_dataset(dataset, args.n_response)
    
    # Find the maximum number of judgments across all pairs
    max_judgments = max(len(values) for values in preference_data.values()) if preference_data else 0
    print(f"Maximum number of judgments per pair: {max_judgments}")
    
    # Calculate intransitivity for each individual judgment
    intransitivity_counts = []
    total_cycles = 0
    
    for judgment_idx in range(max_judgments):
        # Check if this judgment index exists for all pairs
        valid_judgment = all(judgment_idx < len(values) for values in preference_data.values())
        
        if valid_judgment:
            cycle_count, cycles, graph, merged_nodes = calculate_intransitivity_for_single_judgment(
                preference_data, judgment_idx, args.n_response, args.type
            )
            intransitivity_counts.append(cycle_count)
            total_cycles += cycle_count
            
            # if judgment_idx < 30:  # Print first few examples for debugging
            #     print(f"\n--- Judgment {judgment_idx} ---")
            #     print(f"Raw preference values for this judgment:")
            #     for (i, j), values in preference_data.items():
            #         if judgment_idx < len(values):
            #             raw_value = values[judgment_idx]
            #             pref_value = transform_digit_to_preference(args.type, raw_value)
            #             print(f"  Pair ({i},{j}): raw={raw_value}, pref={pref_value}")
                
            #     print(f"Node merging mapping: {merged_nodes}")
            #     print(f"Graph nodes: {list(graph.nodes())}")
            #     print(f"Graph edges: {list(graph.edges())}")
            #     print(f"Cycle count: {cycle_count}")
            #     if cycles:
            #         print(f"Cycles found: {cycles}")
            #     else:
            #         print("No cycles found")
        else:
            print(f"Judgment {judgment_idx}: Skipped (not all pairs have this judgment)")
    
    print(f"\nResults:")
    print(f"Total judgments processed: {len(intransitivity_counts)}")
    print(f"Total cycles found: {total_cycles}")
    print(f"Average cycles per judgment: {np.mean(intransitivity_counts):.4f}")
    print(f"Judgments with cycles: {np.sum(np.array(intransitivity_counts) > 0)}")
    print(f"Percentage of judgments with intransitivity: {100 * np.mean(np.array(intransitivity_counts) > 0):.2f}%")
    
    # Additional statistics
    if intransitivity_counts:
        print(f"Max cycles in a single judgment: {max(intransitivity_counts)}")
        print(f"Standard deviation of cycle counts: {np.std(intransitivity_counts):.4f}")


if __name__ == "__main__":
    main()
