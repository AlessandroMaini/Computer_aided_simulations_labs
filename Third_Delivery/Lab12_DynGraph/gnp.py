import random
import math
import json
import networkx as nx
from tqdm import tqdm
from networkx.algorithms import approximation

def fast_gnp_random_graph(n, p):
    """
    Generates a G(n, p) graph in O(n) time.

    Parameters:
    - n: Number of nodes
    - p: Probability of edge creation

    Returns:
    - G: A NetworkX graph instance representing the G(n, p) graph
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n)

    # We start with the first possible edge (0, 1)
    v = 1
    w = -1
    
    while v < n:
        # Step 2: Calculate skip distance 's' from Geometric Distribution
        # random.random() gives r in (0, 1]
        r = random.random()
        # To avoid log(0)
        while r == 0:
            r = random.random()
            
        s = math.ceil(math.log(r) / math.log(1.0 - p))
        
        # Step 3 & 4: Advance indices and handle row wrapping
        w = w + s
        while w >= v and v < n:
            w = w - v
            v = v + 1
        
        # Step 5: Add the edge
        if v < n:
            G.add_edge(v, w)
            
    return G


def run_gnp_experiments(n_list, a_list, p_type, seed=42, output_file="gnp_results.json"):
    """
    Run G(n, p) graph experiments for given lists of n and a values.
    
    Parameters:
    - n_list: List of node counts to test
    - a_list: List of 'a' values where p = a / n or p = a * log(n) / n
    - p_type: 'linear' for p = a/n, 'logarithmic' for p = a*log(n)/n
    - seed: Random seed for reproducibility
    - output_file: Path to save JSON results
    
    Returns:
    - Dictionary containing all experiment results
    """
    random.seed(seed)
    results = []
    
    for ni in n_list:
        if p_type == 'linear':
            p_list = [ai / ni for ai in a_list]
        elif p_type == 'logarithmic':
            p_list = [ai * math.log(ni) / ni for ai in a_list]
        else:
            raise ValueError("p_type must be either 'linear' or 'logarithmic'")
        for ai, pi in zip(a_list, p_list):
            print(f"Generating 10 instances of G({ni}, {pi})")
            tot_nodes = 0
            tot_edges = 0
            n_connected = 0
            tot_diameter = 0
            tot_largest_cc = 0
            tot_second_cc = 0
            
            for _ in tqdm(range(10)):
                sparse_graph = fast_gnp_random_graph(ni, pi)
                tot_nodes += sparse_graph.number_of_nodes()
                tot_edges += sparse_graph.number_of_edges()
                # Check if the graph is connected
                if nx.is_connected(sparse_graph):
                    # Compute the diameter
                    tot_diameter += approximation.diameter(sparse_graph)
                    n_connected += 1
                    tot_largest_cc += sparse_graph.number_of_nodes()
                else:
                    # Compute the size of the largest and second largest connected components
                    components = sorted(nx.connected_components(sparse_graph), key=len, reverse=True)
                    tot_largest_cc += len(components[0])
                    tot_second_cc += len(components[1])

            # Calculate averages
            avg_nodes = tot_nodes / 10
            avg_edges = tot_edges / 10
            avg_diameter = tot_diameter / n_connected if n_connected > 0 else None
            avg_largest_cc = tot_largest_cc / 10 if 10 > 0 else 0
            avg_second_cc = tot_second_cc / (10 - n_connected) if (10 - n_connected) > 0 else 0
            
            # Print results
            print(f"   Average Nodes: {avg_nodes}")
            print(f"   Average Edges: {avg_edges}")
            print(f"   Connected Instances: {n_connected} out of 10")
            if n_connected > 0:
                print(f"   Average Diameter (connected graphs): {avg_diameter}")
            print(f"   Average Largest CC Size (disconnected graphs): {avg_largest_cc}")
            if n_connected < 10:
                print(f"   Average Second Largest CC Size (disconnected graphs): {avg_second_cc}")
            print("--------------------------------------------------")
            
            # Store results
            result_entry = {
                "n": ni,
                "a": ai,
                "p": pi,
                "avg_nodes": avg_nodes,
                "avg_edges": avg_edges,
                "connected_instances": n_connected,
                "avg_diameter": avg_diameter,
                "avg_largest_cc": avg_largest_cc,
                "avg_second_cc": avg_second_cc if n_connected < 10 else None
            }
            results.append(result_entry)
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    # Example usage
    n = [10000, 100000, 1000000]
    a = [0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2]
    
    run_gnp_experiments(n, a, p_type='linear', seed=42, output_file="gnp_linear_results.json")
    run_gnp_experiments(n, a, p_type='logarithmic', seed=42, output_file="gnp_logarithmic_results.json")