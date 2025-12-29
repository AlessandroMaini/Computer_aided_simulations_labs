import networkx as nx
import random
from tqdm import tqdm
import json
import math
from networkx.algorithms import approximation

def gnr_random_graph(n, r):
    """
    Generates a G(n, r) random geometric graph.
    
    Parameters:
    - n: Number of nodes
    - r: Distance threshold for edge creation
    
    Returns:
    - G: A NetworkX graph instance representing the G(n, r) graph
    """
    # Step 1: Place n nodes uniformly in the unit square
    positions = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n)}
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Step 2: Add edges based on distance threshold r
    # Use spatial indexing for better performance with large n
    cell_size = r
    grid = {}
    
    # Assign nodes to grid cells
    for i in range(n):
        cell_x = int(positions[i][0] / cell_size)
        cell_y = int(positions[i][1] / cell_size)
        if (cell_x, cell_y) not in grid:
            grid[(cell_x, cell_y)] = []
        grid[(cell_x, cell_y)].append(i)
    
    # Check only nodes in neighboring cells
    for i in range(n):
        cell_x = int(positions[i][0] / cell_size)
        cell_y = int(positions[i][1] / cell_size)
        
        # Check current and adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in grid:
                    for j in grid[neighbor_cell]:
                        if i < j:
                            dist = ((positions[i][0] - positions[j][0]) ** 2 + 
                                    (positions[i][1] - positions[j][1]) ** 2) ** 0.5
                            if dist < r:
                                G.add_edge(i, j)
    
    return G

def run_gnr_experiments(n_list, a_list, seed=42, output_file="gnr_results.json"):
    """
    Run G(n, r) graph experiments for given lists of n and a values.
    
    Parameters:
    - n_list: List of node counts to test
    - a_list: List of 'a' values where r = sqrt(a * log(n) / (pi * n))
    - seed: Random seed for reproducibility
    - output_file: Path to save JSON results
    
    Returns:
    - Dictionary containing all experiment results
    """
    random.seed(seed)
    results = []
    
    for ni in n_list:
        r_list = [math.sqrt(ai * math.log(ni) / (math.pi * ni)) for ai in a_list] 
        for ai, ri in zip(a_list, r_list):
            print(f"Generating 10 instances of G({ni}, {ri})")
            tot_nodes = 0
            tot_edges = 0
            n_connected = 0
            tot_diameter = 0
            tot_largest_cc = 0
            tot_second_cc = 0
            
            for _ in tqdm(range(10)):
                sparse_graph = gnr_random_graph(ni, ri)
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
                "r": ri,
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
    n = [1000, 10000, 100000, 1000000]
    a = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    
    run_gnr_experiments(n, a, seed=42, output_file="gnr_results.json")