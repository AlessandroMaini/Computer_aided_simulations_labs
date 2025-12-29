import random
import math
import json
import networkx as nx
from tqdm import tqdm
from networkx.algorithms import approximation
from gnp import fast_gnp_random_graph

def majority_model(G, initial_opinion_prob, max_time=1000):
    """
    Simulate the majority model on graph G.
    
    Parameters:
    - G: A NetworkX graph instance
    - initial_opinion_prob: Probability of initializing a node with +1 opinion
    - max_time: Maximum duration of the simulation
    
    Returns:
    - consensus_reached: Boolean indicating if consensus was reached
    - time: Time needed to reach consensus or max_time if not reached
    """
    # Step 1: Initialize opinions
    opinions = {node: (1 if random.random() < initial_opinion_prob else -1) for node in G.nodes()}
    
    # Every node v wakes up according to a Poisson process with rate 1
    time = 0
    total_rate = len(G)  # Since each node has rate 1
    while time < max_time:
        interval = random.expovariate(total_rate)
        time += interval
        # Select a random node to update uniformly
        v = random.choice(list(G.nodes()))
        
        # Get the opinions of neighbors
        neighbor_opinions = [opinions[neighbor] for neighbor in G.neighbors(v)]
        
        if not neighbor_opinions:
            continue  # No neighbors to influence
        
        # Determine the majority opinion among neighbors
        if sum(neighbor_opinions) > 0:
            majority_opinion = 1
        elif sum(neighbor_opinions) < 0:
            majority_opinion = -1
        else:
            majority_opinion = random.choice([-1, 1])  # Tie-breaker
        
        # Update the opinion of node v
        opinions[v] = majority_opinion
        
        # Check for consensus
        if all(op == 1 for op in opinions.values()) or all(op == -1 for op in opinions.values()):
            return True, time

def run_majority_model_experiments(n, p, initial_opinion_probs, seed=42, output_file="majority_model_results.json"):
    """
    Run majority model experiments on G(n, p) graphs for given initial opinion probabilities.
    
    Parameters:
    - n: Number of nodes
    - p: Probability of edge creation
    - initial_opinion_probs: List of initial opinion probabilities to test
    - seed: Random seed for reproducibility
    - output_file: Path to save JSON results
    
    Returns:
    - Dictionary containing all experiment results
    """
    random.seed(seed)
    results = []
    
    for prob in initial_opinion_probs:
        print(f"Running majority model for initial opinion probability {prob}")
        consensus_count = 0
        total_time = 0
        
        for _ in tqdm(range(10)):
            G = fast_gnp_random_graph(n, p)
            consensus_reached, time = majority_model(G, prob)
            if consensus_reached:
                consensus_count += 1
            total_time += time
        
        avg_time = total_time / 10
        result_entry = {
            "n": n,
            "p": p,
            "initial_opinion_prob": prob,
            "consensus_probability": consensus_count / 10,
            "average_time": avg_time
        }
        results.append(result_entry)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    n = 10000
    p = 3 * math.log(n) / n
    initial_opinion_probs = [0.5, 0.6, 0.7, 0.8]
    
    run_majority_model_experiments(n, p, initial_opinion_probs, seed=42, output_file="majority_model_results.json")