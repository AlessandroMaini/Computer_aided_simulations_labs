import random
import math
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from gnp import fast_gnp_random_graph
import numpy as np

def majority_model_with_tracking(G, initial_opinion_prob, max_time=1000, sample_interval=0.1):
    """
    Simulate the majority model on graph G and track opinion counts over time.
    
    Parameters:
    - G: A NetworkX graph instance
    - initial_opinion_prob: Probability of initializing a node with +1 opinion
    - max_time: Maximum duration of the simulation
    - sample_interval: Time interval for recording opinion counts
    
    Returns:
    - times: List of time points
    - positive_counts: List of counts for +1 opinion at each time point
    - negative_counts: List of counts for -1 opinion at each time point
    - consensus_reached: Boolean indicating if consensus was reached
    - consensus: +1 if consensus on +1, -1 if consensus on -1, 0 otherwise
    - final_time: Time when consensus was reached or max_time
    """
    # Initialize opinions
    opinions = {node: (1 if random.random() < initial_opinion_prob else -1) for node in G.nodes()}
    
    # Track opinion counts over time
    times = [0]
    positive_counts = [sum(1 for op in opinions.values() if op == 1)]
    negative_counts = [sum(1 for op in opinions.values() if op == -1)]
    
    time = 0
    next_sample_time = sample_interval
    total_rate = len(G)
    consensus_reached = False
    consensus = None
    
    while time < max_time:
        interval = random.expovariate(total_rate)
        time += interval
        
        # Select a random node to update
        v = random.choice(list(G.nodes()))
        
        # Get the opinions of neighbors
        neighbor_opinions = [opinions[neighbor] for neighbor in G.neighbors(v)]
        
        if not neighbor_opinions:
            continue
        
        # Determine the majority opinion among neighbors
        if sum(neighbor_opinions) > 0:
            majority_opinion = 1
        elif sum(neighbor_opinions) < 0:
            majority_opinion = -1
        else:
            majority_opinion = random.choice([-1, 1])
        
        # Update the opinion of node v
        opinions[v] = majority_opinion
        
        # Record opinion counts at sample intervals
        while next_sample_time <= time and next_sample_time <= max_time:
            times.append(next_sample_time)
            positive_counts.append(sum(1 for op in opinions.values() if op == 1))
            negative_counts.append(sum(1 for op in opinions.values() if op == -1))
            next_sample_time += sample_interval
        
        # Check for consensus
        if all(op == 1 for op in opinions.values()) or all(op == -1 for op in opinions.values()):
            consensus_reached = True
            consensus = 1 if opinions[v] == 1 else -1
            # Record final state
            if times[-1] != time:
                times.append(time)
                positive_counts.append(sum(1 for op in opinions.values() if op == 1))
                negative_counts.append(sum(1 for op in opinions.values() if op == -1))
            break
    
    return times, positive_counts, negative_counts, consensus_reached, consensus, time

def plot_opinion_dynamics(times, positive_counts, negative_counts, title="Opinion Dynamics Over Time", save_path=None, 
                         all_times=None, all_positive_counts=None, all_negative_counts=None):
    """
    Plot the number of nodes with each opinion over time.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual replications with transparency if provided
    if all_times is not None:
        for t, pos, neg in zip(all_times, all_positive_counts, all_negative_counts):
            plt.plot(t, pos, color='blue', alpha=0.2, linewidth=1)
            plt.plot(t, neg, color='red', alpha=0.2, linewidth=1)
    
    # Plot average with full opacity
    plt.plot(times, positive_counts, label='Opinion +1', color='blue', linewidth=2)
    plt.plot(times, negative_counts, label='Opinion -1', color='red', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def compute_average_time_series(all_times, all_positive_counts, all_negative_counts, num_points=1000):
    """
    Compute the average time series across multiple replications.
    """    
    # Find the maximum time across all replications
    max_time = max(times[-1] for times in all_times)
    
    # Create a common time grid
    avg_times = np.linspace(0, max_time, num_points)
    
    # Interpolate each replication onto the common time grid
    interpolated_positive = []
    interpolated_negative = []
    
    for times, pos_counts, neg_counts in zip(all_times, all_positive_counts, all_negative_counts):
        # Interpolate onto the common time grid
        interp_pos = np.interp(avg_times, times, pos_counts)
        interp_neg = np.interp(avg_times, times, neg_counts)
        interpolated_positive.append(interp_pos)
        interpolated_negative.append(interp_neg)
    
    # Convert to numpy arrays for easier computation
    interpolated_positive = np.array(interpolated_positive)
    interpolated_negative = np.array(interpolated_negative)
    
    # Compute mean and standard deviation
    avg_positive_counts = np.mean(interpolated_positive, axis=0)
    avg_negative_counts = np.mean(interpolated_negative, axis=0)
    
    return avg_times, avg_positive_counts, avg_negative_counts

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
        consensus_positive_count = 0
        total_time = 0
        
        # Data structures to store time series for all replications
        all_times = []
        all_positive_counts = []
        all_negative_counts = []
        
        for _ in tqdm(range(10)):
            G = fast_gnp_random_graph(n, p)
            times, pos_counts, neg_counts, consensus_reached, consensus, time = majority_model_with_tracking(G, prob)
            
            # Store the time series data
            all_times.append(times)
            all_positive_counts.append(pos_counts)
            all_negative_counts.append(neg_counts)
            
            if consensus_reached:
                consensus_count += 1
                if consensus == 1:
                    consensus_positive_count += 1
            total_time += time
        
        # Compute average time series across replications
        avg_times, avg_pos_counts, avg_neg_counts = compute_average_time_series(
            all_times, all_positive_counts, all_negative_counts
        )
        
        avg_time = total_time / 10
        result_entry = {
            "n": n,
            "p": p,
            "initial_opinion_prob": prob,
            "consensus_probability": consensus_count / 10,
            "consensus_positive_probability": consensus_positive_count / consensus_count if consensus_count > 0 else 0,
            "average_time": avg_time
        }
        results.append(result_entry)

        # Print a summary of results
        print(f"   Consensus Probability: {result_entry['consensus_probability']}")
        print(f"   Consensus Positive Probability: {result_entry['consensus_positive_probability']}")
        
        # Plot the average across replications with individual trajectories
        plot_opinion_dynamics(avg_times.tolist(), avg_pos_counts.tolist(), avg_neg_counts.tolist(), 
                              title=f"Average Majority Model Dynamics (p={prob:.3f}, n={n}, 10 reps)",
                              save_path=f"majority_model_avg_dynamics_n{n}_prob{prob}.png",
                              all_times=all_times, all_positive_counts=all_positive_counts, 
                              all_negative_counts=all_negative_counts)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    n = 10000
    p = 3 * math.log(n) / n
    initial_opinion_probs = [0.5, 0.6, 0.7, 0.8]
    
    run_majority_model_experiments(n, p, initial_opinion_probs, seed=42, output_file="majority_model_results.json")