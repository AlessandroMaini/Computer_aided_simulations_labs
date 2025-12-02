"""
Parameter Sensitivity Analysis for Carnivore-Herbivore CTMC Model
Systematically tests parameter variations to assess stability and ergodicity.
"""

import numpy as np
import matplotlib.pyplot as plt
from carnivore_herbivore import SimulationEngine
from typing import Dict, List, Tuple
import pandas as pd

def run_parameter_sweep(base_params: Dict[str, float], 
                       param_name: str, 
                       param_values: List[float],
                       initial_pop: Dict[str, int],
                       t_max: float = 500.0,
                       num_runs: int = 10) -> Dict:
    """
    Run simulations sweeping one parameter while keeping others constant.
    
    Returns dictionary with metrics for each parameter value.
    """
    results = {
        'param_values': [],
        'avg_H_mean': [],
        'avg_H_std': [],
        'avg_C_mean': [],
        'avg_C_std': [],
        'extinction_rate': [],
        'cv_H': [],  # Coefficient of variation
        'cv_C': [],
        'HC_ratio_mean': [],
        'HC_ratio_std': []
    }
    
    for param_val in param_values:
        print(f"Testing {param_name}={param_val:.4f}...")
        
        # Create modified parameters
        test_params = base_params.copy()
        test_params[param_name] = param_val
        
        H_values = []
        C_values = []
        HC_ratios = []
        extinctions = 0
        
        for run in range(num_runs):
            np.random.seed(1000 + run)
            
            sim = SimulationEngine(t_max=t_max, initial_pop=initial_pop, params=test_params)
            sim.event_loop()
            
            total_time = sim.current_time
            if total_time > 0:
                avg_H = (sim.metrics.area_under_herbivore_males + 
                        sim.metrics.area_under_herbivore_females + 
                        sim.metrics.area_under_herbivore_females_preg) / total_time
                avg_C = (sim.metrics.area_under_carnivore_males + 
                        sim.metrics.area_under_carnivore_females + 
                        sim.metrics.area_under_carnivore_females_preg) / total_time
                
                # Check for extinction (very low populations)
                if avg_H < 1.0 or avg_C < 0.5:
                    extinctions += 1
                    
                H_values.append(avg_H)
                C_values.append(avg_C)
                if avg_C > 0:
                    HC_ratios.append(avg_H / avg_C)
        
        # Compute statistics
        results['param_values'].append(param_val)
        results['avg_H_mean'].append(np.mean(H_values) if H_values else 0)
        results['avg_H_std'].append(np.std(H_values) if H_values else 0)
        results['avg_C_mean'].append(np.mean(C_values) if C_values else 0)
        results['avg_C_std'].append(np.std(C_values) if C_values else 0)
        results['extinction_rate'].append(extinctions / num_runs)
        
        # Coefficient of variation (std/mean) - lower is more stable
        H_mean = np.mean(H_values) if H_values else 0
        C_mean = np.mean(C_values) if C_values else 0
        results['cv_H'].append(np.std(H_values) / H_mean if H_mean > 0 else np.inf)
        results['cv_C'].append(np.std(C_values) / C_mean if C_mean > 0 else np.inf)
        
        results['HC_ratio_mean'].append(np.mean(HC_ratios) if HC_ratios else 0)
        results['HC_ratio_std'].append(np.std(HC_ratios) if HC_ratios else 0)
    
    return results

def plot_sensitivity_results(results: Dict, param_name: str, param_label: str):
    """Plot sensitivity analysis results for one parameter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    param_vals = results['param_values']
    
    # 1. Population means with error bars
    ax = axes[0, 0]
    ax.errorbar(param_vals, results['avg_H_mean'], yerr=results['avg_H_std'], 
                label='Herbivores', marker='o', capsize=5, color='green')
    ax.errorbar(param_vals, results['avg_C_mean'], yerr=results['avg_C_std'], 
                label='Carnivores', marker='s', capsize=5, color='red')
    ax.set_xlabel(param_label)
    ax.set_ylabel('Average Population')
    ax.set_title(f'Population vs {param_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Coefficient of Variation (stability measure)
    ax = axes[0, 1]
    ax.plot(param_vals, results['cv_H'], marker='o', label='CV Herbivores', color='green')
    ax.plot(param_vals, results['cv_C'], marker='s', label='CV Carnivores', color='red')
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='CV=0.1 (good stability)')
    ax.set_xlabel(param_label)
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title(f'Stability (CV) vs {param_label}')
    ax.set_ylim(0, min(1.0, max(results['cv_H'] + results['cv_C']) * 1.1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Extinction rate
    ax = axes[1, 0]
    ax.plot(param_vals, results['extinction_rate'], marker='D', color='black', linewidth=2)
    ax.fill_between(param_vals, 0, results['extinction_rate'], alpha=0.3, color='red')
    ax.set_xlabel(param_label)
    ax.set_ylabel('Extinction Rate')
    ax.set_title(f'Extinction Risk vs {param_label}')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    # 4. H/C Ratio
    ax = axes[1, 1]
    ax.errorbar(param_vals, results['HC_ratio_mean'], yerr=results['HC_ratio_std'], 
                marker='o', capsize=5, color='blue')
    ax.axhline(y=3.0, color='orange', linestyle='--', label='Target Ratio=3.0', linewidth=2)
    ax.set_xlabel(param_label)
    ax.set_ylabel('H/C Ratio')
    ax.set_title(f'Predator-Prey Ratio vs {param_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'sensitivity_{param_name}.png', dpi=150)
    print(f"Saved plot: sensitivity_{param_name}.png")
    plt.show()

def find_optimal_parameters(base_params: Dict[str, float], 
                           initial_pop: Dict[str, int]) -> Dict[str, float]:
    """
    Systematically search for stable parameter configuration.
    """
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80 + "\n")
    
    # Define parameter ranges to test
    param_configs = {
        'r_H': (np.linspace(0.3, 0.7, 9), 'Herbivore Conception Rate'),
        'r_C': (np.linspace(0.2, 0.6, 9), 'Carnivore Conception Rate'),
        'gestation_H': (np.linspace(4.0, 12.0, 9), 'Herbivore Gestation Period'),
        'gestation_C': (np.linspace(3.0, 9.0, 9), 'Carnivore Gestation Period'),
        'm_H': (np.linspace(0.01, 0.05, 9), 'Herbivore Mortality Rate'),
        'm_C': (np.linspace(0.015, 0.06, 9), 'Carnivore Mortality Rate'),
        'pred': (np.linspace(0.0005, 0.002, 9), 'Predation Rate'),
        'conflict_H': (np.linspace(0.00001, 0.0002, 9), 'Herbivore Conflict Rate'),
        'conflict_C': (np.linspace(0.0002, 0.001, 9), 'Carnivore Conflict Rate'),
    }
    
    all_results = {}
    
    for param_name, (values, label) in param_configs.items():
        print(f"\n--- Analyzing {label} ({param_name}) ---")
        results = run_parameter_sweep(base_params, param_name, values, 
                                      initial_pop, t_max=500.0, num_runs=10)
        all_results[param_name] = results
        
        # Plot results
        plot_sensitivity_results(results, param_name, label)
        
        # Find best value (lowest CV, no extinctions, ratio near 3.0)
        scores = []
        for i in range(len(values)):
            if results['extinction_rate'][i] > 0.2:  # Too risky
                scores.append(np.inf)
            else:
                # Score combines CV (want low), ratio deviation (want near 3.0)
                cv_score = (results['cv_H'][i] + results['cv_C'][i]) / 2
                ratio_score = abs(results['HC_ratio_mean'][i] - 3.0)
                scores.append(cv_score + 0.5 * ratio_score)
        
        best_idx = np.argmin(scores)
        best_val = values[best_idx]
        print(f"  Best {param_name}: {best_val:.4f}")
        print(f"    - H population: {results['avg_H_mean'][best_idx]:.2f} ± {results['avg_H_std'][best_idx]:.2f}")
        print(f"    - C population: {results['avg_C_mean'][best_idx]:.2f} ± {results['avg_C_std'][best_idx]:.2f}")
        print(f"    - CV_H: {results['cv_H'][best_idx]:.3f}, CV_C: {results['cv_C'][best_idx]:.3f}")
        print(f"    - H/C ratio: {results['HC_ratio_mean'][best_idx]:.2f} ± {results['HC_ratio_std'][best_idx]:.2f}")
        print(f"    - Extinction rate: {results['extinction_rate'][best_idx]:.1%}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("="*80)
    
    sensitivity_ranking = []
    for param_name, results in all_results.items():
        # Measure sensitivity as range of CV values
        cv_range = max(results['cv_H']) - min(results['cv_H'])
        sensitivity_ranking.append((param_name, cv_range))
    
    sensitivity_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print("\nParameter Sensitivity Ranking (most to least influential):")
    for i, (param, sensitivity) in enumerate(sensitivity_ranking, 1):
        print(f"{i}. {param:15s} - CV range: {sensitivity:.3f}")
    
    return all_results

if __name__ == "__main__":
    # Base parameters (current configuration)
    base_parameters = {
        'r_H': 0.5,
        'r_C': 0.4,
        'gestation_H': 6.0,
        'gestation_C': 5.0,
        'm_H': 0.02,
        'm_C': 0.03,
        'pred': 0.001,
        'conflict_H': 0.00005,
        'conflict_C': 0.0005,
        'H_threshold': 200,
        'HC_ratio_threshold': 3.0
    }
    
    initial_population = {'H_M': 50, 'H_F': 50, 'C_M': 10, 'C_F': 10}
    
    # Run comprehensive sensitivity analysis
    results = find_optimal_parameters(base_parameters, initial_population)
    
    print("\n" + "="*80)
    print("Analysis complete! Check generated PNG files for detailed plots.")
    print("="*80)
