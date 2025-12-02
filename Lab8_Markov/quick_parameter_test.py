"""
Quick Parameter Sensitivity Analysis - Focus on Key Parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from carnivore_herbivore import SimulationEngine
from typing import Dict, List

def quick_stability_test(params: Dict[str, float], 
                        initial_pop: Dict[str, int],
                        num_runs: int = 20,
                        t_max: float = 500.0) -> Dict:
    """Quick test of parameter stability."""
    H_values = []
    C_values = []
    extinctions = 0
    
    for run in range(num_runs):
        np.random.seed(2000 + run)
        
        sim = SimulationEngine(t_max=t_max, initial_pop=initial_pop, params=params)
        sim.event_loop()
        
        total_time = sim.current_time
        if total_time > 0:
            avg_H = (sim.metrics.area_under_herbivore_males + 
                    sim.metrics.area_under_herbivore_females + 
                    sim.metrics.area_under_herbivore_females_preg) / total_time
            avg_C = (sim.metrics.area_under_carnivore_males + 
                    sim.metrics.area_under_carnivore_females + 
                    sim.metrics.area_under_carnivore_females_preg) / total_time
            
            if avg_H < 1.0 or avg_C < 0.5:
                extinctions += 1
                
            H_values.append(avg_H)
            C_values.append(avg_C)
    
    H_mean = np.mean(H_values) if H_values else 0
    C_mean = np.mean(C_values) if C_values else 0
    H_std = np.std(H_values) if H_values else 0
    C_std = np.std(C_values) if C_values else 0
    
    return {
        'H_mean': H_mean,
        'H_std': H_std,
        'C_mean': C_mean,
        'C_std': C_std,
        'cv_H': H_std / H_mean if H_mean > 0 else np.inf,
        'cv_C': C_std / C_mean if C_mean > 0 else np.inf,
        'extinction_rate': extinctions / num_runs,
        'HC_ratio': H_mean / C_mean if C_mean > 0 else 0
    }

def test_configurations():
    """Test multiple parameter configurations."""
    
    initial_pop = {'H_M': 50, 'H_F': 50, 'C_M': 10, 'C_F': 10}
    
    configs = {
        'Current': {
            'r_H': 0.5, 'r_C': 0.4,
            'gestation_H': 6.0, 'gestation_C': 5.0,
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.001, 'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        'Config_1 (Lower reproduction)': {
            'r_H': 0.4, 'r_C': 0.35,
            'gestation_H': 7.0, 'gestation_C': 6.0,
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.001, 'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        'Config_2 (Slower predation)': {
            'r_H': 0.5, 'r_C': 0.4,
            'gestation_H': 6.0, 'gestation_C': 5.0,
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.0007, 'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        'Config_3 (Longer gestation)': {
            'r_H': 0.5, 'r_C': 0.4,
            'gestation_H': 8.0, 'gestation_C': 7.0,
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.001, 'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        'Config_4 (Balanced - RECOMMENDED)': {
            'r_H': 0.45, 'r_C': 0.38,
            'gestation_H': 7.0, 'gestation_C': 6.0,
            'm_H': 0.025, 'm_C': 0.032,
            'pred': 0.0008, 'conflict_H': 0.00003, 'conflict_C': 0.0004,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        'Config_5 (High stability focus)': {
            'r_H': 0.42, 'r_C': 0.36,
            'gestation_H': 8.0, 'gestation_C': 7.0,
            'm_H': 0.023, 'm_C': 0.030,
            'pred': 0.0007, 'conflict_H': 0.00002, 'conflict_C': 0.0003,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        }
    }
    
    print("\n" + "="*100)
    print("TESTING MULTIPLE PARAMETER CONFIGURATIONS")
    print("="*100 + "\n")
    
    results = {}
    for name, params in configs.items():
        print(f"Testing {name}...")
        result = quick_stability_test(params, initial_pop, num_runs=20, t_max=500.0)
        results[name] = result
        
        print(f"  H: {result['H_mean']:.1f} ± {result['H_std']:.1f} (CV={result['cv_H']:.3f})")
        print(f"  C: {result['C_mean']:.1f} ± {result['C_std']:.1f} (CV={result['cv_C']:.3f})")
        print(f"  H/C Ratio: {result['HC_ratio']:.2f}")
        print(f"  Extinction Rate: {result['extinction_rate']:.1%}")
        print(f"  Stability Score: {(result['cv_H'] + result['cv_C'])/2:.3f}\n")
    
    # Summary comparison
    print("\n" + "="*100)
    print("CONFIGURATION COMPARISON")
    print("="*100)
    print(f"{'Configuration':<35} {'H Pop':<15} {'C Pop':<15} {'CV_H':<10} {'CV_C':<10} {'Extinct%':<10} {'Score':<10}")
    print("-"*100)
    
    for name, result in results.items():
        score = (result['cv_H'] + result['cv_C']) / 2 + 2 * result['extinction_rate']
        print(f"{name:<35} {result['H_mean']:6.1f}±{result['H_std']:4.1f}   "
              f"{result['C_mean']:6.1f}±{result['C_std']:4.1f}   "
              f"{result['cv_H']:<10.3f} {result['cv_C']:<10.3f} "
              f"{result['extinction_rate']*100:<10.1f} {score:<10.3f}")
    
    # Find best configuration
    best_config = min(results.items(), 
                     key=lambda x: (x[1]['cv_H'] + x[1]['cv_C'])/2 + 2*x[1]['extinction_rate'])
    
    print("\n" + "="*100)
    print(f"RECOMMENDED CONFIGURATION: {best_config[0]}")
    print("="*100)
    print("\nParameter values:")
    for param, value in configs[best_config[0]].items():
        print(f"  {param:20s}: {value}")
    
    print(f"\nExpected Performance:")
    print(f"  - Herbivore population: {best_config[1]['H_mean']:.1f} ± {best_config[1]['H_std']:.1f}")
    print(f"  - Carnivore population: {best_config[1]['C_mean']:.1f} ± {best_config[1]['C_std']:.1f}")
    print(f"  - Coefficient of Variation: H={best_config[1]['cv_H']:.3f}, C={best_config[1]['cv_C']:.3f}")
    print(f"  - H/C Ratio: {best_config[1]['HC_ratio']:.2f}")
    print(f"  - Extinction Risk: {best_config[1]['extinction_rate']:.1%}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(results.keys())
    cv_h_vals = [results[n]['cv_H'] for n in names]
    cv_c_vals = [results[n]['cv_C'] for n in names]
    ext_vals = [results[n]['extinction_rate'] for n in names]
    
    # Plot 1: CV comparison
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width/2, cv_h_vals, width, label='CV Herbivores', color='green', alpha=0.7)
    axes[0].bar(x + width/2, cv_c_vals, width, label='CV Carnivores', color='red', alpha=0.7)
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Coefficient of Variation')
    axes[0].set_title('Stability Comparison (lower is better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.split('(')[0].strip() for n in names], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.1, color='blue', linestyle='--', alpha=0.5, label='Target CV=0.1')
    
    # Plot 2: Population means
    h_means = [results[n]['H_mean'] for n in names]
    c_means = [results[n]['C_mean'] for n in names]
    axes[1].bar(x - width/2, h_means, width, label='Herbivores', color='green', alpha=0.7)
    axes[1].bar(x + width/2, c_means, width, label='Carnivores', color='red', alpha=0.7)
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Average Population')
    axes[1].set_title('Population Levels')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.split('(')[0].strip() for n in names], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Extinction rate
    axes[2].bar(x, ext_vals, color='darkred', alpha=0.7)
    axes[2].set_xlabel('Configuration')
    axes[2].set_ylabel('Extinction Rate')
    axes[2].set_title('Extinction Risk (lower is better)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([n.split('(')[0].strip() for n in names], rotation=45, ha='right')
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('configuration_comparison.png', dpi=150)
    print(f"\nPlot saved: configuration_comparison.png")
    plt.show()
    
    return best_config[0], configs[best_config[0]]

if __name__ == "__main__":
    best_name, best_params = test_configurations()
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nUpdate carnivore_herbivore.py with the '{best_name}' parameters for optimal stability.")
