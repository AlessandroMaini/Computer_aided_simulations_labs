"""
Test Configurations for Carnivore-Herbivore CTMC System

This script tests 5 strategic configurations to analyze:
1. ERGODICITY: Does the system reach a stable equilibrium independent of initial conditions?
2. ABSORBING STATES: Can the system reach extinction states (H=0 or C=0)?
3. PARAMETER SENSITIVITY: How do key rates affect long-term behavior?

Configurations chosen to test:
1. BASELINE - Balanced parameters (current default)
2. HIGH_PREDATION - Test carnivore dominance → potential H extinction
3. LOW_PREDATION - Test herbivore explosion → potential C starvation
4. HIGH_CONFLICT - Test density-dependent mortality → population collapse
5. EXTREME_REPRODUCTION - Test rapid growth dynamics and stability
"""

import numpy as np
import matplotlib.pyplot as plt
from carnivore_herbivore import SimulationEngine, run_multiple_simulations
from typing import Dict, List, Tuple
import json

# --------------------------------------------------
# CONFIGURATION DEFINITIONS
# --------------------------------------------------

CONFIGURATIONS = {
    "1_BASELINE": {
        "description": """
        BASELINE - Balanced Coexistence
        - Default parameters from original simulation
        - Tests: Stable equilibrium, ergodicity from different initial conditions
        - Expected: Both species coexist around H≈150-200, C≈8-15
        - Absorbing risk: LOW (parameters tuned for stability)
        """,
        "initial_pop": {'H_M': 80, 'H_F': 80, 'C_M': 5, 'C_F': 5},
        "params": {
            'r_H': 0.5, 'r_C': 0.4,
            'gestation_H': 6.0, 'gestation_C': 5.0,
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.0007,
            'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        "test_initial_conditions": [
            {'H_M': 75, 'H_F': 75, 'C_M': 25, 'C_F': 25},    # IC1: H=150, C=50 (3:1 ratio)
            {'H_M': 50, 'H_F': 50, 'C_M': 50, 'C_F': 50},    # IC2: H=100, C=100 (1:1 ratio)
            {'H_M': 25, 'H_F': 25, 'C_M': 35, 'C_F': 35},    # IC3: H=50, C=70 (carnivore dominated)
            {'H_M': 50, 'H_F': 50, 'C_M': 5, 'C_F': 5},      # IC4: H=100, C=10 (10:1 ratio)
            {'H_M': 200, 'H_F': 200, 'C_M': 25, 'C_F': 25},  # IC5: H=400, C=50 (8:1 ratio)
        ],
        "color": "green"
    },
    
    "2_HIGH_PREDATION": {
        "description": """
        HIGH PREDATION - Dual Extinction via Overpredation
        - Predation rate increased 7x (0.0007 → 0.005)
        - Carnivore reproduction increased slightly (0.4 → 0.5) for initial boom
        - HC ratio threshold reduced (3.0 → 1.0) to dramatically increase carnivore starvation risk
        - Tests: Herbivore extinction followed by carnivore starvation
        - Expected: Carnivores initially thrive, deplete prey, then starve (both extinct)
        - Absorbing risk: VERY HIGH - Both species typically go extinct (H→0, then C→0)
        """,
        "initial_pop": {'H_M': 80, 'H_F': 80, 'C_M': 5, 'C_F': 5},
        "params": {
            'r_H': 0.5, 
            'r_C': 0.5,  # Slightly increased for initial carnivore boom
            'gestation_H': 6.0, 'gestation_C': 5.0,
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.005,  # 7x INCREASE (strong overpredation)
            'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 1.0  # FURTHER REDUCED from 1.5 to 1.0
        },
        "test_initial_conditions": [
            {'H_M': 100, 'H_F': 100, 'C_M': 30, 'C_F': 30},  # IC1: H=200, C=60 (3.3:1 ratio)
            {'H_M': 80, 'H_F': 80, 'C_M': 80, 'C_F': 80},    # IC2: H=160, C=160 (1:1 ratio)
            {'H_M': 40, 'H_F': 40, 'C_M': 50, 'C_F': 50},    # IC3: H=80, C=100 (carnivore dominated)
            {'H_M': 150, 'H_F': 150, 'C_M': 15, 'C_F': 15},  # IC4: H=300, C=30 (10:1 ratio)
            {'H_M': 250, 'H_F': 250, 'C_M': 40, 'C_F': 40},  # IC5: H=500, C=80 (6.25:1 ratio)
        ],
        "color": "red"
    },
    
    "3_LOW_PREDATION": {
        "description": """
        LOW PREDATION - Carnivore Extinction via Starvation
        - Predation rate decreased 10x (0.0007 → 0.00007) - slightly increased from before
        - Carnivore mortality increased 2x (0.03 → 0.06) - slightly reduced from before
        - Carnivore reproduction decreased (0.4 → 0.3)
        - Tests: Carnivore extinction (absorbing state C=0), herbivore persistence
        - Expected: Carnivores cannot sustain population, herbivores stabilize at threshold
        - Absorbing risk: VERY HIGH for carnivores, herbivores reach equilibrium alone
        """,
        "initial_pop": {'H_M': 80, 'H_F': 80, 'C_M': 5, 'C_F': 5},
        "params": {
            'r_H': 0.5, 
            'r_C': 0.3,   # DECREASED reproduction
            'gestation_H': 6.0, 'gestation_C': 5.0,
            'm_H': 0.02, 
            'm_C': 0.06,  # 2x INCREASE in mortality (less extreme than before)
            'pred': 0.00007,  # 10x DECREASE (less extreme than before)
            'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        "test_initial_conditions": [
            {'H_M': 100, 'H_F': 100, 'C_M': 30, 'C_F': 30},  # IC1: H=200, C=60 (3.3:1 ratio)
            {'H_M': 75, 'H_F': 75, 'C_M': 60, 'C_F': 60},    # IC2: H=150, C=120 (1.25:1 ratio)
            {'H_M': 40, 'H_F': 40, 'C_M': 45, 'C_F': 45},    # IC3: H=80, C=90 (carnivore dominated)
            {'H_M': 80, 'H_F': 80, 'C_M': 8, 'C_F': 8},      # IC4: H=160, C=16 (10:1 ratio)
            {'H_M': 250, 'H_F': 250, 'C_M': 35, 'C_F': 35},  # IC5: H=500, C=70 (7.1:1 ratio)
        ],
        "color": "orange"
    },
    
    "4_HIGH_CONFLICT": {
        "description": """
        HIGH CONFLICT - Density-Dependent Collapse
        - Herbivore conflict increased 10x (0.00005 → 0.0005) - less extreme
        - Carnivore conflict increased 5x (0.0005 → 0.0025) - less extreme
        - Tests: Dual extinction via intraspecific competition
        - Expected: Populations crash when density increases, potential extinction
        - Absorbing risk: HIGH - Negative feedback can collapse both but not guaranteed
        """,
        "initial_pop": {'H_M': 80, 'H_F': 80, 'C_M': 5, 'C_F': 5},
        "params": {
            'r_H': 0.5, 'r_C': 0.4,
            'gestation_H': 6.0, 'gestation_C': 5.0,
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.0007,
            'conflict_H': 0.0005,   # 10x INCREASE (less extreme)
            'conflict_C': 0.0025,   # 5x INCREASE (less extreme)
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        "test_initial_conditions": [
            {'H_M': 75, 'H_F': 75, 'C_M': 25, 'C_F': 25},    # IC1: H=150, C=50 (3:1 ratio)
            {'H_M': 50, 'H_F': 50, 'C_M': 50, 'C_F': 50},    # IC2: H=100, C=100 (1:1 ratio)
            {'H_M': 25, 'H_F': 25, 'C_M': 35, 'C_F': 35},    # IC3: H=50, C=70 (carnivore dominated)
            {'H_M': 50, 'H_F': 50, 'C_M': 5, 'C_F': 5},      # IC4: H=100, C=10 (10:1 ratio)
            {'H_M': 200, 'H_F': 200, 'C_M': 25, 'C_F': 25},  # IC5: H=400, C=50 (8:1 ratio)
        ],
        "color": "purple"
    },
    
    "5_EXTREME_REPRODUCTION": {
        "description": """
        EXTREME REPRODUCTION - Rapid Growth Test
        - Reproduction rates increased 1.5x (r_H: 0.5→0.75, r_C: 0.4→0.6) - less extreme
        - Gestation times reduced 1.5x (faster births) - less extreme
        - Tests: System stability under enhanced growth, oscillations vs equilibrium
        - Expected: Oscillations with potential crashes but more stable than extreme
        - Absorbing risk: LOW-MODERATE - May overshoot but usually recovers
        """,
        "initial_pop": {'H_M': 40, 'H_F': 40, 'C_M': 3, 'C_F': 3},  # Start lower
        "params": {
            'r_H': 0.75,  # 1.5x INCREASE (less extreme)
            'r_C': 0.6,   # 1.5x INCREASE (less extreme)
            'gestation_H': 4.0,  # 1.5x FASTER (less extreme)
            'gestation_C': 3.3,  # 1.5x FASTER (less extreme)
            'm_H': 0.02, 'm_C': 0.03,
            'pred': 0.0007,
            'conflict_H': 0.00005, 'conflict_C': 0.0005,
            'H_threshold': 200, 'HC_ratio_threshold': 3.0
        },
        "test_initial_conditions": [
            {'H_M': 75, 'H_F': 75, 'C_M': 25, 'C_F': 25},    # IC1: H=150, C=50 (3:1 ratio)
            {'H_M': 50, 'H_F': 50, 'C_M': 50, 'C_F': 50},    # IC2: H=100, C=100 (1:1 ratio)
            {'H_M': 25, 'H_F': 25, 'C_M': 35, 'C_F': 35},    # IC3: H=50, C=70 (carnivore dominated)
            {'H_M': 50, 'H_F': 50, 'C_M': 5, 'C_F': 5},      # IC4: H=100, C=10 (10:1 ratio)
            {'H_M': 200, 'H_F': 200, 'C_M': 25, 'C_F': 25},  # IC5: H=400, C=50 (8:1 ratio)
        ],
        "color": "blue"
    }
}

# --------------------------------------------------
# TEST EXECUTION FUNCTIONS
# --------------------------------------------------

def test_single_configuration(config_name: str, config: Dict, t_max: float = 1000.0, 
                              num_runs: int = 10, base_seed: int = 42) -> Dict:
    """
    Test a single configuration with multiple initial conditions and runs.
    
    Returns:
        Dictionary with results including extinction rates, averages, and convergence
    """
    print(f"\n{'='*80}")
    print(f"TESTING CONFIGURATION: {config_name}")
    print(f"{'='*80}")
    print(config["description"])
    print(f"{'='*80}\n")
    
    results = {
        'config_name': config_name,
        'initial_conditions': [],
        'extinction_events': {'H_extinct': 0, 'C_extinct': 0, 'both_extinct': 0, 'both_survive': 0},
        'all_runs_data': []
    }
    
    # Test each initial condition
    for ic_idx, initial_pop in enumerate(config["test_initial_conditions"]):
        print(f"\n--- Initial Condition {ic_idx+1}/{len(config['test_initial_conditions'])} ---")
        print(f"Starting populations: H={initial_pop['H_M'] + initial_pop['H_F']}, C={initial_pop['C_M'] + initial_pop['C_F']}")
        
        ic_results = {
            'initial_pop': initial_pop,
            'runs': [],
            'extinctions': {'H': 0, 'C': 0, 'both': 0, 'none': 0}
        }
        
        # Run multiple replications
        for run_idx in range(num_runs):
            seed = base_seed + ic_idx * 1000 + run_idx
            np.random.seed(seed)
            
            sim = SimulationEngine(t_max=t_max, initial_pop=initial_pop.copy(), 
                                  params=config["params"].copy())
            sim.event_loop()
            
            # Check final state
            final_H = sim.H_M + sim.H_F + sim.H_F_preg
            final_C = sim.C_M + sim.C_F + sim.C_F_preg
            
            # Calculate time-averaged populations
            total_time = sim.current_time
            if total_time > 0:
                avg_H = (sim.metrics.area_under_herbivore_males + 
                        sim.metrics.area_under_herbivore_females + 
                        sim.metrics.area_under_herbivore_females_preg) / total_time
                avg_C = (sim.metrics.area_under_carnivore_males + 
                        sim.metrics.area_under_carnivore_females + 
                        sim.metrics.area_under_carnivore_females_preg) / total_time
            else:
                avg_H = avg_C = 0.0
            
            run_data = {
                'seed': seed,
                'final_H': final_H,
                'final_C': final_C,
                'avg_H': avg_H,
                'avg_C': avg_C,
                'H_extinct': final_H == 0,
                'C_extinct': final_C == 0,
                'simulation_time': total_time
            }
            
            ic_results['runs'].append(run_data)
            results['all_runs_data'].append(run_data)
            
            # Track extinctions
            if final_H == 0 and final_C == 0:
                ic_results['extinctions']['both'] += 1
                results['extinction_events']['both_extinct'] += 1
            elif final_H == 0:
                ic_results['extinctions']['H'] += 1
                results['extinction_events']['H_extinct'] += 1
            elif final_C == 0:
                ic_results['extinctions']['C'] += 1
                results['extinction_events']['C_extinct'] += 1
            else:
                ic_results['extinctions']['none'] += 1
                results['extinction_events']['both_survive'] += 1
            
            print(f"  Run {run_idx+1:2d}: H_final={final_H:4d} (avg={avg_H:6.1f}), "
                  f"C_final={final_C:4d} (avg={avg_C:6.1f})", end="")
            if final_H == 0 or final_C == 0:
                print(" ⚠ EXTINCTION")
            else:
                print(" ✓")
        
        results['initial_conditions'].append(ic_results)
        
        # Summary for this initial condition
        print(f"\n  Extinction Summary for IC {ic_idx+1}:")
        print(f"    H extinct: {ic_results['extinctions']['H']}/{num_runs} ({100*ic_results['extinctions']['H']/num_runs:.1f}%)")
        print(f"    C extinct: {ic_results['extinctions']['C']}/{num_runs} ({100*ic_results['extinctions']['C']/num_runs:.1f}%)")
        print(f"    Both extinct: {ic_results['extinctions']['both']}/{num_runs} ({100*ic_results['extinctions']['both']/num_runs:.1f}%)")
        print(f"    Both survive: {ic_results['extinctions']['none']}/{num_runs} ({100*ic_results['extinctions']['none']/num_runs:.1f}%)")
    
    return results

def plot_configuration_comparison(all_results: Dict[str, Dict], t_max: float = 1000.0):
    """
    Create comprehensive visualization comparing all configurations.
    Split into multiple separate figures for clarity.
    """
    # Filter to only show non-extinct configurations
    non_extinct_configs = ['1_BASELINE', '4_HIGH_CONFLICT', '5_EXTREME_REPRODUCTION']
    config_names = [cn for cn in sorted(all_results.keys()) if cn in non_extinct_configs]
    
    # ============================================================================
    # FIGURE 1: EQUILIBRIUM POPULATIONS WITH H/C RATIO
    # ============================================================================
    fig1 = plt.figure(figsize=(16, 6))
    avg_H_means = []
    avg_C_means = []
    avg_ratio_means = []
    avg_H_stds = []
    avg_C_stds = []
    avg_ratio_stds = []
    survival_counts = []
    
    x = np.arange(len(config_names))
    
    for config_name in config_names:
        results = all_results[config_name]
        surviving_runs = [r for r in results['all_runs_data'] if r['final_H'] > 0 and r['final_C'] > 0]
        survival_counts.append(len(surviving_runs))
        
        if surviving_runs:
            h_vals = [r['avg_H'] for r in surviving_runs]
            c_vals = [r['avg_C'] for r in surviving_runs]
            ratio_vals = [h/c for h, c in zip(h_vals, c_vals)]
            
            avg_H_means.append(np.mean(h_vals))
            avg_C_means.append(np.mean(c_vals))
            avg_ratio_means.append(np.mean(ratio_vals))
            avg_H_stds.append(np.std(h_vals))
            avg_C_stds.append(np.std(c_vals))
            avg_ratio_stds.append(np.std(ratio_vals))
        else:
            avg_H_means.append(0)
            avg_C_means.append(0)
            avg_ratio_means.append(0)
            avg_H_stds.append(0)
            avg_C_stds.append(0)
            avg_ratio_stds.append(0)
    
    # Create subplots: populations and ratio
    ax1 = plt.subplot(1, 2, 1)
    width = 0.35
    bars1 = ax1.bar(x - width/2, avg_H_means, width, yerr=avg_H_stds, 
                    label='Herbivores', color='#2ecc71', alpha=0.8, 
                    capsize=5, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, avg_C_means, width, yerr=avg_C_stds, 
                    label='Carnivores', color='#e74c3c', alpha=0.8, 
                    capsize=5, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (h, c, count) in enumerate(zip(avg_H_means, avg_C_means, survival_counts)):
        if h > 0:
            ax1.text(i - width/2, h + avg_H_stds[i], f'{h:.0f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        if c > 0:
            ax1.text(i + width/2, c + avg_C_stds[i], f'{c:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Time-Averaged Population', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_title('Equilibrium Populations', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([cn.split('_', 1)[1].replace('_', ' ') for cn in config_names], 
                        rotation=15, ha='right', fontsize=9)
    ax1.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # H/C Ratio subplot
    ax2 = plt.subplot(1, 2, 2)
    bars3 = ax2.bar(x, avg_ratio_means, width*2, yerr=avg_ratio_stds, 
                    label='H/C Ratio', color='#9b59b6', alpha=0.8, 
                    capsize=5, edgecolor='black', linewidth=1.5)
    
    # Add value labels and sample counts
    for i, (ratio, count) in enumerate(zip(avg_ratio_means, survival_counts)):
        if ratio > 0:
            ax2.text(i, ratio + avg_ratio_stds[i], f'{ratio:.2f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i, -0.5, f'n={count}', ha='center', va='top', fontsize=8, 
                style='italic', color='gray')
    
    ax2.set_ylabel('H/C Ratio', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_title('Herbivore/Carnivore Ratio', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([cn.split('_', 1)[1].replace('_', ' ') for cn in config_names], 
                        rotation=15, ha='right', fontsize=9)
    ax2.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fig1_equilibrium_populations_and_ratio.png', dpi=300, bbox_inches='tight')
    print("\n📊 Figure 1 saved: fig1_equilibrium_populations_and_ratio.png")
    plt.show()
    
    # ============================================================================
    # FIGURE 2: ERGODICITY TEST (BASELINE ONLY)
    # ============================================================================
    fig2 = plt.figure(figsize=(12, 10))
    
    ax3 = plt.subplot(1, 1, 1)
    
    # Only plot BASELINE configuration for ergodicity test
    baseline_config_name = '1_BASELINE'
    if baseline_config_name in all_results:
        results = all_results[baseline_config_name]
        config = CONFIGURATIONS[baseline_config_name]
        
        # Different colors/markers for different initial conditions
        ic_colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']  # Red, Blue, Orange, Purple, Teal
        ic_markers = ['o', 's', '^', 'D', 'v']  # Circle, Square, Triangle up, Diamond, Triangle down
        ic_labels = []
        
        from matplotlib.lines import Line2D
        
        for ic_idx, ic_data in enumerate(results['initial_conditions']):
            initial_H = ic_data['initial_pop']['H_M'] + ic_data['initial_pop']['H_F']
            initial_C = ic_data['initial_pop']['C_M'] + ic_data['initial_pop']['C_F']
            
            ic_label = f'IC{ic_idx+1}: H₀={initial_H}, C₀={initial_C}'
            ic_labels.append(ic_label)
            
            for run in ic_data['runs']:
                if run['final_H'] > 0 and run['final_C'] > 0:  # Only surviving runs
                    # Plot herbivores
                    ax3.scatter(initial_H, run['avg_H'], alpha=0.7, 
                              color=ic_colors[ic_idx % len(ic_colors)], s=80, 
                              marker=ic_markers[ic_idx % len(ic_markers)], 
                              edgecolor='black', linewidth=1.0)
                    # Plot carnivores  
                    ax3.scatter(initial_C, run['avg_C'], alpha=0.7, 
                              color=ic_colors[ic_idx % len(ic_colors)], s=80, 
                              marker=ic_markers[ic_idx % len(ic_markers)], 
                              edgecolor='black', linewidth=1.0)
        
        # Add horizontal lines at equilibrium values (show ergodic convergence)
        surviving_runs = [r for r in results['all_runs_data'] if r['final_H'] > 0 and r['final_C'] > 0]
        if surviving_runs:
            mean_H = np.mean([r['avg_H'] for r in surviving_runs])
            mean_C = np.mean([r['avg_C'] for r in surviving_runs])
            std_H = np.std([r['avg_H'] for r in surviving_runs])
            std_C = np.std([r['avg_C'] for r in surviving_runs])
            
            # Herbivore equilibrium band
            ax3.axhspan(mean_H - std_H, mean_H + std_H, alpha=0.15, color='green', 
                       label=f'H Equilibrium: {mean_H:.0f}±{std_H:.0f}')
            ax3.axhline(mean_H, color='green', linestyle='--', linewidth=2, alpha=0.7)
            
            # Carnivore equilibrium band
            ax3.axhspan(mean_C - std_C, mean_C + std_C, alpha=0.15, color='red',
                       label=f'C Equilibrium: {mean_C:.1f}±{std_C:.1f}')
            ax3.axhline(mean_C, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Create custom legend
        legend_elements = [
            Line2D([0], [0], marker=ic_markers[i], color='w', 
                  markerfacecolor=ic_colors[i], markersize=10, markeredgecolor='black',
                  label=ic_labels[i]) for i in range(len(ic_labels))
        ]
    
    ax3.set_xlabel('Initial Population', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time-Averaged Final Population', fontsize=12, fontweight='bold')
    ax3.set_title('Ergodicity Test: BASELINE Configuration', 
                 fontsize=14, fontweight='bold', pad=20)
    ax3.legend(handles=legend_elements, fontsize=10, loc='lower right', framealpha=0.9)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_xlim(-20, 450)
    
    plt.tight_layout()
    plt.savefig('fig2_ergodicity_test.png', dpi=300, bbox_inches='tight')
    print("📊 Figure 2 saved: fig2_ergodicity_test.png")
    plt.show()
    
    # ============================================================================
    # FIGURE 3: SAMPLE TRAJECTORIES (ALL 5 CONFIGS, 5 DIFFERENT SEEDS)
    # ============================================================================
    fig3 = plt.figure(figsize=(18, 12))
    
    seeds_to_test = [42, 123, 456, 789, 1011]  # 5 different random seeds
    trajectory_ic = {'H_M': 75, 'H_F': 75, 'C_M': 25, 'C_F': 25}  # H=150, C=50
    
    # Use ALL configurations for trajectory plot
    all_config_names = sorted(all_results.keys())
    
    for idx, config_name in enumerate(all_config_names):
        config = CONFIGURATIONS[config_name]
        
        ax = plt.subplot(2, 3, idx + 1)
        
        # Run 5 simulations with different seeds
        colors_H = ['#27ae60', '#2ecc71', '#52be80', '#7dcea0', '#a9dfbf']
        colors_C = ['#c0392b', '#e74c3c', '#ec7063', '#f1948a', '#f5b7b1']
        
        max_time = 0  # Track longest simulation time
        
        for seed_idx, seed in enumerate(seeds_to_test):
            np.random.seed(seed)
            sim = SimulationEngine(t_max=t_max, 
                                  initial_pop=trajectory_ic.copy(),
                                  params=config['params'].copy())
            sim.event_loop()
            
            times = sim.metrics.history_time
            h_total = (np.array(sim.metrics.history_H_M) + 
                      np.array(sim.metrics.history_H_F) + 
                      np.array(sim.metrics.history_H_F_preg))
            c_total = (np.array(sim.metrics.history_C_M) + 
                      np.array(sim.metrics.history_C_F) + 
                      np.array(sim.metrics.history_C_F_preg))
            
            # Update max time
            if len(times) > 0:
                max_time = max(max_time, times[-1])
            
            # Plot with varying transparency and thickness - make all visible
            alpha_val = 0.7 if seed_idx == 0 else 0.5
            linewidth_val = 2.0 if seed_idx == 0 else 1.5
            label_H = 'Herbivores' if seed_idx == 0 else None
            label_C = 'Carnivores' if seed_idx == 0 else None
            
            ax.plot(times, h_total, color=colors_H[seed_idx], linewidth=linewidth_val, 
                   label=label_H, alpha=alpha_val)
            ax.plot(times, c_total, color=colors_C[seed_idx], linewidth=linewidth_val, 
                   label=label_C, alpha=alpha_val)
        
        # Get statistics from the first seed for title
        np.random.seed(seeds_to_test[0])
        sim = SimulationEngine(t_max=t_max, 
                              initial_pop=trajectory_ic.copy(),
                              params=config['params'].copy())
        sim.event_loop()
        final_H = sim.H_M + sim.H_F + sim.H_F_preg
        final_C = sim.C_M + sim.C_F + sim.C_F_preg
        
        title = config_name.split("_", 1)[1].replace('_', ' ')
        
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('Population', fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Set dynamic x-axis limit based on longest simulation
        ax.set_xlim(0, max_time * 1.02)  # Add 2% padding
    
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    plt.savefig('fig3_sample_trajectories.png', dpi=300, bbox_inches='tight')
    print("📊 Figure 3 saved: fig3_sample_trajectories.png")
    plt.show()

def generate_analysis_report(all_results: Dict[str, Dict], output_file: str = 'ergodicity_analysis.txt'):
    """
    Generate comprehensive text report analyzing ergodicity and absorbing states.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ERGODICITY AND ABSORBING STATE ANALYSIS\n")
        f.write("Carnivore-Herbivore CTMC System\n")
        f.write("="*80 + "\n\n")
        
        for config_name, results in all_results.items():
            config = CONFIGURATIONS[config_name]
            total_runs = len(results['all_runs_data'])
            
            f.write(f"\n{'='*80}\n")
            f.write(f"CONFIGURATION: {config_name}\n")
            f.write(f"{'='*80}\n")
            f.write(config['description'])
            f.write(f"\n{'='*80}\n\n")
            
            # Extinction analysis
            f.write("ABSORBING STATE ANALYSIS:\n")
            f.write(f"  Total runs: {total_runs}\n")
            f.write(f"  H extinct:     {results['extinction_events']['H_extinct']:3d} ({100*results['extinction_events']['H_extinct']/total_runs:5.1f}%)\n")
            f.write(f"  C extinct:     {results['extinction_events']['C_extinct']:3d} ({100*results['extinction_events']['C_extinct']/total_runs:5.1f}%)\n")
            f.write(f"  Both extinct:  {results['extinction_events']['both_extinct']:3d} ({100*results['extinction_events']['both_extinct']/total_runs:5.1f}%)\n")
            f.write(f"  Both survive:  {results['extinction_events']['both_survive']:3d} ({100*results['extinction_events']['both_survive']/total_runs:5.1f}%)\n\n")
            
            # Ergodicity analysis
            f.write("ERGODICITY ANALYSIS:\n")
            
            surviving_runs = [r for r in results['all_runs_data'] if r['final_H'] > 0 and r['final_C'] > 0]
            
            if surviving_runs:
                avg_H_vals = [r['avg_H'] for r in surviving_runs]
                avg_C_vals = [r['avg_C'] for r in surviving_runs]
                
                f.write(f"  Surviving runs: {len(surviving_runs)}/{total_runs}\n")
                f.write(f"  Herbivore avg: {np.mean(avg_H_vals):7.2f} ± {np.std(avg_H_vals):6.2f} (CV={100*np.std(avg_H_vals)/np.mean(avg_H_vals):5.1f}%)\n")
                f.write(f"  Carnivore avg: {np.mean(avg_C_vals):7.2f} ± {np.std(avg_C_vals):6.2f} (CV={100*np.std(avg_C_vals)/np.mean(avg_C_vals):5.1f}%)\n")
                
                # Test convergence across initial conditions
                if len(config['test_initial_conditions']) > 1:
                    f.write(f"\n  Initial Condition Comparison:\n")
                    for ic_idx, ic_data in enumerate(results['initial_conditions']):
                        ic_surviving = [r for r in ic_data['runs'] if r['final_H'] > 0 and r['final_C'] > 0]
                        if ic_surviving:
                            ic_avg_H = np.mean([r['avg_H'] for r in ic_surviving])
                            ic_avg_C = np.mean([r['avg_C'] for r in ic_surviving])
                            initial_H = ic_data['initial_pop']['H_M'] + ic_data['initial_pop']['H_F']
                            initial_C = ic_data['initial_pop']['C_M'] + ic_data['initial_pop']['C_F']
                            f.write(f"    IC{ic_idx+1} (H₀={initial_H:3d}, C₀={initial_C:2d}): H_avg={ic_avg_H:6.1f}, C_avg={ic_avg_C:6.1f}\n")
                    
                    # Calculate coefficient of variation across ICs
                    ic_means_H = []
                    ic_means_C = []
                    for ic_data in results['initial_conditions']:
                        ic_surviving = [r for r in ic_data['runs'] if r['final_H'] > 0 and r['final_C'] > 0]
                        if ic_surviving:
                            ic_means_H.append(np.mean([r['avg_H'] for r in ic_surviving]))
                            ic_means_C.append(np.mean([r['avg_C'] for r in ic_surviving]))
                    
                    if len(ic_means_H) > 1:
                        cv_H = 100 * np.std(ic_means_H) / np.mean(ic_means_H)
                        cv_C = 100 * np.std(ic_means_C) / np.mean(ic_means_C)
                        f.write(f"\n  Inter-IC Variability: H_CV={cv_H:.1f}%, C_CV={cv_C:.1f}%\n")
                        
                        if cv_H < 10 and cv_C < 10:
                            f.write(f"  ✓ ERGODIC: Low variability across initial conditions suggests ergodicity\n")
                        elif cv_H < 30 and cv_C < 30:
                            f.write(f"  → WEAK ERGODICITY: Moderate convergence across initial conditions\n")
                        else:
                            f.write(f"  ✗ NON-ERGODIC: High variability suggests path dependence\n")
            else:
                f.write(f"  ✗ NO SURVIVING RUNS - System always reaches absorbing state\n")
                f.write(f"  CONCLUSION: System is NOT ERGODIC (all paths lead to extinction)\n")
            
            f.write("\n")
    
    print(f"\n📄 Analysis report saved to: {output_file}")

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    """
    Run all test configurations and generate comprehensive analysis.
    """
    print("\n" + "="*80)
    print("ERGODICITY AND ABSORBING STATE TESTING")
    print("Carnivore-Herbivore CTMC System")
    print("="*80)
    print(f"\nTesting {len(CONFIGURATIONS)} configurations with multiple initial conditions")
    print("This will take several minutes...\n")
    
    # Test parameters
    T_MAX = 1000.0      # Simulation time
    NUM_RUNS = 5        # Runs per initial condition
    BASE_SEED = 42      # For reproducibility
    
    all_results = {}
    
    # Test each configuration
    for config_name in sorted(CONFIGURATIONS.keys()):
        config = CONFIGURATIONS[config_name]
        results = test_single_configuration(
            config_name=config_name,
            config=config,
            t_max=T_MAX,
            num_runs=NUM_RUNS,
            base_seed=BASE_SEED
        )
        all_results[config_name] = results
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING ANALYSIS OUTPUTS")
    print("="*80)
    plot_configuration_comparison(all_results, t_max=T_MAX)
    
    # Generate text report
    generate_analysis_report(all_results)
    
    # Save raw results as JSON
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open('test_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("\n💾 Raw results saved to: test_results.json")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for config_name in sorted(all_results.keys()):
        results = all_results[config_name]
        total_runs = len(results['all_runs_data'])
        survival_rate = 100 * results['extinction_events']['both_survive'] / total_runs
        
        print(f"\n{config_name}:")
        print(f"  Survival rate: {survival_rate:.1f}%")
        
        surviving_runs = [r for r in results['all_runs_data'] if r['final_H'] > 0 and r['final_C'] > 0]
        if surviving_runs:
            avg_H = np.mean([r['avg_H'] for r in surviving_runs])
            avg_C = np.mean([r['avg_C'] for r in surviving_runs])
            print(f"  Equilibrium: H≈{avg_H:.0f}, C≈{avg_C:.0f}")
    
    print("\n" + "="*80)
    print("✓ TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
