import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List, Optional
import sys
import os
from confidence_interval import ConfidenceInterval

# --------------------------------------------------
# EVENT DEFINITION
# --------------------------------------------------
class EventType(Enum):
    # Base Vital Dynamics
    REPRODUCTION_H = 1
    REPRODUCTION_C = 2
    
    BIRTH_H_M = 3
    BIRTH_H_F = 4
    BIRTH_C_M = 5
    BIRTH_C_F = 6
    
    DEATH_H_M = 7
    DEATH_H_F = 8
    DEATH_H_F_PREG = 9 
    DEATH_C_M = 10
    DEATH_C_F = 11
    DEATH_C_F_PREG = 12 
    
    # Interactions
    PREDATION = 13          
    CONFLICT_H_M = 14      
    CONFLICT_H_F = 15      
    CONFLICT_C_M = 16      
    CONFLICT_C_F = 17      

class Event:
    """Discrete event in a CTMC context."""
    def __init__(self, event_type: EventType, time: float):
        self.event_type = event_type
        self.time = time
    
    def __lt__(self, other) -> bool:
        return self.time < other.time

# --------------------------------------------------
# METRICS COLLECTION
# --------------------------------------------------
class Metrics:
    """Collects population time-series data."""
    def __init__(self):
        self.history_time = []
        self.history_H_M = []
        self.history_H_F = []
        self.history_H_F_preg = []
        self.history_C_M = []
        self.history_C_F = []
        self.history_C_F_preg = []
        
        self.counts = {
            'conceptions_H': 0, 'conceptions_C': 0,
            'births_H': 0, 'births_C': 0,
            'deaths_H': 0, 'deaths_C': 0,
            'predations': 0,
            'conflicts_H': 0, 'conflicts_C': 0
        }

        # Time-weighted statistics
        self.last_time = 0.0
        self.area_under_herbivore_males = 0.0
        self.area_under_herbivore_females = 0.0
        self.area_under_herbivore_females_preg = 0.0
        self.area_under_carnivore_males = 0.0
        self.area_under_carnivore_females = 0.0
        self.area_under_carnivore_females_preg = 0.0

    def record_state(self, t, H_M, H_F, H_F_preg, C_M, C_F, C_F_preg):
        self.history_time.append(t)
        self.history_H_M.append(H_M)
        self.history_H_F.append(H_F)
        self.history_H_F_preg.append(H_F_preg)
        self.history_C_M.append(C_M)
        self.history_C_F.append(C_F)
        self.history_C_F_preg.append(C_F_preg)

    def update_time_weighted_stats(self, current_time: float, H_M: int, H_F: int, H_F_preg: int, C_M: int, C_F: int, C_F_preg: int):
        """Update time-weighted statistics."""
        time_diff = current_time - self.last_time
        self.area_under_herbivore_males += H_M * time_diff
        self.area_under_herbivore_females += H_F * time_diff
        self.area_under_herbivore_females_preg += H_F_preg * time_diff
        self.area_under_carnivore_males += C_M * time_diff
        self.area_under_carnivore_females += C_F * time_diff
        self.area_under_carnivore_females_preg += C_F_preg * time_diff
        self.last_time = current_time

# --------------------------------------------------
# SIMULATION ENGINE (CTMC)
# --------------------------------------------------
class SimulationEngine:
    def __init__(self, t_max: float,
                 initial_pop: Dict[str, int],
                 params: Dict[str, float]):
        self.t_max = t_max
        
        self.H_M = initial_pop['H_M']
        self.H_F = initial_pop['H_F']
        self.H_F_preg = 0
        
        self.C_M = initial_pop['C_M']
        self.C_F = initial_pop['C_F']
        self.C_F_preg = 0
        
        # PARAMETERS
        self.params = params
        self.metrics = Metrics()
        self.current_time = 0.0

    def calculate_rates(self) -> Dict[EventType, float]:
        rates = {}
        
        # --- 1. CURRENT TOTALS ---
        H_total = self.H_M + self.H_F + self.H_F_preg
        C_total = self.C_M + self.C_F + self.C_F_preg
        
        # Avoid division by zero
        ratio_HC = H_total / C_total if C_total > 0 else float('inf')

        # --- 2. APPLY THRESHOLD LOGIC (Resource Competition) ---
        
        # Herbivore Competition
        k_H = 0.1
        h_fert_factor = 1.0 / (1.0 + np.exp(k_H * (H_total - self.params['H_threshold'])))
        h_mort_factor = 1.0 + 1.0 / (1.0 + np.exp(-k_H * (H_total - self.params['H_threshold'])))
            
        # Carnivore Competition
        k_C = 2.0
        c_fert_factor = 1.0 / (1.0 + np.exp(-k_C * (ratio_HC - self.params['HC_ratio_threshold'])))
        c_mort_factor = 1.0 + 1.0 / (1.0 + np.exp(k_C * (ratio_HC - self.params['HC_ratio_threshold'])))

        # --- 3. CALCULATE RATES ---
        
        # REPRODUCTION (CONCEPTION)
        # Rate depends on NON-PREGNANT females
        can_mate_H = 1 if self.H_M > 0 else 0
        can_mate_C = 1 if self.C_M > 0 else 0
        
        rates[EventType.REPRODUCTION_H] = self.params['r_H'] * h_fert_factor * self.H_F * can_mate_H
        rates[EventType.REPRODUCTION_C] = self.params['r_C'] * c_fert_factor * self.C_F * can_mate_C

        # BIRTH (End of Gestation)
        # Rate = 1/gestation_period * number_pregnant (Markovian approximation)
        rate_birth_H = (1.0 / self.params['gestation_H']) * self.H_F_preg
        rates[EventType.BIRTH_H_M] = 0.5 * rate_birth_H
        rates[EventType.BIRTH_H_F] = 0.5 * rate_birth_H
        
        rate_birth_C = (1.0 / self.params['gestation_C']) * self.C_F_preg
        rates[EventType.BIRTH_C_M] = 0.5 * rate_birth_C
        rates[EventType.BIRTH_C_F] = 0.5 * rate_birth_C

        # NATURAL MORTALITY
        rates[EventType.DEATH_H_M] = self.params['m_H'] * h_mort_factor * self.H_M
        rates[EventType.DEATH_H_F] = self.params['m_H'] * h_mort_factor * self.H_F
        rates[EventType.DEATH_H_F_PREG] = self.params['m_H'] * h_mort_factor * self.H_F_preg
        
        rates[EventType.DEATH_C_M] = self.params['m_C'] * c_mort_factor * self.C_M
        rates[EventType.DEATH_C_F] = self.params['m_C'] * c_mort_factor * self.C_F
        rates[EventType.DEATH_C_F_PREG] = self.params['m_C'] * c_mort_factor * self.C_F_preg

        # PREDATION
        # Predators encounter Prey (including pregnant ones)
        rates[EventType.PREDATION] = self.params['pred'] * H_total * C_total

        # CONFLICT
        rates[EventType.CONFLICT_H_M] = self.params['conflict_H'] * self.H_M * (self.H_M - 1)
        rates[EventType.CONFLICT_H_F] = self.params['conflict_H'] * self.H_F * (self.H_F - 1)
        rates[EventType.CONFLICT_C_M] = self.params['conflict_C'] * self.C_M * (self.C_M - 1)
        rates[EventType.CONFLICT_C_F] = self.params['conflict_C'] * self.C_F * (self.C_F - 1)

        return rates

    def step(self):
        rates = self.calculate_rates()
        total_rate = sum(rates.values())

        if total_rate <= 0:
            return False

        dt = np.random.exponential(1.0 / total_rate)
        
        if self.current_time + dt > self.t_max:
            self.current_time = self.t_max
            return False
        
        self.current_time += dt

        event_types = list(rates.keys())
        event_probs = np.array(list(rates.values())) / total_rate
        
        selected_type = np.random.choice(event_types, p=event_probs)
        
        event = Event(selected_type, self.current_time)
        self.handle_event(event)
        
        return True

    def handle_event(self, event: Event):
        t = event.event_type
        
        # --- REPRODUCTION (Conception) ---
        if t == EventType.REPRODUCTION_H:
            self.H_F -= 1
            self.H_F_preg += 1
            self.metrics.counts['conceptions_H'] += 1
        elif t == EventType.REPRODUCTION_C:
            self.C_F -= 1
            self.C_F_preg += 1
            self.metrics.counts['conceptions_C'] += 1

        # --- BIRTHS ---
        elif t == EventType.BIRTH_H_M:
            self.H_F_preg -= 1
            self.H_F += 1 # Mother returns
            self.H_M += 1 # Offspring
            self.metrics.counts['births_H'] += 1
        elif t == EventType.BIRTH_H_F:
            self.H_F_preg -= 1
            self.H_F += 1
            self.H_F += 1 # Offspring
            self.metrics.counts['births_H'] += 1
        elif t == EventType.BIRTH_C_M:
            self.C_F_preg -= 1
            self.C_F += 1
            self.C_M += 1
            self.metrics.counts['births_C'] += 1
        elif t == EventType.BIRTH_C_F:
            self.C_F_preg -= 1
            self.C_F += 1
            self.C_F += 1
            self.metrics.counts['births_C'] += 1
            
        # --- NATURAL DEATHS ---
        elif t == EventType.DEATH_H_M: 
            self.H_M -= 1
            self.metrics.counts['deaths_H'] += 1
        elif t == EventType.DEATH_H_F: 
            self.H_F -= 1
            self.metrics.counts['deaths_H'] += 1
        elif t == EventType.DEATH_H_F_PREG: 
            self.H_F_preg -= 1
            self.metrics.counts['deaths_H'] += 1
        
        elif t == EventType.DEATH_C_M: 
            self.C_M -= 1
            self.metrics.counts['deaths_C'] += 1
        elif t == EventType.DEATH_C_F: 
            self.C_F -= 1
            self.metrics.counts['deaths_C'] += 1
        elif t == EventType.DEATH_C_F_PREG: 
            self.C_F_preg -= 1
            self.metrics.counts['deaths_C'] += 1
            
        # --- PREDATION ---
        elif t == EventType.PREDATION:
            H_total = self.H_M + self.H_F + self.H_F_preg
            if H_total > 0:
                rand = np.random.rand() * H_total
                if rand < self.H_M:
                    self.H_M -= 1
                elif rand < self.H_M + self.H_F:
                    self.H_F -= 1
                else:
                    self.H_F_preg -= 1
                self.metrics.counts['predations'] += 1
                
        # --- CONFLICTS ---
        elif t == EventType.CONFLICT_H_M and self.H_M > 1: 
            self.H_M -= 1
            self.metrics.counts['conflicts_H'] += 1
        elif t == EventType.CONFLICT_H_F and self.H_F > 1: 
            self.H_F -= 1
            self.metrics.counts['conflicts_H'] += 1
        elif t == EventType.CONFLICT_C_M and self.C_M > 1: 
            self.C_M -= 1
            self.metrics.counts['conflicts_C'] += 1
        elif t == EventType.CONFLICT_C_F and self.C_F > 1: 
            self.C_F -= 1
            self.metrics.counts['conflicts_C'] += 1

    def event_loop(self):
        self.metrics.record_state(0, self.H_M, self.H_F, self.H_F_preg, self.C_M, self.C_F, self.C_F_preg)
        while self.current_time < self.t_max:
            active = self.step()
            self.metrics.record_state(self.current_time, self.H_M, self.H_F, self.H_F_preg, self.C_M, self.C_F, self.C_F_preg)
            self.metrics.update_time_weighted_stats(self.current_time, self.H_M, self.H_F, self.H_F_preg, self.C_M, self.C_F, self.C_F_preg)
            if not active: 
                break

    def print_statistics(self):
        m = self.metrics
        total_time = self.current_time
        
        print("\n" + "="*60)
        print("SIMULATION STATISTICS")
        print("="*60)
        
        if total_time > 0:
            avg_H_M = m.area_under_herbivore_males / total_time
            avg_H_F = m.area_under_herbivore_females / total_time
            avg_H_F_preg = m.area_under_herbivore_females_preg / total_time
            avg_H_total = avg_H_M + avg_H_F + avg_H_F_preg
            
            avg_C_M = m.area_under_carnivore_males / total_time
            avg_C_F = m.area_under_carnivore_females / total_time
            avg_C_F_preg = m.area_under_carnivore_females_preg / total_time
            avg_C_total = avg_C_M + avg_C_F + avg_C_F_preg
            
            print(f"\nTime-Averaged Populations:")
            print(f"  Herbivore Males:            {avg_H_M:>8.2f}")
            print(f"  Herbivore Females:          {avg_H_F:>8.2f}")
            print(f"  Herbivore Females (Preg):   {avg_H_F_preg:>8.2f}")
            print(f"  Herbivore Total (avg):      {avg_H_total:>8.2f}")
            print(f"\n  Carnivore Males:            {avg_C_M:>8.2f}")
            print(f"  Carnivore Females:          {avg_C_F:>8.2f}")
            print(f"  Carnivore Females (Preg):   {avg_C_F_preg:>8.2f}")
            print(f"  Carnivore Total (avg):      {avg_C_total:>8.2f}")
            
        print(f"\nFinal Populations at t={total_time:.2f}:")
        print(f"  Herbivore Males:            {self.H_M:>8}")
        print(f"  Herbivore Females:          {self.H_F:>8}")
        print(f"  Herbivore Females (Preg):   {self.H_F_preg:>8}")
        print(f"  Herbivore Total:            {self.H_M + self.H_F + self.H_F_preg:>8}")
        print(f"\n  Carnivore Males:            {self.C_M:>8}")
        print(f"  Carnivore Females:          {self.C_F:>8}")
        print(f"  Carnivore Females (Preg):   {self.C_F_preg:>8}")
        print(f"  Carnivore Total:            {self.C_M + self.C_F + self.C_F_preg:>8}")
        
        print(f"\nEvent Counts:")
        print(f"  Herbivore Conceptions:      {m.counts['conceptions_H']:>8}")
        print(f"  Herbivore Births:           {m.counts['births_H']:>8}")
        print(f"  Herbivore Deaths:           {m.counts['deaths_H']:>8}")
        print(f"  Herbivore Conflicts:        {m.counts['conflicts_H']:>8}")
        print(f"\n  Carnivore Conceptions:      {m.counts['conceptions_C']:>8}")
        print(f"  Carnivore Births:           {m.counts['births_C']:>8}")
        print(f"  Carnivore Deaths:           {m.counts['deaths_C']:>8}")
        print(f"  Carnivore Conflicts:        {m.counts['conflicts_C']:>8}")
        print(f"\n  Predation Events:           {m.counts['predations']:>8}")
        print("="*60)

    def plot_results(self, ci_data: Optional[Dict[str, tuple]] = None, num_runs: int = 1):
        times = self.metrics.history_time
        
        # Aggregation: Total H = H_M + H_F + H_F_preg
        h_total = np.array(self.metrics.history_H_M) + np.array(self.metrics.history_H_F) + np.array(self.metrics.history_H_F_preg)
        
        # Aggregation: Total C = C_M + C_F + C_F_preg
        c_total = np.array(self.metrics.history_C_M) + np.array(self.metrics.history_C_F) + np.array(self.metrics.history_C_F_preg)

        plt.figure(figsize=(12, 6))
        
        # Population Dynamics with CI bands
        
        if num_runs > 1:
            plt.plot(times, h_total, label='Herbivores (Last Run)', color='green', linewidth=1.5)
            plt.plot(times, c_total, label='Carnivores (Last Run)', color='red', linewidth=1.5)
        else:
            plt.plot(times, h_total, label='Herbivores', color='green', linewidth=1.5)
            plt.plot(times, c_total, label='Carnivores', color='red', linewidth=1.5)
        
        # Add CI bands if available
        if ci_data:
            if 'avg_H_total' in ci_data:
                h_lower, h_upper, _ = ci_data['avg_H_total']
                run_mean = (self.metrics.area_under_herbivore_females + self.metrics.area_under_herbivore_males + self.metrics.area_under_herbivore_females_preg) / self.current_time
                plt.axhline(y=run_mean, color='green', linestyle=':', alpha=0.7, linewidth=1, 
                           label=f'H Mean: {run_mean:.1f}')
                plt.axhspan(h_lower, h_upper, color='green', alpha=0.15, 
                           label=f'H 95% CI: [{h_lower:.1f}, {h_upper:.1f}]')
            
            if 'avg_C_total' in ci_data:
                c_lower, c_upper, _ = ci_data['avg_C_total']
                run_mean = (self.metrics.area_under_carnivore_females + self.metrics.area_under_carnivore_males + self.metrics.area_under_carnivore_females_preg) / self.current_time
                plt.axhline(y=run_mean, color='red', linestyle=':', alpha=0.7, linewidth=1, 
                           label=f'C Mean: {run_mean:.1f}')
                plt.axhspan(c_lower, c_upper, color='red', alpha=0.15, 
                           label=f'C 95% CI: [{c_lower:.1f}, {c_upper:.1f}]')
        
        # Plot herbivore threshold
        if self.params and 'H_threshold' in self.params:
            plt.axhline(y=self.params['H_threshold'], color='gray', linestyle='--', 
                       alpha=0.3, label='H Threshold')
        
        plt.xlabel('Time')
        plt.ylabel('Population')
        if num_runs > 1:
            plt.title(f'Population Dynamics (95% CI, n={num_runs} runs)')
        else:
            plt.title('Population Dynamics')
        plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.savefig('population_dynamics.png', dpi=300) # Save if needed
        plt.show()

# --------------------------------------------------
# MULTIPLE RUNS WITH CONFIDENCE INTERVALS
# --------------------------------------------------
def run_multiple_simulations(num_runs: int, base_seed: int, t_max: float, 
                             initial_pop: Dict[str, int], params: Dict[str, float],
                             show_plots: bool = True, verbose: bool = False):
    """
    Run multiple independent simulations with different seeds and compute confidence intervals.
    
    Args:
        num_runs: Number of independent simulation runs
        base_seed: Base seed value (each run uses base_seed + i)
        t_max: Simulation time
        initial_pop: Initial population dictionary
        params: Parameter dictionary
        show_plots: Whether to show plots for the last run
        verbose: Whether to print detailed statistics for each run
    
    Returns:
        Dictionary with CI results for key metrics
    """
    print(f"\n{'='*80}")
    print(f"RUNNING {num_runs} INDEPENDENT SIMULATIONS")
    print(f"{'='*80}\n")
    
    ci_calculators = {
        'avg_H_M': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_H_F': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_H_F_preg': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_H_total': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_C_M': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_C_F': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_C_F_preg': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_C_total': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
    }
    
    all_metrics = []
    last_sim = None
    
    # Run simulations
    for i in range(num_runs):
        seed = base_seed + i
        np.random.seed(seed)
        
        print(f"Run {i+1}/{num_runs} (seed={seed})...", end=" ")
        
        sim = SimulationEngine(t_max=t_max, initial_pop=initial_pop, params=params)
        sim.event_loop()
        
        total_time = sim.current_time
        if total_time > 0:
            avg_H_M = sim.metrics.area_under_herbivore_males / total_time
            avg_H_F = sim.metrics.area_under_herbivore_females / total_time
            avg_H_F_preg = sim.metrics.area_under_herbivore_females_preg / total_time
            avg_H_total = avg_H_M + avg_H_F + avg_H_F_preg
            
            avg_C_M = sim.metrics.area_under_carnivore_males / total_time
            avg_C_F = sim.metrics.area_under_carnivore_females / total_time
            avg_C_F_preg = sim.metrics.area_under_carnivore_females_preg / total_time
            avg_C_total = avg_C_M + avg_C_F + avg_C_F_preg
        else:
            avg_H_M = avg_H_F = avg_H_F_preg = avg_H_total = 0.0
            avg_C_M = avg_C_F = avg_C_F_preg = avg_C_total = 0.0
        
        # Collect metrics
        metrics = {
            'avg_H_M': avg_H_M,
            'avg_H_F': avg_H_F,
            'avg_H_F_preg': avg_H_F_preg,
            'avg_H_total': avg_H_total,
            'avg_C_M': avg_C_M,
            'avg_C_F': avg_C_F,
            'avg_C_F_preg': avg_C_F_preg,
            'avg_C_total': avg_C_total,
        }
        all_metrics.append(metrics)
        
        # Add to CI calculators
        for metric_name, value in metrics.items():
            ci_calculators[metric_name].add_data_point(value)
        
        print(f"H={sim.H_M + sim.H_F + sim.H_F_preg}, C={sim.C_M + sim.C_F + sim.C_F_preg}")
        
        if verbose:
            sim.print_statistics()
        
        last_sim = sim

    # Compute confidence intervals
    print(f"\n{'='*80}")
    print(f"CONFIDENCE INTERVAL RESULTS (95% confidence, {num_runs} runs)")
    print(f"{'='*80}\n")
    
    ci_results = {}
    
    print("--- TIME-WEIGHTED AVERAGES ---\n")
    for metric_name in ['avg_H_M', 'avg_H_F', 'avg_H_F_preg', 'avg_H_total',
                        'avg_C_M', 'avg_C_F', 'avg_C_F_preg', 'avg_C_total']:
        calculator = ci_calculators[metric_name]
        if calculator.has_enough_data():
            result = calculator.compute_interval()
            if result:
                final, (lower, upper) = result
                status = "✓ FINAL" if final else "→ CONVERGING"
                ci_results[metric_name] = (lower, upper, calculator.average)
                
                # Format metric name for display
                display_name = metric_name.replace('avg_', 'Avg ').replace('_', ' ').title()
                print(f"{display_name:25} {status:15} Mean: {calculator.average:7.2f}  CI: [{lower:7.2f}, {upper:7.2f}]  Width: {upper-lower:6.2f}")
            else:
                print(f"{metric_name:25} {'✗ COMPUTATION FAILED':20}")
        else:
            print(f"{metric_name:25} {'✗ INSUFFICIENT DATA':20}")
    
    print(f"\n{'='*80}\n")
    
    # Summary statistics
    print("Summary Statistics Across All Runs:")
    print(f"{'='*80}")
    for metric in ['avg_H_M', 'avg_H_F', 'avg_H_F_preg', 'avg_H_total',
                   'avg_C_M', 'avg_C_F', 'avg_C_F_preg', 'avg_C_total']:
        values = [m[metric] for m in all_metrics]
        display_name = metric.replace('avg_', 'Avg ').replace('_', ' ').title()
        print(f"{display_name:25} Mean: {np.mean(values):7.2f}  Std: {np.std(values):6.2f}  Min: {np.min(values):7.2f}  Max: {np.max(values):7.2f}")
    print(f"{'='*80}")

    # Show plots for last run if requested
    if show_plots and last_sim:
        print("\nGenerating plots for last run...")
        last_sim.plot_results(ci_data=ci_results, num_runs=num_runs)
    
    return ci_results, all_metrics

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    # --- CONFIGURATION ---
    MODE = "multiple"  # Change to "single" for single detailed run with plots
    
    # Initial Populations
    initial_pop = {'H_M': 75, 'H_F': 75, 'C_M': 25, 'C_F': 25}
    
    # Model Parameters (OPTIMIZED FOR MAXIMUM STABILITY)
    parameters = {
        # Reproduction Rates (per female)
        'r_H': 0.5,           # Herbivore conception rate
        'r_C': 0.4,           # Carnivore conception rate
        
        # Gestation periods (mean time in exponential distribution)
        'gestation_H': 6.0,   # Herbivore gestation time
        'gestation_C': 5.0,   # Carnivore gestation time
        
        # Natural Mortality Rates (per individual)
        'm_H': 0.02,          # Herbivore mortality
        'm_C': 0.03,          # Carnivore mortality
        
        # Predation efficiency (interaction rate)
        'pred': 0.0007,       # Predation rate
        
        # Conflict rates (density dependent)
        'conflict_H': 0.00005,# Herbivore conflict rate
        'conflict_C': 0.0005, # Carnivore conflict rate
        
        # Competition thresholds
        'H_threshold': 200,       # Herbivore carrying capacity
        'HC_ratio_threshold': 3.0 # Minimum H/C ratio for carnivore viability
    }
    
    if MODE == "single":
        # Single detailed run with plots
        print("Running Single Simulation...")
        np.random.seed(42)
        
        sim_engine = SimulationEngine(t_max=1000.0, initial_pop=initial_pop, params=parameters)
        sim_engine.event_loop()
        sim_engine.print_statistics()
        
        print("Plotting results...")
        sim_engine.plot_results(ci_data=None, num_runs=1)
        print("Done.")
    
    elif MODE == "multiple":
        # Multiple runs with confidence intervals
        NUM_RUNS = 20        # Number of independent replications
        BASE_SEED = 42       # Base seed value
        T_MAX = 1000.0        # Simulation time (longer for steady-state analysis)
        SHOW_PLOTS = True    # Show plots for last run
        VERBOSE = False      # Set to True to see detailed stats for each run
        
        ci_results, all_metrics = run_multiple_simulations(
            num_runs=NUM_RUNS,
            base_seed=BASE_SEED,
            t_max=T_MAX,
            initial_pop=initial_pop,
            params=parameters,
            show_plots=SHOW_PLOTS,
            verbose=VERBOSE
        )