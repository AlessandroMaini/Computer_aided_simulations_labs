import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict
import sys
import os
from confidence_interval import ConfidenceInterval

# --------------------------------------------------
# EVENT DEFINITION
# --------------------------------------------------
class EventType(Enum):
    # Base Vital Dynamics
    BIRTH_H_M = 1
    BIRTH_H_F = 2
    BIRTH_C_M = 3
    BIRTH_C_F = 4
    
    DEATH_H_M = 5
    DEATH_H_F = 6
    DEATH_C_M = 7
    DEATH_C_F = 8
    
    # Interactions
    PREDATION = 9          # C eats H
    CONFLICT_H_M = 10      # H_M vs H_M fight
    CONFLICT_H_F = 11      # H_F vs H_F fight
    CONFLICT_C_M = 12      # C_M vs C_M fight
    CONFLICT_C_F = 13      # C_F vs C_F fight

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
        self.history_C_M = []
        self.history_C_F = []
        
        self.counts = {
            'births_H': 0, 'births_C': 0,
            'deaths_H': 0, 'deaths_C': 0,
            'predations': 0,
            'conflicts_H': 0, 'conflicts_C': 0
        }

        # Time-weighted statistics
        self.last_time = 0.0
        self.area_under_herbivore_males = 0.0
        self.area_under_herbivore_females = 0.0
        self.area_under_carnivore_males = 0.0
        self.area_under_carnivore_females = 0.0

    def record_state(self, t, H_M, H_F, C_M, C_F):
        self.history_time.append(t)
        self.history_H_M.append(H_M)
        self.history_H_F.append(H_F)
        self.history_C_M.append(C_M)
        self.history_C_F.append(C_F)

    def update_time_weighted_stats(self, current_time: float, num_H_M: int, num_H_F: int, num_C_M: int, num_C_F: int):
        """Update time-weighted statistics."""
        time_diff = current_time - self.last_time
        self.area_under_herbivore_males += num_H_M * time_diff
        self.area_under_herbivore_females += num_H_F * time_diff
        self.area_under_carnivore_males += num_C_M * time_diff
        self.area_under_carnivore_females += num_C_F * time_diff
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
        self.C_M = initial_pop['C_M']
        self.C_F = initial_pop['C_F']
        
        # PARAMETERS
        self.params = params
        self.metrics = Metrics()
        self.current_time = 0.0

    def calculate_rates(self) -> Dict[EventType, float]:
        """
        Calculates the Rate for every possible transition
        based on the current state and Competition Thresholds.
        """
        rates = {}
        
        # --- 1. CURRENT TOTALS ---
        H_total = self.H_M + self.H_F
        C_total = self.C_M + self.C_F
        
        # Avoid division by zero
        ratio_HC = H_total / C_total if C_total > 0 else float('inf')

        # --- 2. APPLY THRESHOLD LOGIC (Resource Competition) ---
        
        # Herbivore Competition - Logistic decay
        # "If herbivore population exceeds threshold... reduce fertility, increase mortality"
        # Steepness parameter for logistic transition
        k_H = 0.1  # Adjust for smoother/sharper transition
        h_fert_factor = 1.0 / (1.0 + np.exp(k_H * (H_total - self.params['H_threshold'])))
        h_mort_factor = 1.0 + 1.0 / (1.0 + np.exp(-k_H * (H_total - self.params['H_threshold'])))
            
        # Carnivore Competition - Logistic decay        
        # Steepness parameter for logistic transition
        k_C = 2.0  # Adjust for smoother/sharper transition
        c_fert_factor = 1.0 / (1.0 + np.exp(-k_C * (ratio_HC - self.params['HC_ratio_threshold'])))
        c_mort_factor = 1.0 + 1.0 / (1.0 + np.exp(k_C * (ratio_HC - self.params['HC_ratio_threshold'])))

        # --- 3. CALCULATE RATES (Mass Action & Constant per Capita) ---
        
        # REPRODUCTION (Rate proportional to pairs)
        # Simplified: Rate = r * Females (assuming Males exist to mate)
        can_mate_H = 1 if self.H_M > 0 else 0
        can_mate_C = 1 if self.C_M > 0 else 0
        
        # Same probability of birth for both males and females
        rates[EventType.BIRTH_H_M] = 0.5 * self.params['r_H'] * h_fert_factor * self.H_F * can_mate_H
        rates[EventType.BIRTH_H_F] = 0.5 * self.params['r_H'] * h_fert_factor * self.H_F * can_mate_H
        rates[EventType.BIRTH_C_M] = 0.5 * self.params['r_C'] * c_fert_factor * self.C_F * can_mate_C
        rates[EventType.BIRTH_C_F] = 0.5 * self.params['r_C'] * c_fert_factor * self.C_F * can_mate_C

        # NATURAL MORTALITY (Rate proportional to N)
        rates[EventType.DEATH_H_M] = self.params['m_H'] * h_mort_factor * self.H_M
        rates[EventType.DEATH_H_F] = self.params['m_H'] * h_mort_factor * self.H_F
        rates[EventType.DEATH_C_M] = self.params['m_C'] * c_mort_factor * self.C_M
        rates[EventType.DEATH_C_F] = self.params['m_C'] * c_mort_factor * self.C_F

        # PREDATION (Rate proportional to H * C)
        # Predators encounter Prey based on density
        predation_rate_total = self.params['pred'] * H_total * C_total
        rates[EventType.PREDATION] = predation_rate_total # We will handle sex selection in handler

        # CONFLICT (Rate proportional to N * (N-1))
        # Intraspecific conflict scales with density squared (encounters)
        rates[EventType.CONFLICT_H_M] = self.params['conflict_H'] * self.H_M * (self.H_M - 1)
        rates[EventType.CONFLICT_H_F] = self.params['conflict_H'] * self.H_F * (self.H_F - 1)
        rates[EventType.CONFLICT_C_M] = self.params['conflict_C'] * self.C_M * (self.C_M - 1)
        rates[EventType.CONFLICT_C_F] = self.params['conflict_C'] * self.C_F * (self.C_F - 1)

        return rates

    def step(self):
        """Performs one Simulation Step."""
        
        # 1. Calculate all reaction rates
        rates = self.calculate_rates()
        total_rate = sum(rates.values())

        # Check for extinction or stasis
        if total_rate <= 0:
            return False

        # 2. Determine Time to Next Event (Exponential Distribution)
        # This satisfies the Markov Property (Memorylessness)
        dt = np.random.exponential(1.0 / total_rate)
        
        if self.current_time + dt > self.t_max:
            self.current_time = self.t_max
            return False
        
        self.current_time += dt

        # 3. Determine Which Event Occurs (Weighted probability)
        # Create cumulative probability distribution
        event_types = list(rates.keys())
        event_probs = np.array(list(rates.values())) / total_rate
        
        # Select event
        selected_type = np.random.choice(event_types, p=event_probs)
        
        # 4. Schedule and Process (Immediate execution)
        event = Event(selected_type, self.current_time)
        self.handle_event(event)
        
        return True

    def handle_event(self, event: Event):
        """Updates the Macroscopic State based on the event."""
        
        t = event.event_type
        
        # --- BIRTHS ---
        if t == EventType.BIRTH_H_M:
            self.H_M += 1
            self.metrics.counts['births_H'] += 1
        elif t == EventType.BIRTH_H_F:
            self.H_F += 1
            self.metrics.counts['births_H'] += 1
        elif t == EventType.BIRTH_C_M:
            self.C_M += 1
            self.metrics.counts['births_C'] += 1
        elif t == EventType.BIRTH_C_F:
            self.C_F += 1
            self.metrics.counts['births_C'] += 1
            
        # --- NATURAL DEATHS ---
        elif t == EventType.DEATH_H_M and self.H_M > 0:
            self.H_M -= 1
            self.metrics.counts['deaths_H'] += 1
        elif t == EventType.DEATH_H_F and self.H_F > 0:
            self.H_F -= 1
            self.metrics.counts['deaths_H'] += 1
        elif t == EventType.DEATH_C_M and self.C_M > 0:
            self.C_M -= 1
            self.metrics.counts['deaths_C'] += 1
        elif t == EventType.DEATH_C_F and self.C_F > 0:
            self.C_F -= 1
            self.metrics.counts['deaths_C'] += 1
            
        # --- PREDATION (C eats H) ---
        elif t == EventType.PREDATION:
            # Decide if Male or Female H is eaten based on proportion
            total_H = self.H_M + self.H_F
            if total_H > 0:
                if np.random.rand() < (self.H_M / total_H):
                    self.H_M -= 1
                else:
                    self.H_F -= 1
                self.metrics.counts['predations'] += 1
                
        # --- CONFLICTS (One dies) ---
        elif t == EventType.CONFLICT_H_M and self.H_M > 1:
            self.H_M -= 1 # One killed
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
        self.metrics.record_state(0, self.H_M, self.H_F, self.C_M, self.C_F)
        
        while self.current_time < self.t_max:
            active = self.step()
            self.metrics.record_state(self.current_time, self.H_M, self.H_F, self.C_M, self.C_F)
            # Update time weighted statistics
            self.metrics.update_time_weighted_stats(self.current_time, self.H_M, self.H_F, self.C_M, self.C_F)
            if not active:
                break

    def print_statistics(self):
        """Print simulation statistics including counts and time-weighted averages."""
        m = self.metrics
        total_time = self.current_time
        
        print("\n" + "="*60)
        print("SIMULATION STATISTICS")
        print("="*60)
        
        print(f"\nSimulation Time: {total_time:.2f}")
        
        # Total Counts
        print("\n--- Event Counts ---")
        print(f"Herbivore Births:     {m.counts['births_H']:>8}")
        print(f"Carnivore Births:     {m.counts['births_C']:>8}")
        print(f"Herbivore Deaths:     {m.counts['deaths_H']:>8}")
        print(f"Carnivore Deaths:     {m.counts['deaths_C']:>8}")
        print(f"Predation Events:     {m.counts['predations']:>8}")
        print(f"Herbivore Conflicts:  {m.counts['conflicts_H']:>8}")
        print(f"Carnivore Conflicts:  {m.counts['conflicts_C']:>8}")
        
        # Time-Weighted Averages
        print("\n--- Time-Weighted Averages ---")
        if total_time > 0:
            avg_H_M = m.area_under_herbivore_males / total_time
            avg_H_F = m.area_under_herbivore_females / total_time
            avg_C_M = m.area_under_carnivore_males / total_time
            avg_C_F = m.area_under_carnivore_females / total_time
            avg_H_total = avg_H_M + avg_H_F
            avg_C_total = avg_C_M + avg_C_F
            
            print(f"Herbivore Males:      {avg_H_M:>8.2f}")
            print(f"Herbivore Females:    {avg_H_F:>8.2f}")
            print(f"Herbivore Total:      {avg_H_total:>8.2f}")
            print(f"Carnivore Males:      {avg_C_M:>8.2f}")
            print(f"Carnivore Females:    {avg_C_F:>8.2f}")
            print(f"Carnivore Total:      {avg_C_total:>8.2f}")
            
            if avg_C_total > 0:
                print(f"H/C Ratio (avg):      {avg_H_total/avg_C_total:>8.2f}")
        
        # Final State
        print("\n--- Final State ---")
        print(f"Herbivore Males:      {self.H_M:>8}")
        print(f"Herbivore Females:    {self.H_F:>8}")
        print(f"Herbivore Total:      {self.H_M + self.H_F:>8}")
        print(f"Carnivore Males:      {self.C_M:>8}")
        print(f"Carnivore Females:    {self.C_F:>8}")
        print(f"Carnivore Total:      {self.C_M + self.C_F:>8}")
        
        print("="*60 + "\n")

# --------------------------------------------------
# MULTIPLE RUNS WITH CONFIDENCE INTERVALS
# --------------------------------------------------
def run_multiple_simulations(num_runs: int, base_seed: int, t_max: float, 
                             initial_pop: Dict[str, int], params: Dict[str, float],
                             show_plots: bool = False, verbose: bool = False):
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
    
    # Initialize confidence interval calculators for key metrics
    ci_calculators = {
        'avg_H_M': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_H_F': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_C_M': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_C_F': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_H_total': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'avg_C_total': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
    }
    
    all_metrics = []
    last_sim = None
    
    # Run simulations
    for i in range(num_runs):
        seed = base_seed + i
        np.random.seed(seed)
        
        print(f"Run {i+1}/{num_runs} (seed={seed})...", end=" ")
        
        # Create and run simulation
        sim = SimulationEngine(t_max=t_max, initial_pop=initial_pop, params=params)
        sim.event_loop()
        
        # Calculate time-weighted averages
        total_time = sim.current_time
        if total_time > 0:
            avg_H_M = sim.metrics.area_under_herbivore_males / total_time
            avg_H_F = sim.metrics.area_under_herbivore_females / total_time
            avg_C_M = sim.metrics.area_under_carnivore_males / total_time
            avg_C_F = sim.metrics.area_under_carnivore_females / total_time
        else:
            avg_H_M = avg_H_F = avg_C_M = avg_C_F = 0.0
        
        # Collect metrics
        metrics = {
            'avg_H_M': avg_H_M,
            'avg_H_F': avg_H_F,
            'avg_C_M': avg_C_M,
            'avg_C_F': avg_C_F,
            'avg_H_total': avg_H_M + avg_H_F,
            'avg_C_total': avg_C_M + avg_C_F,
        }
        all_metrics.append(metrics)
        
        # Add to CI calculators
        for metric_name, value in metrics.items():
            ci_calculators[metric_name].add_data_point(value)
        
        print(f"H={sim.H_M + sim.H_F}, C={sim.C_M + sim.C_F}")
        
        if verbose:
            sim.print_statistics()
        
        last_sim = sim
    
    # Compute confidence intervals
    print(f"\n{'='*80}")
    print(f"CONFIDENCE INTERVAL RESULTS (95% confidence, {num_runs} runs)")
    print(f"{'='*80}\n")
    
    ci_results = {}
    
    print("--- TIME-WEIGHTED AVERAGES ---\n")
    for metric_name in ['avg_H_M', 'avg_H_F', 'avg_H_total', 'avg_C_M', 'avg_C_F', 'avg_C_total']:
        calculator = ci_calculators[metric_name]
        if calculator.has_enough_data():
            result = calculator.compute_interval()
            if result:
                final, (lower, upper) = result
                status = "✓ FINAL" if final else "→ CONVERGING"
                ci_results[metric_name] = (lower, upper, calculator.average)
                
                # Format metric name for display
                display_name = metric_name.replace('avg_', 'Avg ').replace('_', ' ').title()
                print(f"{display_name:20} {status:15} Mean: {calculator.average:7.2f}  CI: [{lower:7.2f}, {upper:7.2f}]  Width: {upper-lower:6.2f}")
            else:
                print(f"{metric_name:25} {'✗ COMPUTATION FAILED':20}")
        else:
            print(f"{metric_name:25} {'✗ INSUFFICIENT DATA':20}")
    
    print(f"\n{'='*80}\n")
    
    # Summary statistics
    print("Summary Statistics Across All Runs:")
    print(f"{'='*80}")
    for metric in ['avg_H_M', 'avg_H_F', 'avg_H_total', 'avg_C_M', 'avg_C_F', 'avg_C_total']:
        values = [m[metric] for m in all_metrics]
        display_name = metric.replace('avg_', 'Avg ').replace('_', ' ').title()
        print(f"{display_name:20} Mean: {np.mean(values):7.2f}  Std: {np.std(values):6.2f}  Min: {np.min(values):7.2f}  Max: {np.max(values):7.2f}")
    print(f"{'='*80}")
    
    # Show plots for last run if requested
    if show_plots and last_sim:
        print("\nGenerating plots for last run...")
        m = last_sim.metrics
        t = m.history_time
        H_total = np.array(m.history_H_M) + np.array(m.history_H_F)
        C_total = np.array(m.history_C_M) + np.array(m.history_C_F)
        
        # Extract confidence intervals for plotting
        h_total_ci = ci_results.get('avg_H_total', None)
        c_total_ci = ci_results.get('avg_C_total', None)
        
        plt.figure(figsize=(12, 6))
        
        # 1. Population Dynamics with CI bands
        plt.subplot(2, 1, 1)
        plt.plot(t, H_total, label='Herbivores (Last Run)', color='green', linewidth=1.5)
        plt.plot(t, C_total, label='Carnivores (Last Run)', color='red', linewidth=1.5)
        
        # Add CI bands if available
        if h_total_ci:
            h_lower, h_upper, h_mean = h_total_ci
            plt.axhline(y=h_mean, color='green', linestyle=':', alpha=0.7, linewidth=1, label=f'H Mean: {h_mean:.1f}')
            plt.axhspan(h_lower, h_upper, color='green', alpha=0.15, label=f'H 95% CI: [{h_lower:.1f}, {h_upper:.1f}]')
        
        if c_total_ci:
            c_lower, c_upper, c_mean = c_total_ci
            plt.axhline(y=c_mean, color='red', linestyle=':', alpha=0.7, linewidth=1, label=f'C Mean: {c_mean:.1f}')
            plt.axhspan(c_lower, c_upper, color='red', alpha=0.15, label=f'C 95% CI: [{c_lower:.1f}, {c_upper:.1f}]')
        
        plt.axhline(y=params['H_threshold'], color='gray', linestyle='--', alpha=0.3, label='H Threshold')
        plt.ylabel('Population')
        plt.title(f'Macroscopic CTMC Dynamics with 95% CI (n={num_runs} runs)')
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 2. Phase Portrait (H vs C)
        plt.subplot(2, 1, 2)
        plt.plot(H_total, C_total, color='blue', alpha=0.6)
        plt.plot(H_total[0], C_total[0], 'go', label='Start')
        plt.plot(H_total[-1], C_total[-1], 'rx', label='End')
        plt.xlabel('Herbivores')
        plt.ylabel('Carnivores')
        plt.title('Phase Portrait (H vs C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return ci_results, all_metrics

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    # Choose mode: single run or multiple runs with CI
    MODE = "multiple"  # Change to "single" for single detailed run with plots
    
    # Initial Populations
    initial_pop = {'H_M': 50, 'H_F': 50, 'C_M': 10, 'C_F': 10}
    
    # Model Parameters
    parameters = {
        # Reproduction Rates (per female)
        'r_H': 0.6,
        'r_C': 0.4,
        
        # Natural Mortality Rates (per individual)
        'm_H': 0.1,
        'm_C': 0.1,
        
        # Predation efficiency (interaction rate)
        'pred': 0.003,
        
        # Conflict rates (density dependent)
        'conflict_H': 0.0001,
        'conflict_C': 0.002,
        
        # COMPETITION THRESHOLDS (As required by prompt)
        'H_threshold': 200,       # If H > 200, H struggle
        'HC_ratio_threshold': 3.0 # If H/C < 3, C struggle (scarce food)
    }
    
    if MODE == "single":
        # Single detailed run with plots
        np.random.seed(0)
        
        sim = SimulationEngine(t_max=100.0, initial_pop=initial_pop, params=parameters)
        sim.event_loop()
        sim.print_statistics()
        
        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------
        m = sim.metrics
        t = m.history_time
        H_total = np.array(m.history_H_M) + np.array(m.history_H_F)
        C_total = np.array(m.history_C_M) + np.array(m.history_C_F)
        
        plt.figure(figsize=(12, 6))
        
        # 1. Population Dynamics
        plt.subplot(2, 1, 1)
        plt.plot(t, H_total, label='Herbivores', color='green')
        plt.plot(t, C_total, label='Carnivores', color='red')
        plt.axhline(y=parameters['H_threshold'], color='green', linestyle='--', alpha=0.3, label='H Threshold')
        plt.ylabel('Population')
        plt.title('Macroscopic CTMC Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Phase Portrait (H vs C) - Good for Ergodicity Analysis
        plt.subplot(2, 1, 2)
        plt.plot(H_total, C_total, color='blue', alpha=0.6)
        plt.plot(H_total[0], C_total[0], 'go', label='Start')
        plt.plot(H_total[-1], C_total[-1], 'rx', label='End')
        plt.xlabel('Herbivores')
        plt.ylabel('Carnivores')
        plt.title('Phase Portrait (H vs C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    elif MODE == "multiple":
        # Multiple runs with confidence intervals
        NUM_RUNS = 30  # Number of independent replications
        BASE_SEED = 0  # Base seed value
        T_MAX = 100.0  # Simulation time
        SHOW_PLOTS = True  # Show plots for last run
        VERBOSE = False  # Set to True to see detailed stats for each run
        
        ci_results, all_metrics = run_multiple_simulations(
            num_runs=NUM_RUNS,
            base_seed=BASE_SEED,
            t_max=T_MAX,
            initial_pop=initial_pop,
            params=parameters,
            show_plots=SHOW_PLOTS,
            verbose=VERBOSE
        )