"""
Airport Departure Simulation - Test Scenarios Configuration

This file contains different realistic scenarios to test the airport system
performance under various operational conditions.

Each scenario represents a different type of airport operation or stress condition.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import simulation engine
sys.path.append(os.path.dirname(__file__))
from airport_departure import SimulationEngine

# ==============================================================================
# SCENARIO 1: LOW STRESS - Balanced Load and Adequate Resources
# ==============================================================================
SCENARIO_1_LOW_STRESS = {
    "name": "Low Stress Configuration",
    "description": "Off-peak operations: Low flight frequency (2.7 flights/hour), over-staffed resources, and widely dispersed passenger arrivals ensure smooth processing with minimal queuing",
    
    # Temporal parameters
    "SIMULATION_TIME": 24 * 60,  # 24 hours
    
    # Flight scheduling - MODERATE LOAD
    "FLIGHT_FREQUENCY": 24,  # One flight every 24 minutes (more relaxed)
    "AVG_PASSENGERS_PER_FLIGHT": 100.0,  # Standard narrow-body aircraft
    
    # Resource allocation - GENEROUS STAFFING
    "NUM_CASHIER_SERVERS_LAND": 19,  # Extra capacity (+2 vs baseline)
    "NUM_CASHIER_SERVERS_AIR": 10,    # Extra capacity (+2 vs baseline)
    "NUM_SECURITY_SERVERS": 10,        # Extra capacity (+2 vs baseline)
    "NUM_BOARDING_SERVERS": 3,         # Extra capacity (+1 vs baseline)
    
    # Service rates (minutes)
    "CASHIER_SERVICE_RATE": 2.0,
    "SECURITY_SERVICE_RATE": 1.5,
    "BOARDING_SERVICE_RATE": 0.5,
    
    # Passenger behavior
    "BUYING_PROB": 0.8,
    "AVG_COMPANIONS": 1.5,
    "AVG_LANDSIDE_DWELL_TIME": 30.0,
    "AVG_AIRSIDE_DWELL_TIME": 15.0,
    
    # Arrival patterns - WELL DISTRIBUTED
    "AVG_ARRIVAL_TIME_BEFORE_FLIGHT": 120.0,
    "STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT": 35.0,  # Very high variance = maximally spread out arrivals
    
    # Time constraints
    "SECURITY_DEADLINE": 40,
    "BOARDING_WINDOW_START": 45,
    "BOARDING_DEADLINE": 20,
    
    # Priority distribution
    "PRIORITY_PROBS": [0.7, 0.2, 0.1],  # [Economy, Business, First]
}


# ==============================================================================
# SCENARIO 2: MEDIUM STRESS - Increased Load with Same Resources
# ==============================================================================
SCENARIO_2_MEDIUM_STRESS = {
    "name": "Medium Stress Configuration",
    "description": "Normal operations baseline: Balanced 3 flights/hour with standard staffing levels. Matches airport_departure.py reference configuration with acceptable performance under typical conditions",
    
    # Temporal parameters
    "SIMULATION_TIME": 24 * 60,  # 24 hours (matches airport_departure.py)
    
    # Flight scheduling - MATCHES HARDCODED CONFIG
    "FLIGHT_FREQUENCY": 20,  # Matches airport_departure.py
    "AVG_PASSENGERS_PER_FLIGHT": 100.0,  # Matches airport_departure.py
    
    # Resource allocation - MATCHES HARDCODED CONFIG
    "NUM_CASHIER_SERVERS_LAND": 17,  # Matches airport_departure.py
    "NUM_CASHIER_SERVERS_AIR": 8,     # Matches airport_departure.py
    "NUM_SECURITY_SERVERS": 8,        # Matches airport_departure.py
    "NUM_BOARDING_SERVERS": 2,        # Matches airport_departure.py
    
    # Service rates (minutes) - MATCHES HARDCODED CONFIG
    "CASHIER_SERVICE_RATE": 2.0,   # Matches airport_departure.py
    "SECURITY_SERVICE_RATE": 1.5,  # Matches airport_departure.py
    "BOARDING_SERVICE_RATE": 0.5,  # Matches airport_departure.py
    
    # Passenger behavior - MATCHES HARDCODED CONFIG
    "BUYING_PROB": 0.8,              # Matches airport_departure.py
    "AVG_COMPANIONS": 1.5,           # Matches airport_departure.py
    "AVG_LANDSIDE_DWELL_TIME": 30.0, # Matches airport_departure.py
    "AVG_AIRSIDE_DWELL_TIME": 15.0,  # Matches airport_departure.py
    
    # Arrival patterns - MATCHES HARDCODED CONFIG
    "AVG_ARRIVAL_TIME_BEFORE_FLIGHT": 120.0,    # Matches airport_departure.py
    "STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT": 30.0,  # Matches airport_departure.py
    
    # Time constraints
    "SECURITY_DEADLINE": 40,
    "BOARDING_WINDOW_START": 45,
    "BOARDING_DEADLINE": 20,
    
    # Priority distribution
    "PRIORITY_PROBS": [0.7, 0.2, 0.1],
}


# ==============================================================================
# SCENARIO 3: HIGH STRESS - High Frequency with Synchronized Arrivals
# ==============================================================================
SCENARIO_3_HIGH_STRESS = {
    "name": "High Stress Configuration",
    "description": "Peak-hour rush: Elevated frequency (3.2 flights/hour), under-staffed by 12%, and moderately synchronized arrivals create sustained pressure on security checkpoint with elevated drop rates",
    
    # Temporal parameters
    "SIMULATION_TIME": 24 * 60,  # 24 hours
    
    # Flight scheduling - MODERATE-HIGH FREQUENCY
    "FLIGHT_FREQUENCY": 19,  # Higher flight frequency (more moderate)
    "AVG_PASSENGERS_PER_FLIGHT": 100.0,  # Same aircraft capacity
    
    # Resource allocation - REDUCED CAPACITY
    "NUM_CASHIER_SERVERS_LAND": 16,  # Reduced (-1 vs baseline)
    "NUM_CASHIER_SERVERS_AIR": 7,     # Reduced (-1 vs baseline)
    "NUM_SECURITY_SERVERS": 7,         # Reduced - significant bottleneck (-1 vs baseline)
    "NUM_BOARDING_SERVERS": 2,
    
    # Service rates (minutes) - SAME
    "CASHIER_SERVICE_RATE": 2.0,
    "SECURITY_SERVICE_RATE": 1.5,
    "BOARDING_SERVICE_RATE": 0.5,
    
    # Passenger behavior - SAME
    "BUYING_PROB": 0.8,
    "AVG_COMPANIONS": 1.5,
    "AVG_LANDSIDE_DWELL_TIME": 30.0,
    "AVG_AIRSIDE_DWELL_TIME": 15.0,
    
    # Arrival patterns - MODERATELY SYNCHRONIZED
    "AVG_ARRIVAL_TIME_BEFORE_FLIGHT": 120.0,
    "STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT": 27.0,  # Moderate synchronization = arrival waves
    
    # Time constraints
    "SECURITY_DEADLINE": 40,
    "BOARDING_WINDOW_START": 45,
    "BOARDING_DEADLINE": 20,
    
    # Priority distribution
    "PRIORITY_PROBS": [0.7, 0.2, 0.1],
}


# ==============================================================================
# SCENARIO 4: EXTREME STRESS - Maximum Load with Significant Resource Constraints
# ==============================================================================
SCENARIO_4_EXTREME_STRESS = {
    "name": "Extreme Stress Configuration",
    "description": "Crisis scenario: Maximum frequency (3.4 flights/hour), severely under-staffed by 12%, and highly synchronized arrivals create perfect storm conditions with critical passenger drop rates at system limits",
    
    # Temporal parameters
    "SIMULATION_TIME": 24 * 60,  # 24 hours
    
    # Flight scheduling - VERY HIGH FREQUENCY
    "FLIGHT_FREQUENCY": 18,  # Maximum frequency (more aggressive than High)
    "AVG_PASSENGERS_PER_FLIGHT": 100.0,  # Same aircraft capacity
    
    # Resource allocation - SEVERELY REDUCED CAPACITY
    "NUM_CASHIER_SERVERS_LAND": 16,  # Same as High (-1 vs baseline)
    "NUM_CASHIER_SERVERS_AIR": 7,     # Same as High (-1 vs baseline)
    "NUM_SECURITY_SERVERS": 7,         # Same as High - critical bottleneck (-1 vs baseline)
    "NUM_BOARDING_SERVERS": 2,
    
    # Service rates (minutes) - SAME
    "CASHIER_SERVICE_RATE": 2.0,
    "SECURITY_SERVICE_RATE": 1.5,
    "BOARDING_SERVICE_RATE": 0.5,
    
    # Passenger behavior - SAME
    "BUYING_PROB": 0.8,
    "AVG_COMPANIONS": 1.5,
    "AVG_LANDSIDE_DWELL_TIME": 30.0,
    "AVG_AIRSIDE_DWELL_TIME": 15.0,
    
    # Arrival patterns - HIGHLY SYNCHRONIZED (MAJOR STRESS FACTOR)
    "AVG_ARRIVAL_TIME_BEFORE_FLIGHT": 120.0,
    "STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT": 26.5,  # High synchronization = severe arrival peaks
    
    # Time constraints
    "SECURITY_DEADLINE": 40,
    "BOARDING_WINDOW_START": 45,
    "BOARDING_DEADLINE": 20,
    
    # Priority distribution
    "PRIORITY_PROBS": [0.7, 0.2, 0.1],
}


# ==============================================================================
# HELPER FUNCTION TO RUN SCENARIOS
# ==============================================================================
def get_all_scenarios():
    """Returns a list of all scenario configurations."""
    return [
        SCENARIO_1_LOW_STRESS,
        SCENARIO_2_MEDIUM_STRESS,
        SCENARIO_3_HIGH_STRESS,
        SCENARIO_4_EXTREME_STRESS,
    ]


def print_scenario_summary():
    """Prints a summary comparison of all scenarios."""
    scenarios = get_all_scenarios()
    
    print("\n" + "="*100)
    print("AIRPORT SIMULATION SCENARIOS COMPARISON")
    print("="*100)
    
    print(f"\n{'Scenario':<25} {'Flights/Hr':<12} {'Pax/Flight':<12} {'Security':<10} {'Drop Risk':<12}")
    print("-"*100)
    
    for scenario in scenarios:
        flights_per_hour = 60 / scenario['FLIGHT_FREQUENCY']
        pax_per_flight = scenario['AVG_PASSENGERS_PER_FLIGHT']
        security_servers = scenario['NUM_SECURITY_SERVERS']
        
        # Estimate risk level based on load vs capacity
        hourly_pax = flights_per_hour * pax_per_flight
        capacity_ratio = hourly_pax / (security_servers * 40)  # Rough capacity estimate
        
        if capacity_ratio < 0.7:
            risk = "Low"
        elif capacity_ratio < 1.0:
            risk = "Medium"
        elif capacity_ratio < 1.3:
            risk = "High"
        else:
            risk = "Critical"
        
        print(f"{scenario['name']:<25} {flights_per_hour:>6.1f}/hr    "
              f"{pax_per_flight:>6.0f}       {security_servers:>4}       {risk:<12}")
    
    print("\n" + "="*100)
    print("\nKey Differences by Scenario:")
    print("-"*100)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   • Flight Frequency: {scenario['FLIGHT_FREQUENCY']} min")
        print(f"   • Security Servers: {scenario['NUM_SECURITY_SERVERS']}")
        print(f"   • Arrival Std Dev: {scenario['STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT']:.0f} min "
              f"({'Synchronized' if scenario['STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT'] < 25 else 'Distributed'})")
        print(f"   • Shopping Probability: {scenario['BUYING_PROB']*100:.0f}%")
    
    print("\n" + "="*100)


def print_hyperparameters_table():
    """Prints complete hyperparameter configuration table for all scenarios."""
    scenarios = get_all_scenarios()
    
    print("\n" + "="*135)
    print("COMPLETE HYPERPARAMETER CONFIGURATION TABLE")
    print("="*135)
    
    # Section 1: Temporal & Flight Configuration
    print("\n--- TEMPORAL & FLIGHT SCHEDULING ---")
    print(f"{'Parameter':<40} {'Low Stress':<20} {'Medium Stress':<20} {'High Stress':<20} {'Extreme Stress':<20}")
    print("-"*135)
    print(f"{'Simulation Time (hours)':<40} {scenarios[0]['SIMULATION_TIME']/60:<20.0f} {scenarios[1]['SIMULATION_TIME']/60:<20.0f} {scenarios[2]['SIMULATION_TIME']/60:<20.0f} {scenarios[3]['SIMULATION_TIME']/60:<20.0f}")
    print(f"{'Flight Frequency (min)':<40} {scenarios[0]['FLIGHT_FREQUENCY']:<20.0f} {scenarios[1]['FLIGHT_FREQUENCY']:<20.0f} {scenarios[2]['FLIGHT_FREQUENCY']:<20.0f} {scenarios[3]['FLIGHT_FREQUENCY']:<20.0f}")
    print(f"{'Avg Passengers/Flight':<40} {scenarios[0]['AVG_PASSENGERS_PER_FLIGHT']:<20.0f} {scenarios[1]['AVG_PASSENGERS_PER_FLIGHT']:<20.0f} {scenarios[2]['AVG_PASSENGERS_PER_FLIGHT']:<20.0f} {scenarios[3]['AVG_PASSENGERS_PER_FLIGHT']:<20.0f}")
    
    # Section 2: Resource Allocation
    print("\n--- RESOURCE ALLOCATION (Server Counts) ---")
    print(f"{'Parameter':<40} {'Low Stress':<20} {'Medium Stress':<20} {'High Stress':<20} {'Extreme Stress':<20}")
    print("-"*135)
    print(f"{'Cashier Servers (Landside)':<40} {scenarios[0]['NUM_CASHIER_SERVERS_LAND']:<20} {scenarios[1]['NUM_CASHIER_SERVERS_LAND']:<20} {scenarios[2]['NUM_CASHIER_SERVERS_LAND']:<20} {scenarios[3]['NUM_CASHIER_SERVERS_LAND']:<20}")
    print(f"{'Cashier Servers (Airside)':<40} {scenarios[0]['NUM_CASHIER_SERVERS_AIR']:<20} {scenarios[1]['NUM_CASHIER_SERVERS_AIR']:<20} {scenarios[2]['NUM_CASHIER_SERVERS_AIR']:<20} {scenarios[3]['NUM_CASHIER_SERVERS_AIR']:<20}")
    print(f"{'Security Servers':<40} {scenarios[0]['NUM_SECURITY_SERVERS']:<20} {scenarios[1]['NUM_SECURITY_SERVERS']:<20} {scenarios[2]['NUM_SECURITY_SERVERS']:<20} {scenarios[3]['NUM_SECURITY_SERVERS']:<20}")
    print(f"{'Boarding Servers':<40} {scenarios[0]['NUM_BOARDING_SERVERS']:<20} {scenarios[1]['NUM_BOARDING_SERVERS']:<20} {scenarios[2]['NUM_BOARDING_SERVERS']:<20} {scenarios[3]['NUM_BOARDING_SERVERS']:<20}")
    
    # Section 3: Service Rates
    print("\n--- SERVICE RATES (minutes/customer) ---")
    print(f"{'Parameter':<40} {'Low Stress':<20} {'Medium Stress':<20} {'High Stress':<20} {'Extreme Stress':<20}")
    print("-"*135)
    print(f"{'Cashier Service Rate':<40} {scenarios[0]['CASHIER_SERVICE_RATE']:<20.1f} {scenarios[1]['CASHIER_SERVICE_RATE']:<20.1f} {scenarios[2]['CASHIER_SERVICE_RATE']:<20.1f} {scenarios[3]['CASHIER_SERVICE_RATE']:<20.1f}")
    print(f"{'Security Service Rate':<40} {scenarios[0]['SECURITY_SERVICE_RATE']:<20.1f} {scenarios[1]['SECURITY_SERVICE_RATE']:<20.1f} {scenarios[2]['SECURITY_SERVICE_RATE']:<20.1f} {scenarios[3]['SECURITY_SERVICE_RATE']:<20.1f}")
    print(f"{'Boarding Service Rate':<40} {scenarios[0]['BOARDING_SERVICE_RATE']:<20.2f} {scenarios[1]['BOARDING_SERVICE_RATE']:<20.2f} {scenarios[2]['BOARDING_SERVICE_RATE']:<20.2f} {scenarios[3]['BOARDING_SERVICE_RATE']:<20.2f}")
    
    # Section 4: Passenger Behavior
    print("\n--- PASSENGER BEHAVIOR ---")
    print(f"{'Parameter':<40} {'Low Stress':<20} {'Medium Stress':<20} {'High Stress':<20} {'Extreme Stress':<20}")
    print("-"*135)
    print(f"{'Shopping Probability':<40} {scenarios[0]['BUYING_PROB']:<20.2f} {scenarios[1]['BUYING_PROB']:<20.2f} {scenarios[2]['BUYING_PROB']:<20.2f} {scenarios[3]['BUYING_PROB']:<20.2f}")
    print(f"{'Avg Companions':<40} {scenarios[0]['AVG_COMPANIONS']:<20.1f} {scenarios[1]['AVG_COMPANIONS']:<20.1f} {scenarios[2]['AVG_COMPANIONS']:<20.1f} {scenarios[3]['AVG_COMPANIONS']:<20.1f}")
    print(f"{'Avg Landside Dwell (min)':<40} {scenarios[0]['AVG_LANDSIDE_DWELL_TIME']:<20.0f} {scenarios[1]['AVG_LANDSIDE_DWELL_TIME']:<20.0f} {scenarios[2]['AVG_LANDSIDE_DWELL_TIME']:<20.0f} {scenarios[3]['AVG_LANDSIDE_DWELL_TIME']:<20.0f}")
    print(f"{'Avg Airside Dwell (min)':<40} {scenarios[0]['AVG_AIRSIDE_DWELL_TIME']:<20.0f} {scenarios[1]['AVG_AIRSIDE_DWELL_TIME']:<20.0f} {scenarios[2]['AVG_AIRSIDE_DWELL_TIME']:<20.0f} {scenarios[3]['AVG_AIRSIDE_DWELL_TIME']:<20.0f}")
    
    # Section 5: Arrival Patterns (KEY DIFFERENTIATOR)
    print("\n--- ARRIVAL PATTERNS (KEY STRESS FACTOR) ---")
    print(f"{'Parameter':<40} {'Low Stress':<20} {'Medium Stress':<20} {'High Stress':<20} {'Extreme Stress':<20}")
    print("-"*135)
    print(f"{'Avg Arrival Before Flight (min)':<40} {scenarios[0]['AVG_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f} {scenarios[1]['AVG_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f} {scenarios[2]['AVG_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f} {scenarios[3]['AVG_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f}")
    print(f"{'StdDev Arrival Time (min) ⭐':<40} {scenarios[0]['STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f} {scenarios[1]['STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f} {scenarios[2]['STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f} {scenarios[3]['STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT']:<20.0f}")
    
    # Section 6: Time Constraints
    print("\n--- TIME CONSTRAINTS (minutes before departure) ---")
    print(f"{'Parameter':<40} {'Low Stress':<20} {'Medium Stress':<20} {'High Stress':<20} {'Extreme Stress':<20}")
    print("-"*135)
    print(f"{'Security Deadline':<40} {scenarios[0]['SECURITY_DEADLINE']:<20} {scenarios[1]['SECURITY_DEADLINE']:<20} {scenarios[2]['SECURITY_DEADLINE']:<20} {scenarios[3]['SECURITY_DEADLINE']:<20}")
    print(f"{'Boarding Window Start':<40} {scenarios[0]['BOARDING_WINDOW_START']:<20} {scenarios[1]['BOARDING_WINDOW_START']:<20} {scenarios[2]['BOARDING_WINDOW_START']:<20} {scenarios[3]['BOARDING_WINDOW_START']:<20}")
    print(f"{'Boarding Deadline':<40} {scenarios[0]['BOARDING_DEADLINE']:<20} {scenarios[1]['BOARDING_DEADLINE']:<20} {scenarios[2]['BOARDING_DEADLINE']:<20} {scenarios[3]['BOARDING_DEADLINE']:<20}")
    
    # Section 7: Priority Distribution
    print("\n--- PRIORITY CLASS DISTRIBUTION [Economy, Business, First] ---")
    print(f"{'Parameter':<40} {'Low Stress':<20} {'Medium Stress':<20} {'High Stress':<20} {'Extreme Stress':<20}")
    print("-"*135)
    print(f"{'Priority Probabilities':<40} {str(scenarios[0]['PRIORITY_PROBS']):<20} {str(scenarios[1]['PRIORITY_PROBS']):<20} {str(scenarios[2]['PRIORITY_PROBS']):<20} {str(scenarios[3]['PRIORITY_PROBS']):<20}")
    
    print("\n" + "="*135)
    print("\n⭐ KEY PARAMETERS VARIED:")
    print("  • Simulation Time: 24 hours for all scenarios")
    print("  • Flight Frequency: 24 → 20 → 19 → 17.5 min (37% load increase)")
    print("  • Arrival StdDev: 35 → 30 → 27 → 25.5 min (27% more synchronization)")
    print("  • Security Servers: 10 → 8 → 7 → 7 (30% capacity reduction)")
    print("  • Cashier Servers (Land): 19 → 17 → 16 → 16 (16% capacity reduction)")
    print("  • Cashier Servers (Air): 10 → 8 → 7 → 7 (30% capacity reduction)")
    print("  • Boarding Servers: 3 → 2 → 2 → 2 (33% decrease for Low Stress)")
    print("  • Medium Stress = airport_departure.py baseline configuration")
    print("="*135 + "\n")


# ==============================================================================
# SIMULATION RUNNER
# ==============================================================================
def run_scenario(scenario_config, seed=42, verbose=True, show_plots=False):
    """
    Run a single scenario simulation.
    
    Args:
        scenario_config: Dictionary with scenario parameters
        seed: Random seed for reproducibility
        verbose: Whether to print detailed statistics
        show_plots: Whether to show time series plots
        
    Returns:
        Dictionary with key performance indicators
    """
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"RUNNING: {scenario_config['name']}")
    print(f"{'='*80}")
    print(f"Description: {scenario_config['description']}")
    print(f"{'='*80}\n")
    
    # Extract parameters with proper naming for SimulationEngine
    engine = SimulationEngine(
        simulation_time=scenario_config['SIMULATION_TIME'],
        buying_prob=scenario_config['BUYING_PROB'],
        num_cashier_servers_land=scenario_config['NUM_CASHIER_SERVERS_LAND'],
        num_cashier_servers_air=scenario_config['NUM_CASHIER_SERVERS_AIR'],
        num_security_servers=scenario_config['NUM_SECURITY_SERVERS'],
        num_boarding_servers=scenario_config['NUM_BOARDING_SERVERS'],
        avg_landside_dwell_time=scenario_config['AVG_LANDSIDE_DWELL_TIME'],
        avg_airside_dwell_time=scenario_config['AVG_AIRSIDE_DWELL_TIME'],
        avg_companions=scenario_config['AVG_COMPANIONS'],
        avg_passengers_per_flight=scenario_config['AVG_PASSENGERS_PER_FLIGHT'],
        priority_probs=scenario_config['PRIORITY_PROBS'],
        avg_arrival_time_before_flight=scenario_config['AVG_ARRIVAL_TIME_BEFORE_FLIGHT'],
        stddev_arrival_time_before_flight=scenario_config['STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT'],
        cashier_service_rate=scenario_config['CASHIER_SERVICE_RATE'],
        security_service_rate=scenario_config['SECURITY_SERVICE_RATE'],
        boarding_service_rate=scenario_config['BOARDING_SERVICE_RATE'],
        security_deadline=scenario_config['SECURITY_DEADLINE'],
        boarding_window_start=scenario_config['BOARDING_WINDOW_START'],
        boarding_deadline=scenario_config['BOARDING_DEADLINE'],
        flight_freq=scenario_config['FLIGHT_FREQUENCY']
    )
    
    # Run simulation
    engine.event_loop()
    
    # Get statistics - always get KPIs first
    if verbose:
        engine.print_statistics(verbose=True)
        # Get KPIs separately when verbose
        kpis = {
            'avg_security_wait': engine.metrics.get_average_security_wait_time(),
            'avg_boarding_wait': engine.metrics.get_average_boarding_wait_time(),
            'avg_total_time': sum(sum(times) for times in engine.metrics.total_time_per_priority.values()) / 
                             sum(len(times) for times in engine.metrics.total_time_per_priority.values()) if sum(len(times) for times in engine.metrics.total_time_per_priority.values()) > 0 else 0.0,
            'drop_percentage': engine.metrics.get_drop_percentage(),
            'security_utilization': engine.metrics.get_security_utilization(engine.simulation_time),
            'boarding_utilization': engine.metrics.get_boarding_utilization(engine.boarding_window_start)
        }
    else:
        kpis = engine.print_statistics(verbose=False)
    
    # Show plots if requested
    if show_plots:
        engine.plot_time_series()
    
    return kpis


def run_all_scenarios(num_runs_per_scenario=1, base_seed=42, verbose=True, show_plots=False):
    """
    Run all scenarios and compare results.
    
    Args:
        num_runs_per_scenario: Number of replications per scenario
        base_seed: Base seed for random number generation
        verbose: Whether to print detailed statistics for each run
        show_plots: Whether to show plots (only for last run of each scenario)
        
    Returns:
        Dictionary with results for all scenarios
    """
    scenarios = get_all_scenarios()
    all_results = {}
    
    print("\n" + "="*100)
    print("RUNNING ALL SCENARIOS")
    print("="*100)
    print(f"Number of runs per scenario: {num_runs_per_scenario}")
    print(f"Base seed: {base_seed}")
    print("="*100)
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        scenario_kpis = []
        
        for run in range(num_runs_per_scenario):
            seed = base_seed + run
            
            if num_runs_per_scenario > 1:
                print(f"\n--- Run {run+1}/{num_runs_per_scenario} for {scenario_name} ---")
                kpis = run_scenario(scenario, seed=seed, verbose=False, show_plots=False)
            else:
                show_plots_this_run = show_plots
                kpis = run_scenario(scenario, seed=seed, verbose=verbose, show_plots=show_plots_this_run)
            
            scenario_kpis.append(kpis)
        
        # Compute averages across runs
        if num_runs_per_scenario > 1:
            avg_kpis = {
                'avg_security_wait': np.mean([k['avg_security_wait'] for k in scenario_kpis]),
                'avg_boarding_wait': np.mean([k['avg_boarding_wait'] for k in scenario_kpis]),
                'avg_total_time': np.mean([k['avg_total_time'] for k in scenario_kpis]),
                'drop_percentage': np.mean([k['drop_percentage'] for k in scenario_kpis]),
                'security_utilization': np.mean([k['security_utilization'] for k in scenario_kpis]),
                'boarding_utilization': np.mean([k['boarding_utilization'] for k in scenario_kpis])
            }
            all_results[scenario_name] = avg_kpis
            
            print(f"\n--- AVERAGE RESULTS FOR {scenario_name} ({num_runs_per_scenario} runs) ---")
            print(f"Drop Rate: {avg_kpis['drop_percentage']:.2f}%")
            print(f"Security Wait: {avg_kpis['avg_security_wait']:.2f} min")
            print(f"Security Utilization: {avg_kpis['security_utilization']*100:.2f}%")
            print(f"Boarding Wait: {avg_kpis['avg_boarding_wait']:.2f} min")
            print(f"Total Time: {avg_kpis['avg_total_time']:.2f} min")
        else:
            all_results[scenario_name] = scenario_kpis[0]
    
    # Print comparison table
    print_comparison_table(all_results)
    
    return all_results


def print_comparison_table(results):
    """Print a comparison table of all scenario results."""
    print("\n" + "="*120)
    print("SCENARIOS COMPARISON - KEY PERFORMANCE INDICATORS")
    print("="*120)
    
    print(f"\n{'Scenario':<27} {'Drop %':<10} {'Sec Wait':<12} {'Sec Util%':<12} {'Board Wait':<12} {'Total Time':<12}")
    print("-"*120)
    
    for scenario_name, kpis in results.items():
        print(f"{scenario_name:<27} "
              f"{kpis['drop_percentage']:>7.2f}%   "
              f"{kpis['avg_security_wait']:>8.2f} min   "
              f"{kpis['security_utilization']*100:>8.2f}%    "
              f"{kpis['avg_boarding_wait']:>8.2f} min   "
              f"{kpis['avg_total_time']:>8.2f} min")
    
    print("="*120)
    
    # Find best and worst
    best_drop = min(results.items(), key=lambda x: x[1]['drop_percentage'])
    worst_drop = max(results.items(), key=lambda x: x[1]['drop_percentage'])
    
    print("\n📊 Key Insights:")
    print(f"  ✓ Best Performance: {best_drop[0]} (Drop Rate: {best_drop[1]['drop_percentage']:.2f}%)")
    print(f"  ✗ Worst Performance: {worst_drop[0]} (Drop Rate: {worst_drop[1]['drop_percentage']:.2f}%)")
    print("\n" + "="*120 + "\n")


def plot_kpi_comparison(results):
    """
    Create comprehensive visualization comparing KPIs across scenarios.
    
    Args:
        results: Dictionary with scenario names as keys and KPI dictionaries as values
    """
    # Extract data for plotting
    scenarios = list(results.keys())
    scenario_labels = [s.replace(' Configuration', '') for s in scenarios]
    
    # Extract KPIs
    drop_rates = [results[s]['drop_percentage'] for s in scenarios]
    sec_wait = [results[s]['avg_security_wait'] for s in scenarios]
    board_wait = [results[s]['avg_boarding_wait'] for s in scenarios]
    total_time = [results[s]['avg_total_time'] for s in scenarios]
    sec_util = [results[s]['security_utilization'] * 100 for s in scenarios]
    board_util = [results[s]['boarding_utilization'] * 100 for s in scenarios]
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Airport Simulation: KPI Comparison Across Scenarios', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color scheme
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Green, Blue, Orange, Red
    
    # Plot 1: Drop Rate
    ax1 = axes[0, 0]
    bars1 = ax1.bar(scenario_labels, drop_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Drop Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Passenger Drop Rate', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(drop_rates) * 1.15)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 2: Security Wait Time
    ax2 = axes[0, 1]
    bars2 = ax2.bar(scenario_labels, sec_wait, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Wait Time (minutes)', fontsize=11, fontweight='bold')
    ax2.set_title('Security Queue Wait Time', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(sec_wait) * 1.15)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 3: Boarding Wait Time
    ax3 = axes[0, 2]
    bars3 = ax3.bar(scenario_labels, board_wait, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Wait Time (minutes)', fontsize=11, fontweight='bold')
    ax3.set_title('Boarding Queue Wait Time', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, max(board_wait) * 1.15)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 4: Total Time
    ax4 = axes[1, 0]
    bars4 = ax4.bar(scenario_labels, total_time, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax4.set_title('Total Time in System', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim(0, max(total_time) * 1.15)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 5: Security Utilization
    ax5 = axes[1, 1]
    bars5 = ax5.bar(scenario_labels, sec_util, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
    ax5.set_title('Security Server Utilization', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.set_ylim(0, 100)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 6: Boarding Utilization
    ax6 = axes[1, 2]
    bars6 = ax6.bar(scenario_labels, board_util, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Boarding Server Utilization', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    ax6.set_ylim(0, 100)
    for bar in bars6:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = 'scenario_kpi_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 KPI comparison plot saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    return fig


# ==============================================================================
# EXPECTED OUTCOMES
# ==============================================================================
"""
Expected Performance Characteristics:

SCENARIO 1 (Low Stress):
- Drop Rate: ~0% (excellent)
- Security Wait: <1 minute
- Security Utilization: 40-50%
- Overall: Over-capacity with dispersed arrivals = minimal queuing

SCENARIO 2 (Medium Stress):
- Drop Rate: ~0-2% (acceptable)
- Security Wait: 3-6 minutes
- Security Utilization: 65-75%
- Overall: Balanced baseline configuration with manageable queues

SCENARIO 3 (High Stress):
- Drop Rate: ~8-10% (concerning - near limit)
- Security Wait: 6-10 minutes
- Security Utilization: 72-78%
- Overall: Under-capacity + synchronized arrivals = sustained pressure

SCENARIO 4 (Extreme Stress):
- Drop Rate: ~10-12% (critical - at system limits)
- Security Wait: 8-12 minutes
- Security Utilization: 74-80%
- Overall: Maximum frequency + high synchronization = crisis conditions

Key Stress Factors Tested:
1. Flight frequency (24 → 20 → 19 → 17.5 min): Progressive 37% increase in arrival rate
2. Arrival synchronization (σ = 35 → 30 → 27 → 25.5 min): Progressive 27% increase in synchronization
3. System capacity - Security servers (10 → 8 → 7 → 7): Progressive 30% capacity reduction
4. System capacity - Cashier servers: (19+10 → 17+8 → 16+7 → 16+7): Progressive 16-30% capacity reduction
5. Combined multi-factor stress escalation maintaining ≤12% drop rate constraint
5. All simulations run for 24 hours
6. Medium Stress matches airport_departure.py hardcoded baseline configuration
"""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run airport simulation scenarios')
    parser.add_argument('--mode', type=str, default='summary', 
                       choices=['summary', 'single', 'all', 'compare'],
                       help='Run mode: summary (just show table), single (one scenario), all (run all), compare (multiple runs)')
    parser.add_argument('--scenario', type=int, default=1, 
                       help='Scenario number (1-4) for single mode')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per scenario for compare mode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--plots', action='store_true',
                       help='Show plots')
    parser.add_argument('--kpi-plot', action='store_true',
                       help='Generate KPI comparison plot (for all/compare modes)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed statistics')
    
    args = parser.parse_args()
    
    if args.mode == 'summary':
        # Just show the scenario comparison table
        print_scenario_summary()
        print_hyperparameters_table()
        
    elif args.mode == 'single':
        # Run a single scenario
        scenarios = get_all_scenarios()
        if 1 <= args.scenario <= len(scenarios):
            scenario = scenarios[args.scenario - 1]
            run_scenario(scenario, seed=args.seed, verbose=args.verbose, show_plots=args.plots)
        else:
            print(f"Error: Scenario number must be between 1 and {len(scenarios)}")
            
    elif args.mode == 'all':
        # Run all scenarios once
        results = run_all_scenarios(num_runs_per_scenario=1, base_seed=args.seed, 
                                    verbose=args.verbose, show_plots=args.plots)
        if args.kpi_plot:
            plot_kpi_comparison(results)
        
    elif args.mode == 'compare':
        # Run all scenarios multiple times and compare
        results = run_all_scenarios(num_runs_per_scenario=args.runs, base_seed=args.seed,
                                    verbose=False, show_plots=False)
        if args.kpi_plot:
            plot_kpi_comparison(results)
