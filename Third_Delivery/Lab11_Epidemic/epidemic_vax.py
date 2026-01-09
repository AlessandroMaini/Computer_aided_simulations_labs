import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# FUNCTION DEFINITIONS
# ==========================================

def get_group_features(idx):
    """Returns (geo_idx, age_idx, work_idx) for a flat group index."""
    g = idx // 6
    rem = idx % 6
    a = rem // 3
    w = rem % 3
    return g, a, w

def build_parameters(num_groups):
    """Build heterogeneous transmission and fatality parameters."""
    beta = np.zeros((num_groups, num_groups))
    delta_I = np.zeros(num_groups)
    delta_Q = np.zeros(num_groups)
    
    # Different base transmission based on geography
    base_transmission = {0: 0.50,  # City
                         1: 0.05,  # Metro
                         2: 0.02}  # Piedmont

    for i in range(num_groups):
        g_i, a_i, w_i = get_group_features(i)
        
        # Fatality Rates
        if a_i == 0: # Young
            delta_I[i], delta_Q[i] = 0.0002, 0.00005
        else:        # Old
            delta_I[i], delta_Q[i] = 0.005, 0.001

        for j in range(num_groups):
            g_j, a_j, w_j = get_group_features(j)
            
            # 1. Geography factor (Mobility)
            if g_i == g_j:
                f_geo = 1.0
            elif (g_i == 0 and g_j == 1) or (g_i == 1 and g_j == 0):
                f_geo = 0.20
            else:
                f_geo = 0.05
            
            # 2. Age factor (Mixing)
            if a_i == 0 and a_j == 0: f_age = 1.2
            elif a_i == 1 and a_j == 1: f_age = 0.8
            else: f_age = 0.6
            
            # 3. Work factor (Exposure)
            risk_map = {0: 2.5, 1: 0.8, 2: 0.2}
            f_work = np.sqrt(risk_map[w_i] * risk_map[w_j])
            
            beta[i, j] = base_transmission[g_i] * f_geo * f_age * f_work

    return beta, delta_I, delta_Q

def get_vaccine_allocation(S_current, strategy_name, max_rate, num_groups):
    """Allocates vaccination capacity based on strategy."""
    v_rates = np.zeros(num_groups)
    remaining_capacity = max_rate
    
    # Define Priority Tiers based on strategy
    tiers = []
    
    if strategy_name == 'No_Vaccination':
        return v_rates
    
    elif strategy_name == 'Old_First':
        t1 = [k for k in range(num_groups) if get_group_features(k)[1] == 1]
        t2 = [k for k in range(num_groups) if get_group_features(k)[1] == 0]
        tiers = [t1, t2]
        
    elif strategy_name == 'HighExp_First':
        t1 = [k for k in range(num_groups) if get_group_features(k)[2] == 0]
        t2 = [k for k in range(num_groups) if get_group_features(k)[2] == 1]
        t3 = [k for k in range(num_groups) if get_group_features(k)[2] == 2]
        tiers = [t1, t2, t3]
        
    elif strategy_name == 'Geographical':
        t1 = [k for k in range(num_groups) if get_group_features(k)[0] == 0]
        t2 = [k for k in range(num_groups) if get_group_features(k)[0] == 1]
        t3 = [k for k in range(num_groups) if get_group_features(k)[0] == 2]
        tiers = [t1, t2, t3]
        
    else:  # Uniform
        tiers = [list(range(num_groups))]

    # Distribute Capacity
    for tier_groups in tiers:
        if remaining_capacity <= 0:
            break
            
        tier_S_total = np.sum(S_current[tier_groups])
        
        if tier_S_total > 0:
            capacity_for_tier = min(remaining_capacity, tier_S_total)
            
            for k in tier_groups:
                if S_current[k] > 0:
                    share = S_current[k] / tier_S_total
                    v_rates[k] = share * capacity_for_tier
            
            remaining_capacity -= capacity_for_tier
            
    return v_rates

def system_deriv(y, t, beta, delta_I, delta_Q, strategy, num_groups, params):
    """ODE system for SAIQRS-V model."""
    # Reshape: 7 cols -> S, A, I, Q, R, D, V
    state = y.reshape((num_groups, 7))
    S, A, I, Q, R, D, V = state[:,0], state[:,1], state[:,2], state[:,3], state[:,4], state[:,5], state[:,6]
    
    # A. Determine Vaccination Rates
    v_rates = np.zeros(num_groups)
    if t >= params['VAX_START_DAY']:
        v_rates = get_vaccine_allocation(S, strategy, params['VAX_RATE_TOTAL'], num_groups)

    # B. Force of Infection with Lockdown
    current_beta = beta.copy()
    in_lockdown = any(start <= t < end for start, end in params['LOCKDOWN_PERIODS'])
    if in_lockdown:
        current_beta = current_beta * params['LOCKDOWN_STRENGTH']
        
    # Need to normalize by population size for proper per-capita rates
    N = S + A + I + Q + R + V
    infectious_load = (I + params['EPSILON'] * A) / N
    Lambda = current_beta.dot(infectious_load)
    
    # C. Derivatives
    dS = -S * Lambda + params['XI'] * R + params['XI_V'] * V - v_rates
    dA = S * Lambda - (params['SIGMA'] + params['THETA_A']) * A
    dI = params['SIGMA'] * A - (params['THETA_I'] + params['GAMMA_I'] + delta_I) * I
    dQ = params['THETA_I'] * I + params['THETA_A'] * A - (params['GAMMA_Q'] + delta_Q) * Q
    dR = params['GAMMA_I'] * I + params['GAMMA_Q'] * Q - params['XI'] * R
    dD = delta_I * I + delta_Q * Q
    dV = v_rates - params['XI_V'] * V
    
    return np.stack([dS, dA, dI, dQ, dR, dD, dV], axis=1).flatten()

def init_populations(num_groups, geos, ages, works, populations, exposure_distribution):
    """Initialize population for each group."""
    y0 = np.zeros((num_groups, 7))

    for idx in range(num_groups):
        g_idx, a_idx, w_idx = get_group_features(idx)
        g = geos[g_idx]
        a = ages[a_idx]
        w = works[w_idx]
        pop_size = populations[g][a] * exposure_distribution[w]
        
        # Seed only in Turin City, Young, High Exposure
        if g_idx == 0 and a_idx == 0 and w_idx == 0:
            y0[idx, 0] = pop_size - 10.0
            y0[idx, 1] = 10.0
        else:
            y0[idx, 0] = pop_size
            y0[idx, 1] = 0.0
        
    return y0.flatten()

def run_strategy_comparison(t_max, dt, beta, delta_I, delta_Q, num_groups, geos, ages, works, 
                            populations, exposure_distribution, params):
    """Run simulation for all vaccination strategies."""
    y0_flat = init_populations(num_groups, geos, ages, works, populations, exposure_distribution)
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    
    strategies = ['No_Vaccination', 'Old_First', 'HighExp_First', 'Geographical', 'Uniform']
    results = {}
    
    print(f"Simulating {len(strategies)} strategies...")
    
    for strat in strategies:
        print(f"Running {strat}...")
        ret = odeint(system_deriv, y0_flat, t, args=(beta, delta_I, delta_Q, strat, num_groups, params))
        results[strat] = ret

    return t, results

def plot_strategy_details(t, data, strategy_name, num_groups, lockdown_periods, vax_start_day):
    """Create detailed plot for a vaccination strategy."""
    data_rs = data.reshape((len(t), num_groups, 7))
    
    # Aggregate across all groups
    total_S = np.sum(data_rs[:, :, 0], axis=1)
    total_A = np.sum(data_rs[:, :, 1], axis=1)
    total_I = np.sum(data_rs[:, :, 2], axis=1)
    total_Q = np.sum(data_rs[:, :, 3], axis=1)
    total_R = np.sum(data_rs[:, :, 4], axis=1)
    total_D = np.sum(data_rs[:, :, 5], axis=1)
    total_V = np.sum(data_rs[:, :, 6], axis=1)
    
    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(t, total_A, label='Total Asymptomatic (A)', color='yellow', linewidth=2)
    plt.plot(t, total_I, label='Total Symptomatic (I)', color='orange', linewidth=2)
    plt.plot(t, total_Q, label='Total Quarantined (Q)', color='blue', linewidth=2)
    plt.plot(t, total_R, label='Total Recovered (R)', color='green', linewidth=2)
    plt.plot(t, total_D, label='Total Deaths (D)', color='black', linewidth=2)
    plt.plot(t, total_V, label='Total Vaccinated (V)', color='purple', linewidth=2)
    
    # Mark lockdown periods
    for i, (start, end) in enumerate(lockdown_periods, 1):
        label_start = f'Lockdown Start' if i == 1 else None
        label_end = f'Lockdown End' if i == 1 else None
        plt.axvline(x=start, linestyle='--', color='red', linewidth=2, alpha=0.7, label=label_start)
        plt.axvline(x=end, linestyle='--', color='darkgreen', linewidth=2, alpha=0.7, label=label_end)
    
    # Mark vaccination start
    plt.axvline(x=vax_start_day, linestyle='--', color='magenta', linewidth=2, alpha=0.7, label='Vaccination Start')
    
    plt.title(f'SAIQRS-V Epidemic Model - Strategy: {strategy_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Epidemic Summary - Strategy: {strategy_name} ===")
    print(f"Total Population: {total_S[0] + total_A[0]:.0f}")
    print(f"Peak Symptomatic: {np.max(total_I):.0f} on day {np.argmax(total_I)}")
    print(f"Peak Asymptomatic: {np.max(total_A):.0f} on day {np.argmax(total_A)}")
    print(f"Total Deaths: {total_D[-1]:.0f}")
    print(f"Final Recovered: {total_R[-1]:.0f}")
    print(f"Final Vaccinated: {total_V[-1]:.0f}")
    print(f"Attack Rate: {(total_R[-1] + total_D[-1]) / (total_S[0] + total_A[0]) * 100:.2f}%")
    print(f"Deaths per 100k: {total_D[-1] / (total_S[0] + total_A[0]) * 100000:.1f}")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # ==========================================
    # 1. SETUP & INDICES
    # ==========================================
    GEOS = ['Turin_City', 'Turin_Metro', 'Piedmont']
    AGES = ['Young', 'Old']
    WORKS = ['High_Exp', 'Mid_Exp', 'Low_Exp']
    NUM_GROUPS = 18
    
    # ==========================================
    # 2. PARAMETERS
    # ==========================================
    T_MAX = 1095        # 3 Years
    DT = 1.0
    
    # Biology
    SIGMA = 1/5.0      # Latency
    EPSILON = 0.4      # Asymptomatic infectivity
    XI = 1/120.0       # Natural immunity waning
    XI_V = 1/365.0     # Vaccine immunity waning
    GAMMA_I = 1/10.0   
    GAMMA_Q = 1/8.0
    THETA_I = 0.15
    THETA_A = 0.005
    
    # Vaccination Parameters
    VAX_START_DAY = 365
    VAX_RATE_TOTAL = 4000.0
    
    # Lockdown parameters
    LOCKDOWN_PERIODS = [
        # (20, 120)
    ]
    LOCKDOWN_STRENGTH = 0.30
    
    # Population data
    POPULATIONS = {
        'Turin_City': { 'Young': 630000, 'Old': 270000 },
        'Turin_Metro': { 'Young': 1040000, 'Old': 360000 },
        'Piedmont': { 'Young': 1400000, 'Old': 500000 },
    }
    
    EXPOSURE_DISTRIBUTION = {
        'High_Exp': 0.15,
        'Mid_Exp': 0.5,
        'Low_Exp': 0.35,
    }
    
    # Package parameters for ODE system
    params = {
        'SIGMA': SIGMA,
        'EPSILON': EPSILON,
        'XI': XI,
        'XI_V': XI_V,
        'GAMMA_I': GAMMA_I,
        'GAMMA_Q': GAMMA_Q,
        'THETA_I': THETA_I,
        'THETA_A': THETA_A,
        'VAX_START_DAY': VAX_START_DAY,
        'VAX_RATE_TOTAL': VAX_RATE_TOTAL,
        'LOCKDOWN_PERIODS': LOCKDOWN_PERIODS,
        'LOCKDOWN_STRENGTH': LOCKDOWN_STRENGTH,
    }
    
    # ==========================================
    # 3. BUILD HETEROGENEOUS PARAMETERS
    # ==========================================
    BETA, DELTA_I, DELTA_Q = build_parameters(NUM_GROUPS)
    
    # ==========================================
    # 4. RUN SIMULATIONS
    # ==========================================
    t, res_dict = run_strategy_comparison(
        T_MAX, DT, BETA, DELTA_I, DELTA_Q, NUM_GROUPS, 
        GEOS, AGES, WORKS, POPULATIONS, EXPOSURE_DISTRIBUTION, params
    )
    
    # ==========================================
    # 5. PLOT RESULTS
    # ==========================================
    for strat, data in res_dict.items():
        plot_strategy_details(t, data, strat, NUM_GROUPS, LOCKDOWN_PERIODS, VAX_START_DAY)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
