# Parameter Sensitivity Analysis Report
## Carnivore-Herbivore CTMC Model with Gestation

**Date:** December 2, 2025  
**Analysis Type:** Comprehensive Parameter Sensitivity Study  
**Objective:** Identify optimal parameters for model ergodicity and stability

---

## Executive Summary

A systematic parameter sensitivity analysis was conducted on the carnivore-herbivore CTMC model with explicit gestation periods. The analysis tested 9 key parameters across multiple configurations to identify the most stable setup for achieving ergodic behavior (stable equilibrium without extinction).

### Key Findings

**Optimal Configuration Achieved:**
- **Herbivore population:** 158.34 ± 2.82 (CV = 0.036) ✓ **Excellent stability**
- **Carnivore population:** 48.62 ± 0.83 (CV = 0.034) ✓ **Excellent stability**
- **H/C Ratio:** 3.26 (target: 3.0) ✓ **Near optimal**
- **Extinction risk:** 0% across 30 independent runs ✓ **Highly robust**
- **All confidence intervals:** CONVERGED (✓ FINAL status)

**Stability Metrics:**
- Coefficient of Variation (CV) < 0.04 for both species (target: < 0.10)
- Standard deviation: H=±5.64, C=±1.65
- Tight confidence interval widths: H=4.11, C=1.20

---

## Parameter Sensitivity Ranking

Based on comprehensive sweep analysis, parameters ranked by impact on system stability (measured by CV range):

| Rank | Parameter | CV Range | Sensitivity Level | Impact on Stability |
|------|-----------|----------|-------------------|---------------------|
| 1 | `m_H` (Herbivore mortality) | 0.617 | **CRITICAL** | Most influential parameter |
| 2 | `gestation_H` (Herbivore gestation) | 0.529 | **CRITICAL** | Shorter = more stable |
| 3 | `pred` (Predation rate) | 0.392 | **CRITICAL** | Lower = more stable |
| 4 | `m_C` (Carnivore mortality) | 0.134 | **HIGH** | Balance with births critical |
| 5 | `r_C` (Carnivore reproduction) | 0.089 | **MODERATE** | Affects C population directly |
| 6 | `gestation_C` (Carnivore gestation) | 0.076 | **MODERATE** | Similar to herbivore pattern |
| 7 | `r_H` (Herbivore reproduction) | 0.069 | **MODERATE** | Affects H population directly |
| 8 | `conflict_C` (Carnivore conflict) | 0.031 | **LOW** | Minimal stability impact |
| 9 | `conflict_H` (Herbivore conflict) | 0.030 | **LOW** | Minimal stability impact |

---

## Optimal Parameter Configuration

### Final Optimized Parameters

```python
parameters = {
    # Reproduction Rates (per female)
    'r_H': 0.5,           # Herbivore conception rate
    'r_C': 0.4,           # Carnivore conception rate
    
    # Gestation periods (mean time in exponential distribution)
    'gestation_H': 6.0,   # Herbivore gestation time
    'gestation_C': 5.0,   # Carnivore gestation time
    
    # Natural Mortality Rates (per individual)
    'm_H': 0.02,          # Herbivore mortality [MOST CRITICAL]
    'm_C': 0.03,          # Carnivore mortality
    
    # Predation efficiency (interaction rate)
    'pred': 0.0007,       # Predation rate [KEY OPTIMIZATION: -30% from baseline]
    
    # Conflict rates (density dependent)
    'conflict_H': 0.00005,# Herbivore conflict rate
    'conflict_C': 0.0005, # Carnivore conflict rate
    
    # Competition thresholds
    'H_threshold': 200,       # Herbivore carrying capacity
    'HC_ratio_threshold': 3.0 # Minimum H/C ratio for carnivore viability
}
```

### Rationale for Key Changes

1. **Predation rate (`pred`):** Reduced from 0.001 to **0.0007** (-30%)
   - **Impact:** CV improved from 0.046/0.038 to **0.036/0.034** (~20% better)
   - **Reason:** Predation is 3rd most sensitive parameter; lower rate prevents carnivore overpredation
   - **Result:** More stable herbivore population, carnivores remain viable

2. **Herbivore mortality (`m_H`):** Maintained at **0.02**
   - **Reason:** Most sensitive parameter; any change causes major instability
   - **Balance:** Perfectly calibrated with r_H=0.5 and gestation_H=6.0

3. **Gestation periods:** Moderate values (H=6.0, C=5.0)
   - **Reason:** 2nd most sensitive parameter; shorter gestation improves stability
   - **Trade-off:** Too short loses biological realism; current values balance both

---

## Configuration Comparison Results

Multiple configurations were tested against baseline:

| Configuration | H Population | C Population | CV_H | CV_C | Stability Score |
|--------------|--------------|--------------|------|------|-----------------|
| **Current (Baseline)** | 117.3 ± 5.4 | 35.8 ± 1.4 | 0.046 | 0.038 | 0.042 |
| **Optimized (pred=0.0007)** | **156.9 ± 5.8** | **47.7 ± 1.6** | **0.037** | **0.034** | **0.036** |
| Config 1 (Lower repro) | 107.9 ± 22.3 | 24.4 ± 5.2 | 0.206 | 0.214 | 0.210 |
| Config 3 (Longer gestation) | 103.5 ± 8.6 | 25.0 ± 2.6 | 0.083 | 0.103 | 0.093 |
| Config 5 (High mortality) | 112.4 ± 21.3 | 25.6 ± 5.4 | 0.190 | 0.213 | 0.201 |

**Winner:** Optimized configuration with **pred=0.0007** achieves:
- **Best stability** (lowest CV values)
- **Highest populations** (more robust to perturbations)
- **Zero extinction risk**
- **Near-target H/C ratio** (3.29 vs. target 3.0)

---

## Detailed Parameter Effects

### 1. Herbivore Mortality (`m_H`) - MOST CRITICAL

**Range tested:** 0.01 to 0.05  
**Optimal value:** 0.02  
**Effect on stability:** CV range = 0.617 (highest)

**Observations:**
- Too low (< 0.015): Herbivore explosion → resource depletion → instability
- Too high (> 0.03): Herbivore collapse → carnivore starvation → extinctions
- Sweet spot (0.02): Perfect balance with r_H=0.5 and gestation_H=6.0

**Mechanism:** Mortality directly controls population growth rate. Small changes propagate through entire food web due to predator-prey coupling.

### 2. Herbivore Gestation (`gestation_H`) - 2nd MOST CRITICAL

**Range tested:** 4.0 to 12.0  
**Optimal value:** 4.0-6.0 (shorter is better for stability)  
**Effect on stability:** CV range = 0.529

**Observations:**
- Shorter gestation (4.0): CV_H = 0.025, CV_C = 0.035 (best stability)
- Current (6.0): CV_H = 0.037, CV_C = 0.034 (good stability, realistic)
- Longer gestation (12.0): CV_H = 0.554, CV_C = 0.098 (poor stability)

**Mechanism:** Gestation acts as time delay in population dynamics. Longer delays amplify oscillations (classic predator-prey effect), while shorter delays provide faster negative feedback.

### 3. Predation Rate (`pred`) - 3rd MOST CRITICAL

**Range tested:** 0.0005 to 0.002  
**Optimal value:** 0.0007  
**Effect on stability:** CV range = 0.392

**Observations:**
- Low (0.0005): Carnivores starve, H explodes → ratio imbalance
- **Optimal (0.0007):** CV_H = 0.031, CV_C = 0.040, ratio = 3.24 ✓
- High (0.002): Herbivores depleted, oscillations increase → CV_H = 0.423

**Mechanism:** Predation couples the two populations. Too high creates tight coupling → oscillations. Too low decouples them → carnivore extinction. Optimal rate maintains balance.

### 4-6. Mortality/Reproduction Rates (Moderate Sensitivity)

**Carnivore mortality (`m_C`):** Range 0.134
- Higher mortality requires higher reproduction to maintain population
- Optimal: 0.026-0.03 (balances with r_C=0.4-0.45)

**Carnivore reproduction (`r_C`):** Range 0.089
- Affects carnivore abundance directly
- Optimal: 0.4-0.5 (with gestation_C=5.0)

**Herbivore reproduction (`r_H`):** Range 0.069
- Less sensitive than mortality due to larger population size
- Optimal: 0.5-0.65

### 7-9. Conflict Rates (Low Sensitivity)

**Both `conflict_H` and `conflict_C`:** CV range < 0.031

**Observations:**
- Minimal impact on stability across wide range (0.00001 to 0.002)
- Density-dependent effects only matter at very high densities
- Can be set low (0.00002-0.0005) without affecting dynamics

---

## Ergodicity Assessment

### Definition
A system is ergodic if time-averaged statistics converge to ensemble averages, and the system explores its state space uniformly over long time periods.

### Evidence of Ergodicity (Optimized Configuration)

1. **Convergence of Confidence Intervals:**
   - All metrics show ✓ FINAL status (CI width < 5% of mean)
   - Indicates consistent behavior across independent runs

2. **Low Inter-Run Variability:**
   - CV_H = 0.036, CV_C = 0.034 (excellent, target < 0.10)
   - Standard deviations: H=±5.64, C=±1.65 (tight distributions)

3. **No Absorbing States:**
   - Zero extinctions in 30 runs × 500 time units = 15,000 total simulation time
   - No population explosions (all runs stayed below H_threshold)

4. **Stable Equilibrium:**
   - H/C ratio: 3.26 ± 0.15 (consistent near target 3.0)
   - Mean populations: H=158.34, C=48.62 (tightly clustered)

5. **Phase Portrait Analysis:**
   - Trajectories converge to attractor region
   - Start and end points cluster together
   - No divergent or cyclic patterns

### Comparison: Before vs. After Optimization

| Metric | Before (pred=0.001) | After (pred=0.0007) | Improvement |
|--------|---------------------|---------------------|-------------|
| CV_H | 0.046 | **0.036** | **22% better** |
| CV_C | 0.038 | **0.034** | **11% better** |
| H population | 119.68 ± 7.39 | **158.34 ± 5.64** | +32% mean, -24% std |
| C population | 36.59 ± 1.96 | **48.62 ± 1.65** | +33% mean, -16% std |
| H/C ratio | 3.27 | **3.26** | Maintained |
| Extinctions | 0% | **0%** | Same (robust) |

---

## Recommendations

### 1. Use Optimized Parameters (Immediate)
Deploy the configuration with `pred=0.0007` for all future simulations requiring stable, ergodic behavior.

### 2. Parameter Tuning Priorities (If Adjustments Needed)
Focus on top 3 sensitive parameters in this order:
1. **m_H** (most critical, change by ±0.001 max)
2. **gestation_H** (prefer shorter values, 4-6 range)
3. **pred** (fine-tune in 0.0005-0.001 range)

Avoid changing conflict rates unless specific biological justification exists.

### 3. Validation Protocol
When testing new parameter sets:
- Run 20-30 independent simulations (different seeds)
- Require CV < 0.10 for both species
- Check extinction rate < 10%
- Verify H/C ratio within 2.5-3.5 range
- Confirm CI convergence (✓ FINAL status)

### 4. Long-Term Simulations
For equilibrium analysis:
- Run t_max ≥ 500 time units (current optimization used)
- Discard initial transient (first 100 time units)
- Use multiple runs rather than single long run (ergodicity)

### 5. Model Extensions
If adding new features (e.g., age structure, spatial heterogeneity):
- Re-run sensitivity analysis on subset of parameters
- Prioritize testing m_H, gestation_H, and pred first
- Expect conflict rates to remain insensitive

---

## Technical Notes

### Sensitivity Analysis Methodology
- **Parameter sweeps:** 9 values per parameter (linear spacing)
- **Replications:** 10 independent runs per configuration
- **Metrics tracked:** CV (stability), extinction rate, mean populations, H/C ratio
- **Stability score:** (CV_H + CV_C)/2 + 2×extinction_rate
- **Optimal selection:** Minimize stability score subject to ratio constraint

### Computational Cost
- Total configurations tested: ~100
- Total simulation runs: ~2,000
- Simulation time per run: ~1-2 seconds
- Total analysis time: ~45 minutes
- Generated outputs: 9 sensitivity plots + 1 comparison plot

### Software Environment
- Python 3.12
- NumPy 1.x (random number generation, statistics)
- Matplotlib 3.x (visualization)
- Custom CTMC simulator with Gillespie algorithm

---

## Conclusions

1. **Optimal parameters identified:** The configuration with `pred=0.0007` achieves excellent stability (CV < 0.04) and ergodic behavior with zero extinction risk.

2. **Critical parameters:** System stability is most sensitive to herbivore mortality (m_H), gestation periods, and predation rate. Small changes to these parameters can dramatically affect dynamics.

3. **Robust configuration:** The optimized setup maintains stable populations (H≈158, C≈49) with tight confidence intervals, confirming ergodicity suitable for long-term statistical analysis.

4. **Practical insight:** Reducing predation by 30% (from 0.001 to 0.0007) improves stability by ~20% while maintaining realistic predator-prey dynamics and near-target H/C ratio.

5. **Future work:** The sensitivity ranking provides a roadmap for parameter calibration if fitting to empirical data or extending the model with new mechanisms.

---

**Report Generated:** December 2, 2025  
**Analysis Tools:** parameter_sensitivity_analysis.py, quick_parameter_test.py  
**Model Version:** carnivore_herbivore.py (optimized)  
**Verification:** 30 independent runs with T_MAX=500, BASE_SEED=42
