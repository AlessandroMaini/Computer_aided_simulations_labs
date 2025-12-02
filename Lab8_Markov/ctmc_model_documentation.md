# Carnivore-Herbivore CTMC Model Documentation

## Overview

This document describes a **Continuous Time Markov Chain (CTMC)** simulation modeling predator-prey population dynamics between herbivores (H) and carnivores (C), with explicit tracking of male and female populations **including pregnant females**. The model implements stochastic population processes using the Gillespie algorithm with density-dependent competition effects and **explicit gestation periods** for realistic reproductive dynamics.

## Model Architecture

### Simulation Paradigm
- **Type**: Continuous Time Markov Chain (CTMC)
- **Algorithm**: Gillespie's Direct Method (stochastic simulation algorithm)
- **Time Handling**: Exponentially distributed inter-event times
- **State Space**: Discrete population counts (H_M, H_F, C_M, C_F)

### Key Features
- Sex-specific population tracking (males and females)
- Logistic competition thresholds for resource limitation
- Mass action kinetics for interactions
- Multiple independent replications with confidence interval computation
- Time-weighted statistics for robust average calculations

---

## Event Types

The model defines **17 discrete event types** that can occur, organized into **conception** (females become pregnant) and **birth** (pregnant females give birth) events:

### Conception Events (2 types)
| Event | Description | Rate Formula |
|-------|-------------|--------------|
| `REPRODUCTION_H` | Herbivore conception | `r_H × h_fert_factor × H_F × can_mate_H` |
| `REPRODUCTION_C` | Carnivore conception | `r_C × c_fert_factor × C_F × can_mate_C` |

**Notes:**
- Conception moves a non-pregnant female to the pregnant compartment
- Requires at least one male present (`can_mate_X = 1` if males exist)
- Only non-pregnant females can conceive

### Birth Events (4 types)
| Event | Description | Rate Formula |
|-------|-------------|--------------|
| `BIRTH_H_M` | Herbivore male birth | `0.5 × (1/gestation_H) × H_F_preg` |
| `BIRTH_H_F` | Herbivore female birth | `0.5 × (1/gestation_H) × H_F_preg` |
| `BIRTH_C_M` | Carnivore male birth | `0.5 × (1/gestation_C) × C_F_preg` |
| `BIRTH_C_F` | Carnivore female birth | `0.5 × (1/gestation_C) × C_F_preg` |

**Notes:**
- Birth rates proportional to **pregnant female population**
- Rate = `1/gestation_period` implements exponential gestation duration (Markovian approximation)
- 50% probability for each sex at birth
- Upon birth: pregnant female returns to non-pregnant pool + offspring added
- Gestation periods: `gestation_H` (typically 6.0), `gestation_C` (typically 5.0)

### Death Events (6 types)
| Event | Description | Rate Formula |
|-------|-------------|--------------|
| `DEATH_H_M` | Herbivore male natural death | `m_H × h_mort_factor × H_M` |
| `DEATH_H_F` | Herbivore female natural death | `m_H × h_mort_factor × H_F` |
| `DEATH_H_F_PREG` | Pregnant herbivore female death | `m_H × h_mort_factor × H_F_preg` |
| `DEATH_C_M` | Carnivore male natural death | `m_C × c_mort_factor × C_M` |
| `DEATH_C_F` | Carnivore female natural death | `m_C × c_mort_factor × C_F` |
| `DEATH_C_F_PREG` | Pregnant carnivore female death | `m_C × c_mort_factor × C_F_preg` |

**Notes:**
- Linear mortality rates (per capita)
- Modified by competition factors
- **Pregnant females subject to same mortality rates** (pregnancy risk)
- Death of pregnant female results in loss of both mother and unborn offspring

### Interaction Events (5 types)
| Event | Description | Rate Formula | Effect |
|-------|-------------|--------------|--------|
| `PREDATION` | Carnivore eats herbivore | `pred × H_total × C_total` | H decreases (sex chosen probabilistically) |
| `CONFLICT_H_M` | Male herbivore intraspecific conflict | `conflict_H × H_M × (H_M - 1)` | H_M decreases by 1 |
| `CONFLICT_H_F` | Female herbivore intraspecific conflict | `conflict_H × H_F × (H_F - 1)` | H_F decreases by 1 |
| `CONFLICT_C_M` | Male carnivore intraspecific conflict | `conflict_C × C_M × (C_M - 1)` | C_M decreases by 1 |
| `CONFLICT_C_F` | Female carnivore intraspecific conflict | `conflict_C × C_F × (C_F - 1)` | C_F decreases by 1 |

**Notes:**
- Predation uses mass action kinetics (H × C)
- Conflicts scale with density squared (N × (N-1))
- Conflicts represent territorial/resource disputes

---

## Main Data Structures

### 1. `EventType` (Enum)
```python
class EventType(Enum):
    # Conception (females become pregnant)
    REPRODUCTION_H = 1, REPRODUCTION_C = 2
    
    # Birth (pregnant females give birth)
    BIRTH_H_M = 3, BIRTH_H_F = 4, BIRTH_C_M = 5, BIRTH_C_F = 6
    
    # Death (including pregnant females)
    DEATH_H_M = 7, DEATH_H_F = 8, DEATH_H_F_PREG = 9
    DEATH_C_M = 10, DEATH_C_F = 11, DEATH_C_F_PREG = 12
    
    # Interactions
    PREDATION = 13
    CONFLICT_H_M = 14, CONFLICT_H_F = 15, CONFLICT_C_M = 16, CONFLICT_C_F = 17
```
**Purpose**: Enumerate all possible transition types in the Markov chain (17 total events)

### 2. `Event` Class
```python
class Event:
    - event_type: EventType
    - time: float
```
**Purpose**: Represent a scheduled discrete event with timing information

### 3. `Metrics` Class
```python
class Metrics:
    # Population trajectories (including pregnant compartments)
    - history_time: list[float]
    - history_H_M: list[int]
    - history_H_F: list[int]
    - history_H_F_preg: list[int]  # NEW: Pregnant herbivore females
    - history_C_M: list[int]
    - history_C_F: list[int]
    - history_C_F_preg: list[int]  # NEW: Pregnant carnivore females
    
    # Event counters (including conceptions)
    - counts: dict[str, int]
        * 'conceptions_H', 'conceptions_C'  # NEW: Conception events
        * 'births_H', 'births_C'
        * 'deaths_H', 'deaths_C'
        * 'predations'
        * 'conflicts_H', 'conflicts_C'
    
    # Time-weighted statistics (including pregnant compartments)
    - last_time: float
    - area_under_herbivore_males: float
    - area_under_herbivore_females: float
    - area_under_herbivore_females_preg: float  # NEW
    - area_under_carnivore_males: float
    - area_under_carnivore_females: float
    - area_under_carnivore_females_preg: float  # NEW
```
**Purpose**: Collect simulation output data for analysis

**Key Methods:**
- `record_state(t, H_M, H_F, C_M, C_F)`: Store population snapshot
- `update_time_weighted_stats(...)`: Accumulate area under curve for time-weighted averages

### 4. `SimulationEngine` Class
```python
class SimulationEngine:
    # State variables (with pregnant compartments)
    - H_M, H_F, H_F_preg: int  # Herbivore males/females/pregnant females
    - C_M, C_F, C_F_preg: int  # Carnivore males/females/pregnant females
    - current_time: float
    - t_max: float
    
    # Configuration
    - params: Dict[str, float]
    - metrics: Metrics
```
**Purpose**: Core simulation engine implementing CTMC dynamics

**Key Methods:**
- `calculate_rates()`: Compute transition rates for all events
- `step()`: Execute one Gillespie step (draw time, select event)
- `handle_event(event)`: Update state based on event type
- `event_loop()`: Main simulation loop until t_max

### 5. Parameter Dictionary
```python
parameters = {
    'r_H': 0.5,              # Herbivore conception rate (per non-pregnant female)
    'r_C': 0.4,              # Carnivore conception rate (per non-pregnant female)
    'gestation_H': 6.0,      # Herbivore gestation period (mean time)
    'gestation_C': 5.0,      # Carnivore gestation period (mean time)
    'm_H': 0.02,             # Herbivore mortality rate (optimized)
    'm_C': 0.03,             # Carnivore mortality rate
    'pred': 0.0007,          # Predation efficiency (optimized)
    'conflict_H': 0.00005,   # Herbivore conflict rate
    'conflict_C': 0.0005,    # Carnivore conflict rate
    'H_threshold': 200,      # Herbivore resource limit
    'HC_ratio_threshold': 3.0 # Minimum H/C ratio for carnivore viability
}
```

### 6. `ConfidenceInterval` Class (External)
```python
from confidence_interval import ConfidenceInterval

ci_calculator = ConfidenceInterval(
    min_samples_count: int,
    max_interval_width: float,
    confidence_level: float
)
```
**Purpose**: Sequential computation of confidence intervals across multiple runs

**Key Methods:**
- `add_data_point(value)`: Add observation from one simulation run
- `compute_interval()`: Return (is_final, (lower, upper)) tuple

---

## Gestation Mechanics

### Two-Stage Reproduction Process

The model implements **explicit gestation periods** using a two-compartment system:

#### Stage 1: Conception
- **Event**: `REPRODUCTION_H` or `REPRODUCTION_C`
- **Condition**: Non-pregnant female + at least one male present
- **Rate**: `r_X × fertility_factor × H_F × can_mate`
- **Effect**: Female moves from non-pregnant (`X_F`) to pregnant (`X_F_preg`) compartment

#### Stage 2: Birth
- **Event**: `BIRTH_X_M` or `BIRTH_X_F`
- **Rate**: `(1 / gestation_X) × X_F_preg`
- **Effect**: Pregnant female returns to non-pregnant pool + offspring added (50% male/female)

### Markovian Approximation

Gestation periods are modeled using **exponential distributions**:
- **Mean gestation time**: `gestation_H` (herbivores) or `gestation_C` (carnivores)
- **Memoryless property**: Birth rate constant regardless of how long female has been pregnant
- **Biological realism**: While real gestations have fixed duration, this approximation:
  - Maintains Markov property (essential for CTMC framework)
  - Produces correct **mean** gestation duration
  - Allows variability in gestation times (stochastic realism)
  - Averages out over many pregnancies

### Key Implications

1. **Population Dynamics**:
   - Total population = Males + Non-pregnant females + Pregnant females
   - Pregnant females still subject to mortality (pregnancy risk)
   - Death of pregnant female = loss of mother + unborn offspring

2. **Regulatory Feedback**:
   - Conception rate regulated by **non-pregnant** female count
   - Shorter gestation → faster population response → **reduced oscillations**
   - Longer gestation → delayed births → **amplified oscillations**

3. **Stability Impact**:
   - **Gestation delays are critical for stability** (2nd most sensitive parameter)
   - Optimal values: `gestation_H = 6.0`, `gestation_C = 5.0`
   - Too long → system becomes oscillatory and unstable

---

## Competition Mechanics

### Herbivore Competition (Resource Limitation)
**Trigger**: When `H_total > H_threshold` (default: 200)

**Mechanism**: Logistic decay functions
```python
k_H = 0.1  # Steepness parameter
h_fert_factor = 1.0 / (1.0 + exp(k_H × (H_total - H_threshold)))
h_mort_factor = 1.0 + 1.0 / (1.0 + exp(-k_H × (H_total - H_threshold)))
```

**Effects**:
- **Fertility reduction**: Birth rates decrease as H exceeds threshold
- **Mortality increase**: Death rates increase as H exceeds threshold
- **Smooth transition**: Logistic function provides gradual rather than abrupt change

### Carnivore Competition (Prey Scarcity)
**Trigger**: When `H_total / C_total < HC_ratio_threshold` (default: 3.0)

**Mechanism**: Logistic decay functions
```python
k_C = 2.0  # Steepness parameter
c_fert_factor = 1.0 / (1.0 + exp(-k_C × (ratio_HC - HC_ratio_threshold)))
c_mort_factor = 1.0 + 1.0 / (1.0 + exp(k_C × (ratio_HC - HC_ratio_threshold)))
```

**Effects**:
- **Fertility reduction**: Birth rates decrease when prey is scarce
- **Mortality increase**: Death rates increase when food is insufficient
- **Ratio-dependent**: Uses H/C ratio rather than absolute counts

---

## Gillespie Algorithm Implementation

### Step-by-Step Process

1. **Calculate Reaction Rates**
   - Compute rate for each of 13 event types
   - Apply competition modifiers
   - Sum to get total rate: `λ_total = Σ λ_i`

2. **Draw Time to Next Event**
   ```python
   dt = Exponential(1 / λ_total)
   ```
   - Memoryless property satisfies Markov assumption
   - Check if `current_time + dt > t_max` (termination)

3. **Select Event Type**
   - Compute probability: `p_i = λ_i / λ_total`
   - Draw categorical: `event ~ Categorical(p_1, p_2, ..., p_13)`

4. **Update State**
   - Increment/decrement appropriate population counter
   - Record event in metrics
   - Update time-weighted statistics

5. **Record State**
   - Append current populations to history
   - Continue loop

### Termination Conditions
- Time limit reached: `current_time ≥ t_max`
- Extinction: `λ_total = 0` (no possible events)

---

## Key Performance Indicators (KPIs)

### 1. Event Count Metrics
Tracked in `metrics.counts` dictionary:

| KPI | Description | Interpretation |
|-----|-------------|----------------|
| `births_H` | Total herbivore births | Reproductive success |
| `births_C` | Total carnivore births | Predator reproductive success |
| `deaths_H` | Total herbivore deaths (natural) | Background mortality |
| `deaths_C` | Total carnivore deaths (natural) | Background mortality |
| `predations` | Total predation events | Trophic interaction frequency |
| `conflicts_H` | Herbivore intraspecific conflicts | Density-dependent mortality |
| `conflicts_C` | Carnivore intraspecific conflicts | Density-dependent mortality |

### 2. Time-Weighted Average Populations
**Most important KPIs** - computed via trapezoidal integration:

$$\bar{N} = \frac{1}{T} \int_0^T N(t) \, dt$$

| KPI | Formula | Purpose |
|-----|---------|---------|
| `avg_H_M` | `area_under_herbivore_males / t_max` | Average male herbivore population |
| `avg_H_F` | `area_under_herbivore_females / t_max` | Average female herbivore population |
| `avg_H_F_preg` | `area_under_herbivore_females_preg / t_max` | Average pregnant herbivore population |
| `avg_C_M` | `area_under_carnivore_males / t_max` | Average male carnivore population |
| `avg_C_F` | `area_under_carnivore_females / t_max` | Average female carnivore population |
| `avg_C_F_preg` | `area_under_carnivore_females_preg / t_max` | Average pregnant carnivore population |
| `avg_H_total` | `avg_H_M + avg_H_F + avg_H_F_preg` | Total average herbivore population |
| `avg_C_total` | `avg_C_M + avg_C_F + avg_C_F_preg` | Total average carnivore population |

**Why time-weighted?**
- Accounts for variable time steps in CTMC
- More robust than simple arithmetic mean
- Represents true average occupancy over time

### 3. Final State Metrics
Instantaneous measurements at `t = t_max`:

| KPI | Description |
|-----|-------------|
| `final_H_M` | Herbivore males at end |
| `final_H_F` | Herbivore females at end |
| `final_C_M` | Carnivore males at end |
| `final_C_F` | Carnivore females at end |
| `final_H_total` | Total herbivores at end |
| `final_C_total` | Total carnivores at end |

**Note**: Final values have high stochastic variance; time-weighted averages are preferred for analysis.

### 4. Derived Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| H/C Ratio (avg) | `avg_H_total / avg_C_total` | Average predator-prey balance |
| H/C Ratio (final) | `final_H_total / final_C_total` | End-state balance |
| Net H growth | `births_H - deaths_H - predations - conflicts_H` | Total herbivore change |
| Net C growth | `births_C - deaths_C - conflicts_C` | Total carnivore change |

### 5. Confidence Interval Metrics
When running multiple replications:

| KPI | Description | Status |
|-----|-------------|--------|
| CI Width | `upper - lower` | Precision of estimate |
| CI Status | ✓ FINAL / → CONVERGING | Whether width < 5% of mean |
| Mean | `Σ x_i / n` | Point estimate across runs |
| Std Dev | `√(Σ(x_i - μ)² / n)` | Variability between runs |

**Convergence Criterion**: CI width ≤ 5% of mean value

---

## Statistical Analysis Framework

### Multiple Simulation Runs
Function: `run_multiple_simulations(num_runs, base_seed, ...)`

**Purpose**: Quantify stochastic variability and compute confidence intervals

**Process**:
1. **Initialize CI Calculators**
   - Create `ConfidenceInterval` object for each KPI
   - Set convergence parameters (min samples, max width, confidence level)

2. **Run Independent Replications**
   - Use different random seeds: `seed = base_seed + i`
   - Each run starts from same initial conditions
   - Extract time-weighted averages from each run

3. **Accumulate Data**
   - Add metrics from each run to CI calculators
   - Check convergence after each addition

4. **Compute Confidence Intervals**
   - For converged metrics: report (lower, upper, mean)
   - For non-converged: report status (→ CONVERGING)

5. **Visualize Uncertainty**
   - Plot last run trajectory
   - Overlay mean reference lines
   - Add shaded 95% CI bands

### Output Structure
```
================================================================================
CONFIDENCE INTERVAL RESULTS (95% confidence, 40 runs)
================================================================================

--- TIME-WEIGHTED AVERAGES ---

Avg H M              ✓ FINAL         Mean:   88.24  CI: [  87.74,   88.73]  Width:   0.99
Avg H F              ✓ FINAL         Mean:   88.22  CI: [  87.41,   89.03]  Width:   1.62
...

Summary Statistics Across All Runs:
================================================================================
Avg H M              Mean:   88.24  Std:   1.58  Min:   85.32  Max:   91.43
...
```

---

## Visualization

### 1. Population Dynamics Plot (with CI bands)
**X-axis**: Time (0 to t_max)  
**Y-axis**: Population count

**Elements**:
- Solid green line: Herbivore trajectory (last run)
- Solid red line: Carnivore trajectory (last run)
- Dotted horizontal lines: Mean values across all runs
- Shaded regions: 95% confidence intervals
- Gray dashed line: H_threshold reference

**Interpretation**:
- Check if last run falls within CI bands (typical behavior)
- Assess CI width (system stability)
- Identify transient vs. equilibrium phases

### 2. Phase Portrait (H vs C)
**X-axis**: Herbivore population  
**Y-axis**: Carnivore population

**Elements**:
- Blue trajectory: Path through state space
- Green dot: Initial state
- Red X: Final state

**Interpretation**:
- Cyclic patterns: Predator-prey oscillations
- Convergence to point: Stable equilibrium
- Divergence: Extinction or explosion

---

## Model Parameters and Sensitivity

### Default Configuration
```python
initial_pop = {'H_M': 50, 'H_F': 50, 'C_M': 10, 'C_F': 10}
t_max = 100.0
num_runs = 40
confidence_level = 0.95
max_interval_width = 0.05  # 5% of mean
```

### Critical Parameters

| Parameter | Effect | Sensitivity |
|-----------|--------|-------------|
| `r_H` | Higher → More herbivore births | High |
| `r_C` | Higher → More carnivore births | High |
| `pred` | Higher → More predation events | Very High |
| `H_threshold` | Lower → Earlier competition effects | Medium |
| `HC_ratio_threshold` | Higher → Carnivores need more prey | High |
| `conflict_X` | Higher → More intraspecific mortality | Medium |

### Typical Outcomes
- **Stable coexistence**: H ≈ 170-180, C ≈ 40-45, ratio ≈ 4:1
- **All CIs converge**: With 40 runs, width < 2 individuals
- **Low variability**: Std dev ≈ 1-3 individuals for time-weighted averages

---

## Usage Modes

### Single Run Mode (Detailed)
```python
MODE = "single"
```
**Output**:
- Complete statistics table
- Event counts and time-weighted averages
- Final state
- H/C ratio
- Two plots (dynamics + phase portrait)

**Use case**: Initial exploration, debugging, visualization

### Multiple Run Mode (Statistical)
```python
MODE = "multiple"
NUM_RUNS = 40
SHOW_PLOTS = True
VERBOSE = False
```
**Output**:
- Progress indicator for each run
- Confidence interval results
- Summary statistics (mean, std, min, max)
- Enhanced plot with CI bands (if SHOW_PLOTS=True)
- Per-run details (if VERBOSE=True)

**Use case**: Hypothesis testing, parameter estimation, publication-quality results

---

## Scientific Relevance

### Ecological Modeling
- Captures predator-prey dynamics (Lotka-Volterra extension)
- Includes sex structure (reproductive realism)
- Implements carrying capacity (resource limitation)
- Models intraspecific competition (territoriality)

### Stochastic Processes
- Demonstrates CTMC properties (memorylessness, state-dependent rates)
- Gillespie algorithm (exact stochastic simulation)
- Ergodicity analysis (phase portraits)
- Statistical inference (confidence intervals)

### Computational Methods
- Event-driven simulation architecture
- Efficient rate calculation
- Time-weighted metric computation
- Parallel replication framework

---

## Key Assumptions

1. **Well-mixed population**: No spatial structure
2. **Instant events**: Conception, birth, death, predation occur instantaneously
3. **Mass action kinetics**: Encounter rates proportional to population products
4. **Exponential distributions**: Inter-event times and **gestation periods** are memoryless
5. **Equal sex ratio**: 50% chance of male/female at birth
6. **Monogamous constraint**: Males required for reproduction (conception)
7. **Two-compartment reproduction**: Non-pregnant and pregnant females tracked separately
8. **Pregnancy mortality**: Pregnant females subject to same mortality rates as non-pregnant
9. **No age structure**: All individuals have same rates (except pregnancy state)
10. **Closed system**: No migration

---

## Parameter Sensitivity Analysis

### Overview

A **comprehensive sensitivity analysis** was conducted to identify critical parameters affecting system stability and optimize the model configuration. The analysis involved:
- **9 parameters tested**: All reproduction, mortality, gestation, and interaction rates
- **Systematic sweeps**: 9 values per parameter across biologically realistic ranges
- **Monte Carlo replication**: 10-30 independent runs per configuration
- **Stability metric**: Coefficient of Variation (CV = std/mean), target < 0.10

### Sensitivity Ranking (by CV Range)

Parameters ranked by their impact on system stability:

| Rank | Parameter | CV Range | Sensitivity | Description |
|------|-----------|----------|-------------|-------------|
| 1 | **m_H** | **0.617** | **CRITICAL** | Herbivore mortality rate |
| 2 | **gestation_H** | **0.529** | **CRITICAL** | Herbivore gestation period |
| 3 | **pred** | **0.392** | **CRITICAL** | Predation rate |
| 4 | m_C | 0.134 | HIGH | Carnivore mortality rate |
| 5 | r_C | 0.089 | MODERATE | Carnivore conception rate |
| 6 | gestation_C | 0.076 | MODERATE | Carnivore gestation period |
| 7 | r_H | 0.069 | MODERATE | Herbivore conception rate |
| 8 | conflict_C | 0.031 | LOW | Carnivore conflict rate |
| 9 | conflict_H | 0.030 | LOW | Herbivore conflict rate |

**CV Range** = max(CV) - min(CV) across tested values; larger range = more sensitive parameter

### Critical Parameter Insights

#### 1. Herbivore Mortality (m_H) - MOST CRITICAL
- **Impact**: Single most important parameter for system stability
- **Optimal value**: 0.02 (carefully balanced)
- **Effect**: Too low → herbivore explosion; too high → extinction
- **Recommendation**: **Calibrate first** when fitting to empirical data

#### 2. Herbivore Gestation (gestation_H) - CRITICAL
- **Impact**: Gestation delays amplify population oscillations
- **Optimal value**: 6.0 (shorter = more stable)
- **Effect**: Longer gestation → delayed births → larger oscillations
- **Mechanism**: Time lags in feedback loops create instability
- **Recommendation**: **Keep short** for stable dynamics

#### 3. Predation Rate (pred) - CRITICAL
- **Impact**: Controls coupling strength between species
- **Optimal value**: 0.0007 (KEY OPTIMIZATION)
- **Effect**: Too high → strong oscillations; too low → decoupling
- **Breakthrough**: Reducing from 0.001 to 0.0007 (-30%) improves CV by ~20%
- **Recommendation**: **Fine-tune carefully** for desired H/C ratio

### Optimized Configuration

Based on sensitivity analysis, the **optimal parameter set** achieves:

**Performance Metrics**:
- **CV_H = 0.036** (herbivore stability, target < 0.10) ✓
- **CV_C = 0.034** (carnivore stability, target < 0.10) ✓
- **H/C ratio = 3.26** (near target of 3.0) ✓
- **Extinction risk = 0%** (across 30 runs) ✓
- **Population levels**: H = 158.34 ± 5.64, C = 48.62 ± 1.65

**Optimized Parameters**:
```python
parameters = {
    'r_H': 0.5,           # Herbivore conception rate
    'r_C': 0.4,           # Carnivore conception rate
    'gestation_H': 6.0,   # Herbivore gestation (CRITICAL: keep short)
    'gestation_C': 5.0,   # Carnivore gestation
    'm_H': 0.02,          # Herbivore mortality (MOST CRITICAL)
    'm_C': 0.03,          # Carnivore mortality
    'pred': 0.0007,       # Predation rate (KEY OPTIMIZATION: -30%)
    'conflict_H': 0.00005,# Herbivore conflict (insensitive)
    'conflict_C': 0.0005, # Carnivore conflict (insensitive)
    'H_threshold': 200,
    'HC_ratio_threshold': 3.0
}
```

### Parameter Tuning Strategy

When adapting the model or fitting to data:

**Priority 1 - Critical Tier** (CV range > 0.3):
1. **m_H** - Calibrate first; small changes have large effects
2. **gestation_H** - Keep short; longer values amplify oscillations
3. **pred** - Fine-tune for desired H/C balance

**Priority 2 - Moderate Tier** (CV range 0.07-0.14):
4. **m_C** - Adjust for carnivore viability
5. **r_C, gestation_C, r_H** - Secondary tuning for population levels

**Priority 3 - Insensitive Tier** (CV range < 0.04):
8. **conflict_H, conflict_C** - Minimal impact; can be ignored

### Validation Protocol

When testing new parameter combinations:
1. **Run 20-30 independent simulations** (different seeds)
2. **Require CV < 0.10** for both species (stability criterion)
3. **Check extinction rate < 10%** (viability criterion)
4. **Verify H/C ratio ≈ 3.0 ± 20%** (ecological balance)
5. **Use t_max ≥ 500** (ensure steady-state reached)

### Tools for Sensitivity Analysis

Two analysis scripts provided:

1. **parameter_sensitivity_analysis.py**:
   - Systematic parameter sweeps (9 values × 9 parameters)
   - Generates sensitivity plots for each parameter
   - Identifies optimal values via stability scoring
   - Output: 9 PNG plots + sensitivity ranking

2. **quick_parameter_test.py**:
   - Fast comparison of pre-defined configurations
   - 20 runs per configuration for quick assessment
   - Output: Comparison bar plot + recommendation

---

## Extensions and Future Work

### Potential Enhancements
- **Spatial structure**: Grid-based or network topology (re-test top 3 sensitive parameters)
- **Age structure**: Juvenile vs. adult rates (may affect mortality sensitivity)
- **Fixed gestation periods**: Implement deterministic gestation using event scheduling
- **Environmental stochasticity**: Time-varying parameters (focus on m_H, pred)
- **Additional species**: Multi-trophic food web (requires new sensitivity analysis)
- **Genetics**: Trait evolution over time
- **Allee effects**: Minimum viable population thresholds

### Analysis Extensions
- ✓ **Sensitivity analysis**: Completed for all 9 parameters
- **Two-parameter surfaces**: Explore interactions (e.g., m_H × pred)
- **Temporal CI bands**: Show how uncertainty evolves over time
- **Phase portrait CI ellipses**: 2D confidence regions
- **Extinction time distribution**: Conditional on parameter values
- **Bifurcation analysis**: Identify critical thresholds for m_H, gestation_H
- **Parameter fitting**: Use sensitivity ranking to prioritize estimable parameters

---

## References and Theory

### Mathematical Foundation
- **Gillespie, D. T. (1977)**: "Exact stochastic simulation of coupled chemical reactions"
- **Markov Property**: Future state depends only on present state, not history
- **Rate Functions**: $\lambda(X) = \sum_i \lambda_i(X)$ where X is current state
- **Time to Next Event**: $\tau \sim \text{Exp}(\lambda(X))$
- **Event Selection**: $\mathbb{P}(\text{event } i) = \lambda_i(X) / \lambda(X)$

### Ecological Theory
- **Lotka-Volterra Model**: Classic predator-prey ODEs
- **Logistic Growth**: Resource limitation via carrying capacity
- **Functional Response**: Predation rate as function of prey density
- **Numerical Response**: Predator birth rate as function of prey availability

---

## Computational Complexity

### Time Complexity
- **Per step**: O(k) where k = number of event types (13)
- **Total**: O(n × k) where n = number of events (depends on rates and t_max)
- **Multiple runs**: O(m × n × k) where m = num_runs

### Space Complexity
- **State variables**: O(1) - fixed number of counters
- **History storage**: O(n) - one entry per event
- **Multiple runs**: O(m × n) - history for all runs

### Performance Characteristics
- **Typical run time**: < 1 second for t_max=100 (single run)
- **40 runs**: ~10-20 seconds total
- **Bottleneck**: Event loop iteration count (depends on rates)
- **Optimization**: Pre-allocate history arrays if size known

---

## Summary

This CTMC model provides a **rigorous stochastic framework** for studying predator-prey dynamics with:

✅ **Sex-structured populations** (males and females)  
✅ **Explicit gestation periods** (two-compartment reproduction)  
✅ **Density-dependent regulation** (competition thresholds)  
✅ **Multiple interaction types** (predation, conflict)  
✅ **Exact stochastic simulation** (Gillespie algorithm)  
✅ **Statistical rigor** (confidence intervals from replications)  
✅ **Time-weighted metrics** (robust average calculations)  
✅ **Visual uncertainty representation** (CI bands on plots)  
✅ **Comprehensive parameter sensitivity analysis** (9 parameters ranked)  
✅ **Optimized configuration** (CV < 0.04, 0% extinction risk)

The model is suitable for:
- Educational purposes (teaching CTMC, Gillespie methods, sensitivity analysis)
- Research applications (ecological modeling, parameter estimation, stability analysis)
- Hypothesis testing (effect of gestation delays, competition, predation on stability)
- Stochastic process exploration (ergodicity, extinction, oscillations)
- Conservation biology (assessing population viability under parameter uncertainty)

**Key Scientific Contributions**:
1. **Gestation mechanics**: Demonstrates impact of reproductive delays on stability
2. **Sensitivity ranking**: Identifies m_H > gestation_H > pred as critical parameters
3. **Optimization**: Systematic approach to finding stable parameter combinations
4. **Validation protocol**: Statistical framework for assessing model reliability

**Output**: Publication-quality results with quantified uncertainty and comprehensive parameter characterization, enabling defensible scientific claims about population dynamics under stochastic conditions with realistic reproductive processes.
