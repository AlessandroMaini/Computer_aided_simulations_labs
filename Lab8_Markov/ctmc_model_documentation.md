# Carnivore-Herbivore CTMC Model Documentation

## Overview

This document describes a **Continuous Time Markov Chain (CTMC)** simulation modeling predator-prey population dynamics between herbivores (H) and carnivores (C), with explicit tracking of male and female populations. The model implements stochastic population processes using the Gillespie algorithm with density-dependent competition effects.

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

The model defines **13 discrete event types** that can occur:

### Birth Events (4 types)
| Event | Description | Rate Formula |
|-------|-------------|--------------|
| `BIRTH_H_M` | Herbivore male birth | `0.5 × r_H × h_fert_factor × H_F × can_mate_H` |
| `BIRTH_H_F` | Herbivore female birth | `0.5 × r_H × h_fert_factor × H_F × can_mate_H` |
| `BIRTH_C_M` | Carnivore male birth | `0.5 × r_C × c_fert_factor × C_F × can_mate_C` |
| `BIRTH_C_F` | Carnivore female birth | `0.5 × r_C × c_fert_factor × C_F × can_mate_C` |

**Notes:**
- Birth rates proportional to female population (assuming sufficient males)
- 50% probability for each sex at birth
- `can_mate_X` is 1 if males exist, 0 otherwise

### Death Events (4 types)
| Event | Description | Rate Formula |
|-------|-------------|--------------|
| `DEATH_H_M` | Herbivore male natural death | `m_H × h_mort_factor × H_M` |
| `DEATH_H_F` | Herbivore female natural death | `m_H × h_mort_factor × H_F` |
| `DEATH_C_M` | Carnivore male natural death | `m_C × c_mort_factor × C_M` |
| `DEATH_C_F` | Carnivore female natural death | `m_C × c_mort_factor × C_F` |

**Notes:**
- Linear mortality rates (per capita)
- Modified by competition factors

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
    BIRTH_H_M = 1, BIRTH_H_F = 2, BIRTH_C_M = 3, BIRTH_C_F = 4
    DEATH_H_M = 5, DEATH_H_F = 6, DEATH_C_M = 7, DEATH_C_F = 8
    PREDATION = 9
    CONFLICT_H_M = 10, CONFLICT_H_F = 11, CONFLICT_C_M = 12, CONFLICT_C_F = 13
```
**Purpose**: Enumerate all possible transition types in the Markov chain

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
    # Population trajectories
    - history_time: list[float]
    - history_H_M: list[int]
    - history_H_F: list[int]
    - history_C_M: list[int]
    - history_C_F: list[int]
    
    # Event counters
    - counts: dict[str, int]
        * 'births_H', 'births_C'
        * 'deaths_H', 'deaths_C'
        * 'predations'
        * 'conflicts_H', 'conflicts_C'
    
    # Time-weighted statistics
    - last_time: float
    - area_under_herbivore_males: float
    - area_under_herbivore_females: float
    - area_under_carnivore_males: float
    - area_under_carnivore_females: float
```
**Purpose**: Collect simulation output data for analysis

**Key Methods:**
- `record_state(t, H_M, H_F, C_M, C_F)`: Store population snapshot
- `update_time_weighted_stats(...)`: Accumulate area under curve for time-weighted averages

### 4. `SimulationEngine` Class
```python
class SimulationEngine:
    # State variables
    - H_M, H_F: int  # Herbivore males/females
    - C_M, C_F: int  # Carnivore males/females
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
    'r_H': 0.6,              # Herbivore reproduction rate
    'r_C': 0.4,              # Carnivore reproduction rate
    'm_H': 0.1,              # Herbivore mortality rate
    'm_C': 0.1,              # Carnivore mortality rate
    'pred': 0.003,           # Predation efficiency
    'conflict_H': 0.0001,    # Herbivore conflict rate
    'conflict_C': 0.002,     # Carnivore conflict rate
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
| `avg_C_M` | `area_under_carnivore_males / t_max` | Average male carnivore population |
| `avg_C_F` | `area_under_carnivore_females / t_max` | Average female carnivore population |
| `avg_H_total` | `avg_H_M + avg_H_F` | Total average herbivore population |
| `avg_C_total` | `avg_C_M + avg_C_F` | Total average carnivore population |

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
2. **Instant events**: Birth, death, predation occur instantaneously
3. **Mass action kinetics**: Encounter rates proportional to population products
4. **Exponential distributions**: Inter-event times are memoryless
5. **Equal sex ratio**: 50% chance of male/female at birth
6. **Monogamous constraint**: Males required for reproduction
7. **No age structure**: All individuals have same rates
8. **Closed system**: No migration

---

## Extensions and Future Work

### Potential Enhancements
- **Spatial structure**: Grid-based or network topology
- **Age structure**: Juvenile vs. adult rates
- **Environmental stochasticity**: Time-varying parameters
- **Additional species**: Multi-trophic food web
- **Genetics**: Trait evolution over time
- **Allee effects**: Minimum viable population thresholds

### Analysis Extensions
- **Sensitivity analysis**: Parameter sweep with CI comparison
- **Temporal CI bands**: Show how uncertainty evolves over time
- **Phase portrait CI ellipses**: 2D confidence regions
- **Extinction probability**: Estimate from multiple runs
- **Bifurcation analysis**: Identify critical parameter values

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
✅ **Density-dependent regulation** (competition thresholds)  
✅ **Multiple interaction types** (predation, conflict)  
✅ **Exact stochastic simulation** (Gillespie algorithm)  
✅ **Statistical rigor** (confidence intervals from replications)  
✅ **Time-weighted metrics** (robust average calculations)  
✅ **Visual uncertainty representation** (CI bands on plots)

The model is suitable for:
- Educational purposes (teaching CTMC and Gillespie methods)
- Research applications (ecological modeling, parameter estimation)
- Hypothesis testing (effect of competition on stability)
- Stochastic process exploration (ergodicity, extinction)

**Output**: Publication-quality results with quantified uncertainty, enabling defensible scientific claims about population dynamics under stochastic conditions.
