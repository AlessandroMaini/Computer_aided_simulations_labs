# Random Variable Generation and Simulation

This lab contains implementations of custom random variable generators and their applications in queueing and car-sharing simulations.

## Files Overview

### Core Components

- **`rv_generation.py`** - Random variable generator class implementing:
  - Hyperexponential distribution (composition method)
  - Erlang-k distribution (convolution method)
  - Pareto distribution (inverse transform method)

- **`confidence_interval.py`** - Statistical confidence interval calculator

### Testing and Validation

- **`independence_test.py`** - Auto-correlation tests for verifying RV independence
- **`distribution_test.py`** - Chi-square goodness-of-fit tests for distribution validation

### Simulation Applications

- **`basic_queue_system.py`** - M/M/1 queue simulator with multiple distribution types
- **`car_sharing.py`** - Car-sharing system simulation with spatial dynamics

---

## How to Run

### 1. Independence Tests

Tests auto-correlation to verify independence of generated random variables.

```bash
python independence_test.py
```

**Output**: 
- `Uniform_autocorrelation.png`
- `Hyperexponential_autocorrelation.png`
- `Erlang-K_autocorrelation.png`
- `Pareto_autocorrelation.png`

**Configuration**: Edit hardcoded parameters at lines 27-37 (sample sizes, distribution parameters).

---

### 2. Distribution Tests

Chi-square goodness-of-fit tests for generated distributions.

```bash
python distribution_test.py
```

**Output**: 
- `H.png` - Hyperexponential distribution comparison
- `E.png` - Erlang-K distribution comparison
- `P.png` - Pareto distribution comparison
- Console output with chi-square statistics and pass/fail results

**Configuration**: Edit test parameters at lines 119-143 (distribution parameters, significance level α=0.05).

---

### 3. Basic Queue System

Comparative analysis of queue performance under different interarrival/service distributions.

```bash
python basic_queue_system.py
```

**Output**: 
- `comparative_queue_performance.png` - Bar charts comparing queue length and waiting times across 10 distribution configurations
- Console output with detailed statistics table

**Configuration**: Parameters at lines 443-473:
- `SIM_TIME` - Simulation duration
- `NUM_SERVERS`, `QUEUE_CAPACITY` - System resources
- `NUM_RUNS` - Number of replications for averaging
- `MEAN_INTERARRIVAL`, `MEAN_SERVICE` - Target mean times
- Distribution configurations (10 scenarios testing various interarrival/service combinations)

---

### 4. Car Sharing System

Event-driven simulation of electric car-sharing with spatial dynamics, charging, and relocation.

```bash
python car_sharing.py
```

**Output**: 
- `arrival_pattern_histogram_{DISTRIBUTION}.png` - Histogram of interarrival times
- `transient_phase_detection_{DISTRIBUTION}.png` - Transient detection plot
- Console output with detailed statistics including:
  - User request metrics (total, abandoned, waiting times)
  - Vehicle metrics (availability, utilization, trip distances)
  - Charging station utilization
  - Confidence interval analysis (if enabled)

**Configuration**: Parameters at lines 1528-1572:
- **Fleet & Infrastructure** (lines 1530-1534): `NUM_CARS`, `NUM_STATIONS`, `MAX_STATION_CAPACITY`
- **Simulation** (line 1535): `SIMULATION_TIME` (7 days default)
- **Vehicle Parameters** (lines 1536-1542): Battery autonomy, charging rate, speed
- **User Parameters** (lines 1543-1545): Trip/pickup distances, max waiting time
- **Relocation** (lines 1546-1547): Interval and cars per relocation event
- **Statistical Analysis** (lines 1549-1555): Confidence interval settings
- **Interarrival Distribution** (lines 1557-1569): Choose distribution type (EXPONENTIAL, HYPEREXPONENTIAL, ERLANG_K, PARETO) by uncommenting desired configuration

**Note**: The current configuration (line 1568) uses PARETO distribution. Other distributions are commented out at lines 1558-1566.

---

## Distribution Implementations

### Hyperexponential
- **Method**: Composition (mixture of exponentials)
- **Parameters**: `lambdas` (list of rates), `probabilities` (mixture weights)
- **Use case**: High-variability processes

### Erlang-k
- **Method**: Convolution (sum of k exponentials)
- **Parameters**: `lambda` (rate), `k` (shape/phases)
- **Use case**: Low-variability processes

### Pareto
- **Method**: Inverse transform
- **Parameters**: `alpha` (shape), `scale` (minimum value)
- **Use case**: Heavy-tailed processes

---

## Requirements

```
numpy
scipy
matplotlib
confidence_interval (included in directory)
rv_generation (included in directory)
```
