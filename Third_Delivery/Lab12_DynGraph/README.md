# Lab 12: Dynamic Graphs

## Description

`gnp.py` - Generates G(n,p) random graphs and analyzes their properties (connectivity, diameter, component sizes) for different values of n and p. Supports both linear (p = a/n) and logarithmic (p = a*log(n)/n) scaling.

`gnr.py` - Generates G(n,r) random geometric graphs where nodes are placed uniformly in a unit square and edges connect nodes within distance r. Analyzes graph properties for different values of n and r.

`majority_model.py` - Simulates the majority opinion dynamics model on G(n,r) graphs. Tracks how opinions spread and converge over time under different initial opinion probabilities.

## Requirements

```bash
pip install networkx numpy matplotlib tqdm
```

## Execution

```bash
# Generate and analyze G(n,p) graphs
python gnp.py

# Generate and analyze G(n,r) graphs
python gnr.py

# Run majority model simulations
python majority_model.py
```

Each script saves results to JSON files. The majority model also produces visualizations of opinion dynamics over time.
