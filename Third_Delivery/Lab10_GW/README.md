# Lab 10: Multi-type Galton-Watson Process

## Description

`mgw.py` - Simulates a multi-type Galton-Watson branching process with two classes of individuals (Class A and Class B). The script estimates extinction probabilities for different reproduction rate parameters (alpha values).

## Requirements

```bash
pip install numpy tqdm
```

## Execution

```bash
python mgw.py
```

The script will run 100 simulations for each alpha value (0.9, 0.95, 1.0, 1.05, 1.1) and output the estimated extinction probability and average last generation for each parameter.
