# Lab 11: Epidemic Model with Vaccination

## Description

`epidemic_vax.py` - Simulates a SAIQRS-V epidemic model across 18 heterogeneous population groups (3 geographic areas × 2 age groups × 3 work exposure levels). Compares five vaccination strategies: No Vaccination, Old First, High Exposure First, Geographical, and Uniform.

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Execution

```bash
python epidemic_vax.py
```

The simulation runs for 3 years with vaccination starting at day 365. Generates plots for each strategy showing epidemic dynamics (asymptomatic, symptomatic, quarantined, recovered, deaths, vaccinated) and prints summary statistics.
