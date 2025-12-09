# Carnivore-Herbivore CTMC Simulation

## How to Run

### Single Simulation Run
```bash
python carnivore_herbivore.py
```
Set `MODE = "single"` in the file (line 616) for a single detailed run with plots.

### Multiple Runs with Confidence Intervals
```bash
python carnivore_herbivore.py
```
Set `MODE = "multiple"` in the file (line 616, default) for multiple runs with statistical analysis.

## Configuration

Edit parameters in the `parameters` dictionary (lines 621-640):
- Initial populations: `initial_pop` (line 619)
- Reproduction rates, mortality rates, predation, conflicts, thresholds

## Output

- **Single mode**: Population plot saved as `population_dynamics.png`
- **Multiple mode**: Envelope plot with confidence intervals saved as `population_dynamics.png`

## Requirements

```
numpy
matplotlib
confidence_interval (included)
```
