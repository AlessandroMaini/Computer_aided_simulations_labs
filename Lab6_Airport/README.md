# Airport Departure Simulation

## How to Run

Run the simulation by executing:
```bash
python airport_departure.py
```

### Execution Modes

Set the `MODE` variable at **line 1200**:
- `MODE = "single"` - Single detailed run with interactive plots
- `MODE = "multiple"` - Multiple runs (default) with statistical analysis

## Configuration

All simulation parameters are located in the `if __name__ == "__main__"` block:

**Single mode parameters** (lines 1207-1243):
- **Temporal**: `SIMULATION_TIME` (24 hours), `FLIGHT_FREQUENCY` (every 20 min)
- **Resources**: Server counts for cashiers (land/air), security, and boarding
- **Service Rates**: Processing times for cashier, security, and boarding operations
- **Passenger Behavior**: Buying probability, companions, dwell times
- **Arrival Patterns**: Mean and standard deviation of arrival times before flight
- **Priority Classes**: Probabilities for Economy/Business/First Class
- **Deadlines**: Security, boarding window start, and boarding deadline times

**Multiple mode parameters** (lines 1257-1272):
- `num_runs` - Number of independent simulation runs
- `base_seed` - Base random seed for reproducibility

## Output

### Single Mode
- **Console**: Detailed statistics including queue metrics, wait times, server utilizations, and missed flights
- **Interactive Plots**:
  - Wait times by priority class (security and boarding)
  - Airport time series (total passengers and breakdown by location)

### Multiple Mode
- **Console**: 95% confidence intervals for key performance indicators:
  - Average security wait time
  - Average boarding wait time
  - Average total time (arrival to departure)
  - Drop percentage (missed flights)
  - Security and boarding server utilization

## Requirements

```
numpy
matplotlib
confidence_interval (included in directory)
```
