"""
Algorithm to detect the end of the transient phase in a stationary system.
 - For a certain measured quantity X, compute the average of X over a sliding window of time of duration T and stride S.
 - Store the last N computed averages of X.
 - If the average of the squared variations between each interval and the following (from 1-st to N-th) is less than a certain threshold P: 
        then the transient ends at the start of the 1-st interval.
 - Otherwise other intervals have to be computed.
"""

# --------------------------------------------------------------------
# TRANSIENT DETECTION
# --------------------------------------------------------------------
class TransientDetection:
    def __init__(self, window_size, stride, num_intervals, threshold):
        self.window_size = window_size  # Duration of the sliding window T
        self.stride = stride              # Stride S
        self.num_intervals = num_intervals  # Number of intervals N
        self.threshold = threshold        # Variation threshold P
        self.averages: dict[float, float] = {}  # Dictionary to store computed averages (start_time: average)
        self.values: dict[float, float] = {}  # Dictionary to store measured values (time: value)

    def add_value(self, time, value):
        """Add a new measured value."""
        self.values[time] = value

    def compute_next_interval_start(self):
        """Compute the start time of the next interval based on the last computed average."""
        if not self.averages:
            return min(self.values.keys())  # Start from the first time if no averages yet

        last_interval_start = max(self.averages.keys())
        next_interval_start = last_interval_start + self.stride
        return next_interval_start

    def compute_average(self):
        """Compute the average of the last window_size time interval."""
        if not self.values:
            return None

        current_time = max(self.values.keys())
        window_start = current_time - self.window_size
        relevant_values = [v for t, v in self.values.items() if window_start <= t <= current_time]

        if not relevant_values:
            return None

        average = sum(relevant_values) / len(relevant_values)
        return average

    def add_average(self, time, average):
        """Add a new computed average to the dictionary."""
        self.averages[time] = average
        if len(self.averages) > self.num_intervals:
            # Remove the oldest average
            oldest_time = min(self.averages.keys())
            del self.averages[oldest_time]

    def is_transient_over(self):
        """Check if the transient phase is over."""
        if len(self.averages) < self.num_intervals:
            return False  # Not enough data to determine

        variations = []
        avg_values = list(self.averages.values())
        for i in range(len(avg_values) - 1):
            variation = (avg_values[i + 1] - avg_values[i]) ** 2
            variations.append(variation)
        squared_sum = sum(variations)
        avg_squared_sum = squared_sum / (len(variations) - 1)
        return avg_squared_sum < self.threshold

    def get_transient_end_time(self):
        """Get the time when the transient phase ends."""
        if self.is_transient_over():
            return list(self.averages.keys())[0]  # Return the start time of the first interval
        return None