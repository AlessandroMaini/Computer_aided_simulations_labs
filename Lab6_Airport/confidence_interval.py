"""
Algorithm to compute confidence intervals for simulation output data.
"""
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIDENCE INTERVAL
# ----------------------------------------------------------------------------
class ConfidenceInterval:
    def __init__(self, min_samples_count: int, max_interval_width: float, confidence_level: float = 0.95):
        self.min_samples_count = min_samples_count
        self.max_interval_width = max_interval_width
        self.confidence_level = confidence_level
        self.data: list[float] = []
        self.average: float = 0.0
        self.std_dev: float = 0.0
        self.sample_count: int = 0

    def add_data_point(self, value: float):
        """Add a new data point to the dataset."""
        self.data.append(value)
        # Update statistics
        self.sample_count += 1
        self.average = sum(self.data) / self.sample_count
        if self.sample_count > 1:
            self.std_dev = (sum((x - self.average) ** 2 for x in self.data) / (self.sample_count - 1)) ** 0.5

    def get_sample_size(self) -> int:
        """Return the current sample size."""
        return self.sample_count

    def has_enough_data(self) -> bool:
        """Check if there are enough data points to compute the confidence interval."""
        return self.sample_count >= self.min_samples_count

    def compute_interval(self) -> tuple[bool, tuple[float, float]] | None:
        """Compute the confidence interval for the data using normal distribution."""
        if self.sample_count < 2:
            return None  # Not enough data to compute interval

        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        margin_of_error = z_score * (self.std_dev / (self.sample_count ** 0.5))

        lower_bound = self.average - margin_of_error
        upper_bound = self.average + margin_of_error

        final_interval = False
        if (upper_bound - lower_bound) <= self.max_interval_width * abs(self.average):
            final_interval = True

        return (final_interval, (lower_bound, upper_bound))