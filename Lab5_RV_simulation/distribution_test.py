from rv_generation import RVGenerator
import numpy as np
from scipy.stats import chi2
import math
import matplotlib.pyplot as plt

def test_hyperexponential_distribution(lambdas: list[float], probabilities: list[float], 
                                       num_intervals: int, num_samples: int, alpha: float) -> tuple[bool, np.ndarray, np.ndarray]:
    """Chi-square goodness-of-fit test for hyperexponential distribution."""
    # Define n intervals in the r.v. support
    intervals = np.linspace(0, 10, num_intervals + 1)

    # Generate samples and count frequencies in each interval
    samples = RVGenerator.hyperexponential_sample(lambdas, probabilities, num_samples)
    observed_freq, _ = np.histogram(samples, bins=intervals)

    # Compute expected frequencies
    expected_freq = []
    for i in range(num_intervals):
        a, b = intervals[i], intervals[i + 1]
        prob = 0
        for lam, p in zip(lambdas, probabilities):
            prob += p * (np.exp(-lam * a) - np.exp(-lam * b))
        expected_freq.append(prob * num_samples)
    expected_freq = np.array(expected_freq)

    # Compute chi-square statistic
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    # Compare with critical value from chi-square distribution
    critical_value = chi2.ppf(1 - alpha, df=num_intervals - 1)
    print(f"Chi-square statistic: {chi_square_stat}, Critical value: {critical_value}")
    if chi_square_stat > critical_value:
        return (False, observed_freq, expected_freq)
    else:
        return (True, observed_freq, expected_freq)
    
def test_erlang_k_distribution(_lambda: float, k: int, 
                               num_intervals: int, num_samples: int, alpha: float) -> tuple[bool, np.ndarray, np.ndarray]:
    """Chi-square goodness-of-fit test for erlang-k distribution."""
    # Define n intervals in the r.v. support
    intervals = np.linspace(0, 10, num_intervals + 1)

    # Generate samples and count frequencies in each interval
    samples = RVGenerator.erlang_k_sample(_lambda, k, num_samples)
    observed_freq, _ = np.histogram(samples, bins=intervals)

    # Compute expected frequencies
    expected_freq = []
    for i in range(num_intervals):
        a, b = intervals[i], intervals[i + 1]
        # Compute probability between [a, b] using erlang-k CDF
        prob_a = 1 - sum((_lambda * a)**i * np.exp(-_lambda * a) / math.factorial(i) for i in range(k))
        prob_b = 1 - sum((_lambda * b)**i * np.exp(-_lambda * b) / math.factorial(i) for i in range(k))
        prob = prob_b - prob_a
        expected_freq.append(prob * num_samples)
    expected_freq = np.array(expected_freq)

    # Compute chi-square statistic
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    # Compare with critical value from chi-square distribution
    critical_value = chi2.ppf(1 - alpha, df=num_intervals - 1)
    print(f"Chi-square statistic: {chi_square_stat}, Critical value: {critical_value}")
    if chi_square_stat > critical_value:
        return (False, observed_freq, expected_freq)
    else:
        return (True, observed_freq, expected_freq)

def test_pareto_distribution(shape: float, num_intervals: int,
                             num_samples: int, alpha: float, scale: float = 1.0) -> tuple[bool, np.ndarray, np.ndarray]:
    """Chi-square goodness-of-fit test for pareto distribution."""
    
    # Define n intervals in the r.v. support
    intervals = np.linspace(scale, 10, num_intervals + 1)

    # Generate samples and count frequencies in each interval
    samples = RVGenerator.pareto_sample(shape, num_samples, scale=scale)
    observed_freq, _ = np.histogram(samples, bins=intervals)

    # Compute expected frequencies
    expected_freq = []
    for i in range(num_intervals):
        a, b = intervals[i], intervals[i + 1]
        # Compute probability between [a, b] using pareto CDF
        prob = (scale / a) ** shape - (scale / b) ** shape
        expected_freq.append(prob * num_samples)
    expected_freq = np.array(expected_freq)

    # Compute chi-square statistic
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    # Compare with critical value from chi-square distribution
    critical_value = chi2.ppf(1 - alpha, df=num_intervals - 1)
    print(f"Chi-square statistic: {chi_square_stat}, Critical value: {critical_value}")
    if chi_square_stat > critical_value:
        return (False, observed_freq, expected_freq)
    else:
        return (True, observed_freq, expected_freq)

def compare_distributions_plot(observed_freq: np.ndarray, expected_freq: np.ndarray, title: str) -> None:
    """Plot observed vs expected frequencies for visual comparison."""
    indices = np.arange(len(observed_freq))
    width = 0.35

    plt.bar(indices, observed_freq, width=width, label='Observed', alpha=0.7)
    plt.bar(indices + width, expected_freq, width=width, label='Expected', alpha=0.7)

    plt.xlabel('Intervals')
    plt.ylabel('Frequencies')
    plt.title(title)
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    np.random.seed(0)
    # Test hyperexponential distribution
    res = test_hyperexponential_distribution(lambdas=[6.0, 4.0, 2.0], probabilities=[0.6, 0.2, 0.2], num_intervals=50, num_samples=1000, alpha=0.05)
    if res[0]:
        print("Hyperexponential test PASSED")
    else:
        print("Hyperexponential test FAILED")

    compare_distributions_plot(res[1], res[2], title="Hyperexponential Distribution: Observed vs Expected Frequencies")

    # Test Erlang-K distribution
    res = test_erlang_k_distribution(_lambda=1.0, k=5, num_intervals=50, num_samples=1000, alpha=0.05)
    if res[0]:
        print("Erlang-K test PASSED")
    else:
        print("Erlang-K test FAILED")

    compare_distributions_plot(res[1], res[2], title="Erlang-K Distribution: Observed vs Expected Frequencies")

    # Test Pareto distribution
    res = test_pareto_distribution(shape=5.0, num_intervals=50, num_samples=1000, alpha=0.05, scale=0.2)
    if res[0]:
        print("Pareto test PASSED")
    else:
        print("Pareto test FAILED")

    compare_distributions_plot(res[1], res[2], title="Pareto Distribution: Observed vs Expected Frequencies")