import numpy as np
import matplotlib.pyplot as plt
from rv_generation import RVGenerator

def plot_autocorrelation(samples: list[np.ndarray], num_lags: int = 100, rv: str = "Uniform"):
    """Auto-correlation test for independence of random variables.
    Compute the auto-correlation function for each lag and plot the results"""

    var = np.sum((samples - np.mean(samples))**2)
    acf = []
    lags = np.linspace(0, len(samples), num_lags, dtype=int)
    for lag in lags:
        cov = np.sum((samples[:len(samples)-lag] - np.mean(samples)) * (samples[lag:] - np.mean(samples)))
        acf.append(cov / var)

    # Plot the auto-correlation function
    plt.title("Auto-correlation Function for " + rv + " Random Variables")
    plt.xlabel("Lag")
    plt.ylabel("Auto-correlation")
    plt.stem(lags, acf)
    plt.legend()
    plt.show()

def correlation_scatter_plot(samples: list[np.ndarray], lag: int = 1, rv: str = "Uniform"):
    """Scatter plot for independence of random variables.
    Plot samples at time t vs samples at time t + lag"""

    plt.title(f"Scatter Plot for {rv} Random Variables (Lag={lag})")
    plt.xlabel("X(t)")
    plt.ylabel(f"X(t + {lag})")
    plt.scatter(samples[:-lag], samples[lag:], alpha=0.5)
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    samples = np.random.rand(1000)
    plot_autocorrelation(samples, num_lags=100)
    correlation_scatter_plot(samples, lag=10)

    # Check independence for generated random variables
    hyperexp_samples = RVGenerator.hyperexponential_sample(lambdas=[1.0, 2.0, 5.0], probabilities=[0.5, 0.3, 0.2], num_samples=1000)
    plot_autocorrelation(hyperexp_samples, num_lags=100, rv="Hyperexponential")
    correlation_scatter_plot(hyperexp_samples, lag=10, rv="Hyperexponential")

    erlangk_samples = RVGenerator.erlang_k_sample(_lambda=1.0, k=5, num_samples=1000)
    plot_autocorrelation(erlangk_samples, num_lags=100, rv="Erlang-K")
    correlation_scatter_plot(erlangk_samples, lag=10, rv="Erlang-K")

    pareto_samples = RVGenerator.pareto_sample(alpha=3.0, num_samples=1000)
    plot_autocorrelation(pareto_samples, num_lags=100, rv="Pareto")
    correlation_scatter_plot(pareto_samples, lag=10, rv="Pareto")