import numpy as np

class RVGenerator:
    """A class for generating random variables based on specified distributions."""
    
    @staticmethod
    def hyperexponential_sample(lambdas: list[float], probabilities: list[float], num_samples: int):
        """Generate samples from a hyperexponential distribution using composition method.

        Args:
            lambdas (list[float]): List of rate parameters for the exponential distributions.
            probabilities (list[float]): List of probabilities for each exponential distribution.
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Generated samples from the hyperexponential distribution.
        """
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.")

        if len(lambdas) != len(probabilities):
            raise ValueError("Length of lambdas and probabilities must be the same.")

        if any(l <= 0 for l in lambdas):
            raise ValueError("All lambda values must be positive.")

        if any(p < 0 for p in probabilities):
            raise ValueError("All probability values must be non-negative.")

        # Generate samples
        samples = []
        for _ in range(num_samples):
            # Select an exponential distribution based on the probabilities
            chosen_dist = np.random.choice(len(lambdas), p=probabilities)
            sample = np.random.exponential(1/lambdas[chosen_dist])
            samples.append(sample)

        return np.array(samples)

    @staticmethod
    def erlang_k_sample(_lambda: float, k: int, num_samples: int):
        """Generate samples from an Erlang-k distribution using convolution method.

        Args:
            _lambda (float): Rate parameter of the Erlang distribution.
            k (int): Number of phases.
            num_samples (int): Number of samples to generate.
        
        Returns:
            np.ndarray: Generated samples from the Erlang-k distribution.
        """
        if _lambda <= 0:
            raise ValueError("Rate must be a positive value.")
        
        if k <= 0 or not isinstance(k, int):
            raise ValueError("Shape parameter k must be a positive integer.")
        
        if num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        
        # Generate samples
        samples = []
        for _ in range(num_samples):
            # Sum of k exponential random variables
            exp_samples = np.random.exponential(1/_lambda, k)
            samples.append(np.sum(exp_samples))

        return np.array(samples)

    @staticmethod
    def pareto_sample(alpha: float, num_samples: int):
        """Generate samples from a Pareto distribution using inverse transform method (k=1).

        Args:
            alpha (float): Shape parameter of the Pareto distribution.
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Generated samples from the Pareto distribution.
        """
        if alpha <= 0:
            raise ValueError("Shape parameter alpha must be a positive value.")
        
        if num_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        
        # Generate samples
        samples = []
        for _ in range(num_samples):
            u = np.random.uniform(0, 1)
            sample = 1 / (u ** (1 / alpha))
            samples.append(sample)

        return np.array(samples)