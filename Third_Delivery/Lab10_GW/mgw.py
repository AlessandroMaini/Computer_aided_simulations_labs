import numpy as np
from enum import Enum
import tqdm


# -----------------------------------------------
# Population
# -----------------------------------------------
class IndividualClass(Enum):
    CLASS_A = 0
    CLASS_B = 1

class Individual:
    def __init__(self, ind_class: IndividualClass):
        self.ind_class = ind_class

    def reproduce(self, mean_matrix: np.ndarray) -> list:
        offspring = []
        if self.ind_class == IndividualClass.CLASS_A:
            num_offspring_A = np.random.poisson(mean_matrix[0][0])
            num_offspring_B = np.random.poisson(mean_matrix[0][1])
        else:
            num_offspring_A = np.random.poisson(mean_matrix[1][0])
            num_offspring_B = np.random.poisson(mean_matrix[1][1])
        
        offspring.extend([Individual(IndividualClass.CLASS_A) for _ in range(num_offspring_A)])
        offspring.extend([Individual(IndividualClass.CLASS_B) for _ in range(num_offspring_B)])
        return offspring
    

# -----------------------------------------------
# Galton-Watson Multi-type Tree
# -----------------------------------------------
class MultiTypeGW:
    def __init__(self, ancestor_class: IndividualClass, mean_matrix: np.ndarray):
        self.ancestor = Individual(ancestor_class)
        self.mean_matrix = mean_matrix
        self.current_generation = [self.ancestor]
        self.generations = []

    def step(self):
        next_generation = []
        for individual in self.current_generation:
            offspring = individual.reproduce(self.mean_matrix)
            next_generation.extend(offspring)
        self.generations.append(self.current_generation)
        self.current_generation = next_generation


# -----------------------------------------------
# Simulation Engine
# -----------------------------------------------
class SimulationEngine:
    def __init__(self, mean_matrix: np.ndarray, initial_class: IndividualClass, max_generations: int):
        self.mean_matrix = mean_matrix
        self.initial_class = initial_class
        self.max_generations = max_generations
        self.simulations_run = 0
        self.extinctions = 0
        self.last_generation = 0

    def run_simulation(self) -> bool:
        gw_process = MultiTypeGW(self.initial_class, self.mean_matrix)
        self.simulations_run += 1
        for generation in range(self.max_generations):
            if not gw_process.current_generation:
                self.extinctions += 1
                self.last_generation = generation
                return True  # Extinction
            gw_process.step()
        self.last_generation = self.max_generations
        return False  # Survived
    
    def estimate_extinction_probability(self) -> float:
        return self.extinctions / self.simulations_run if self.simulations_run > 0 else 0.0
    

# -----------------------------------------------
# Main Execution
# -----------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    alpha_values = [0.9, 0.95, 1.0, 1.05, 1.1]
    mean_matrix_coefficients = [[6/7, 2/7], [2/7, 3/7]]
    MAX_GENERATIONS = 100
    NUM_SIMULATIONS = 100
    INITIAL_CLASS = IndividualClass.CLASS_A # or IndividualClass.CLASS_B

    for alpha in alpha_values:
        mean_matrix = np.array(mean_matrix_coefficients) * alpha
        engine = SimulationEngine(mean_matrix, INITIAL_CLASS, MAX_GENERATIONS)
        avg_last_generation = 0
        
        print(f"Running simulations for alpha = {alpha}...")
        for _ in tqdm.tqdm(range(NUM_SIMULATIONS), desc=f"Simulations for alpha={alpha}", unit="sim"):
            engine.run_simulation()
            avg_last_generation += engine.last_generation
        
        avg_last_generation = avg_last_generation / NUM_SIMULATIONS
        extinction_prob = engine.estimate_extinction_probability()
        print(f"Alpha: {alpha}, Estimated Extinction Probability: {extinction_prob}, Average Last Generation: {avg_last_generation}")