import numpy as np
from tqdm import tqdm

class GravitationalOptimizer:
    def __init__(self, objective_func, num_agents, num_features, max_iter,
                 G_initial=100, alpha=20, epsilon=1e-12, verbose=False):
        """
        :param objective_func: Function to minimize (e.g., 1 - Dice Coefficient)
        :param num_agents: Number of candidate solutions
        :param num_features: Dimension of solution vector
        :param max_iter: Number of iterations
        :param G_initial: Initial gravitational constant
        :param alpha: Decay rate for gravitational constant
        :param epsilon: Small constant to avoid division by zero
        :param verbose: Print optimization progress
        """
        self.obj_func = objective_func
        self.num_agents = num_agents
        self.num_features = num_features
        self.max_iter = max_iter
        self.G_initial = G_initial
        self.alpha = alpha
        self.epsilon = epsilon
        self.verbose = verbose

        # Initialize population (agents) randomly between 0 and 1
        self.positions = np.random.rand(num_agents, num_features)
        self.velocities = np.zeros((num_agents, num_features))

    def _update_gravitational_constant(self, t):
        return self.G_initial * np.exp(-self.alpha * t / self.max_iter)

    def _calculate_fitness(self):
        return np.array([self.obj_func(agent) for agent in self.positions])

    def _calculate_masses(self, fitness):
        best = np.min(fitness)
        worst = np.max(fitness)
        masses = (fitness - worst) / (best - worst + self.epsilon)
        masses = masses / (np.sum(masses) + self.epsilon)
        return masses

    def _calculate_forces(self, masses, G):
        forces = np.zeros_like(self.positions)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    R = np.linalg.norm(self.positions[j] - self.positions[i]) + self.epsilon
                    force = G * (masses[i] * masses[j]) * (self.positions[j] - self.positions[i]) / R
                    forces[i] += force
        return forces

    def _update_positions(self, forces):
        self.velocities = np.random.rand(*self.velocities.shape) * self.velocities + forces
        self.positions += self.velocities

        # Clamp positions between 0 and 1
        self.positions = np.clip(self.positions, 0, 1)

    def run(self):
        best_position = None
        best_score = float('inf')

        for t in tqdm(range(self.max_iter), desc="Optimizing (GOA)"):
            fitness = self._calculate_fitness()
            masses = self._calculate_masses(fitness)
            G = self._update_gravitational_constant(t)
            forces = self._calculate_forces(masses, G)
            self._update_positions(forces)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_score:
                best_score = fitness[current_best_idx]
                best_position = self.positions[current_best_idx].copy()

            if self.verbose:
                print(f"[Iteration {t+1}/{self.max_iter}] Best Score: {best_score:.6f}")

        return best_position, best_score
