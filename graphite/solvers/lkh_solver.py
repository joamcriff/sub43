import numpy as np
import random
import time
import asyncio
from typing import List, Tuple, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2)],
                 population_size=50, max_population_size=100, runs=20, total_time_limit=3600, seed=42):
        super().__init__(problem_types=problem_types)
        self.population_size = population_size
        self.max_population_size = max_population_size
        self.runs = runs
        self.total_time_limit = total_time_limit
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        self.distance_matrix = np.array(formatted_problem)
        self.n = len(formatted_problem)
        self.start_time = time.time()
        self.best_cost = float('inf')
        self.best_tour = None
        
        self.population = [(self.generate_initial_tour(), float('inf')) for _ in range(self.population_size)]

        for run in range(self.runs):
            if self.future_tracker.get(future_id):
                return None
            
            self.evaluate_population()
            self.perform_crossover_and_mutation()
            self.update_best_tour(*min(self.population, key=lambda x: x[1]))
            
            if time.time() - self.start_time >= self.total_time_limit:
                print("*** Time limit exceeded ***")
                break
        
        return self.best_tour

    def evaluate_population(self):
        for i in range(len(self.population)):
            tour, _ = self.find_tour(self.population[i][0])
            cost = self.calculate_cost(tour)
            self.population[i] = (tour, cost)

    def find_tour(self, tour: List[int]) -> Tuple[List[int], float]:
        tour = self.ensure_complete_tour(tour)
        return tour, self.calculate_cost(tour)
    
    def generate_initial_tour(self) -> List[int]:
        # Generate a random tour that starts and ends at node 0
        nodes = list(range(1, self.n))  # Exclude node 0 from randomization
        random.shuffle(nodes)
        return [0] + nodes + [0]  # Ensure tour starts and ends at node 0
    
    def ensure_complete_tour(self, tour: List[int]) -> List[int]:
        # Ensure tour starts and ends at node 0
        if tour[0] != 0:
            tour = [0] + [node for node in tour if node != 0]
        if tour[-1] != 0:
            tour.append(0)
        return tour

    def calculate_cost(self, tour: List[int]) -> float:
        return np.sum(self.distance_matrix[tour[:-1], tour[1:]])

    def update_best_tour(self, tour: List[int], cost: float):
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_tour = tour

    def perform_crossover_and_mutation(self):
        if len(self.population) >= 2:
            parent1, parent2 = random.sample(self.population, 2)
            child_tour = self.crossover(parent1[0], parent2[0])
            child_tour = self.ensure_complete_tour(child_tour)  # Ensure valid tour after crossover
            child_cost = self.calculate_cost(child_tour)
            self.update_population(child_tour, child_cost)
            
            # Apply mutation
            self.mutate(child_tour)
            child_tour = self.ensure_complete_tour(child_tour)  # Ensure valid tour after mutation

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        # Perform crossover between two parents to generate a child tour
        start, end = sorted(random.sample(range(1, self.n-1), 2))  # Exclude start and end nodes (0)
        child = [None] * self.n
        child[start:end + 1] = parent1[start:end + 1]
        current_position = end + 1
        for node in parent2:
            if node not in child:
                if current_position >= self.n:
                    current_position = 0
                child[current_position] = node
                current_position += 1
        return child

    def mutate(self, tour: List[int]):
        # Apply a simple mutation: swap two random nodes, excluding start/end nodes
        if self.n > 2:
            i, j = random.sample(range(1, self.n - 1), 2)  # Avoid swapping the start/end nodes
            tour[i], tour[j] = tour[j], tour[i]

    def update_population(self, tour: List[int], cost: float):
        # Add a new tour to the population or replace the worst tour if population is full
        new_tour = self.generate_initial_tour()
        new_tour = self.ensure_complete_tour(new_tour)
        new_cost = self.calculate_cost(new_tour)
        
        if len(self.population) < self.max_population_size:
            self.population.append((new_tour, new_cost))
        else:
            worst_tour = max(self.population, key=lambda t: t[1])
            if new_cost < worst_tour[1]:
                self.population.remove(worst_tour)
                self.population.append((new_tour, new_cost))

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges  

if __name__ == "__main__":
    # Run the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHGeneticSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
