import numpy as np
import random
import time
import asyncio
from typing import List, Tuple, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')],
                 population_size=10, max_population_size=50, runs=10, total_time_limit=3600, seed=1):
        super().__init__(problem_types=problem_types)
        self.population_size = population_size
        self.max_population_size = max_population_size
        self.runs = runs
        self.total_time_limit = total_time_limit
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        self.distance_matrix = formatted_problem
        self.n = len(formatted_problem)
        self.start_time = time.time()
        self.last_time = self.start_time
        self.best_cost = float('inf')
        self.best_tour = None
        
        self.population = [(self.generate_initial_tour(), float('inf')) for _ in range(self.population_size)]

        for run in range(self.runs):
            if self.future_tracker.get(future_id):
                return None
            
            tour, cost = self.find_tour()  # Using Lin-Kernighan heuristic
            
            self.update_population(cost)
            self.update_best_tour(cost)
            self.perform_crossover()
            
            time_taken = time.time() - self.last_time
            self.update_statistics(cost, time_taken)
            self.last_time = time.time()
            
            if time.time() - self.start_time >= self.total_time_limit:
                print("*** Time limit exceeded ***")
                break

        self.print_statistics()
        return self.best_tour

    def find_tour(self) -> Tuple[List[int], float]:
        # Placeholder for Lin-Kernighan heuristic
        initial_tour = self.generate_initial_tour()
        cost = self.calculate_cost(initial_tour)
        return initial_tour, cost
    
    def generate_initial_tour(self) -> List[int]:
        nodes = list(range(self.n))
        random.shuffle(nodes)
        return nodes

    def calculate_cost(self, tour: List[int]) -> float:
        return sum(self.distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + self.distance_matrix[tour[-1]][tour[0]]

    def update_population(self, cost: float):
        new_tour = self.generate_initial_tour()
        new_cost = self.calculate_cost(new_tour)
        
        if len(self.population) < self.max_population_size:
            self.population.append((new_tour, new_cost))
        else:
            # Thay thế cá thể tồi nhất nếu chi phí mới tốt hơn
            worst_tour = max(self.population, key=lambda t: t[1])  # tìm cá thể có chi phí cao nhất
            if new_cost < worst_tour[1]:
                self.population.remove(worst_tour)
                self.population.append((new_tour, new_cost))
            
    def update_best_tour(self, cost: float):
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_tour = self.generate_initial_tour()

    def perform_crossover(self):
        if len(self.population) >= 2:
            parent1, parent2 = random.sample(self.population, 2)
            child_tour = self.crossover(parent1[0], parent2[0])
            child_cost = self.calculate_cost(child_tour)
            self.update_population(child_cost)

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        half = len(parent1) // 2
        child = parent1[:half] + [node for node in parent2 if node not in parent1[:half]]
        return child

    def update_statistics(self, cost: float, time_taken: float):
        print(f"Run {self.runs}: Cost = {cost}, Time = {time_taken:.2f} sec")

    def print_statistics(self):
        print(f"Best Cost: {self.best_cost}")
        print(f"Best Tour: {self.best_tour}")

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges  # Assume problem.edges gives the distance matrix

if __name__ == "__main__":
    # Run the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHGeneticSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
