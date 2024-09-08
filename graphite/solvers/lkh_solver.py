import numpy as np
import random
import time
import asyncio
from typing import List, Tuple, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem

def path_cost_from_distance_matrix(distances, path):
    return sum(distances[path[i], path[i+1]] for i in range(len(path)-1)) + distances[path[-1], path[0]]

def two_opt_change(route, i, j):
    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return new_route

def two_opt(path, distances):
    best_distance = path_cost_from_distance_matrix(distances, path)
    present_route = path.copy()

    for i in range(len(path) - 2):
        for j in range(i + 1, len(path) - 1):
            new_route = two_opt_change(present_route, i, j)
            new_distance = path_cost_from_distance_matrix(distances, new_route)

            if new_distance < best_distance:
                present_route = new_route
                best_distance = new_distance

    return best_distance, present_route

def simulated_annealing(tour, distances, initial_temp=1000, cooling_rate=0.003):
    current_temp = initial_temp
    current_solution = tour
    best_solution = current_solution
    best_cost = path_cost_from_distance_matrix(distances, current_solution)

    while current_temp > 1:
        i, j = random.sample(range(1, len(tour)-1), 2)
        new_solution = current_solution[:]
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        current_cost = path_cost_from_distance_matrix(distances, current_solution)
        new_cost = path_cost_from_distance_matrix(distances, new_solution)

        if new_cost < current_cost or random.uniform(0, 1) < np.exp((current_cost - new_cost) / current_temp):
            current_solution = new_solution

        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost

        current_temp *= 1 - cooling_rate

    return best_solution, best_cost

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2)],
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
        nodes = list(range(1, self.n))  # Exclude node 0 from randomization
        random.shuffle(nodes)
        return [0] + nodes + [0]  # Ensure tour starts and ends at node 0
    
    def ensure_complete_tour(self, tour: List[int]) -> List[int]:
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
            child_tour = self.ensure_complete_tour(child_tour)
            child_cost = self.calculate_cost(child_tour)

            # Áp dụng 2-opt để cải thiện lộ trình con
            improved_cost, improved_tour = two_opt(child_tour, self.distance_matrix)

            # Áp dụng simulated annealing sau khi crossover và mutation
            improved_tour, improved_cost = simulated_annealing(improved_tour, self.distance_matrix)

            self.update_population(improved_tour, improved_cost)

            # Apply mutation
            self.mutate(improved_tour)
            improved_tour, improved_cost = simulated_annealing(improved_tour, self.distance_matrix)

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        start, end = sorted(random.sample(range(1, self.n-1), 2))
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
        if self.n > 2:
            i, j = random.sample(range(1, self.n - 1), 2)
            tour[i], tour[j] = tour[j], tour[i]

    def update_population(self, tour: List[int], cost: float):
        if len(self.population) < self.max_population_size:
            self.population.append((tour, cost))
        else:
            worst_tour = max(self.population, key=lambda t: t[1])
            if cost < worst_tour[1]:
                self.population.remove(worst_tour)
                self.population.append((tour, cost))

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges

if __name__ == "__main__":
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHGeneticSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
