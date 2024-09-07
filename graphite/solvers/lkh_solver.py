from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
from graphite.utils.graph_utils import timeout
import asyncio
import time
import numpy as np
import random

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')],
                 population_size=10, runs=20, max_trials=500):
        super().__init__(problem_types=problem_types)
        self.population_size = population_size  # Kích thước quần thể
        self.runs = runs  # Số lần chạy GA
        self.max_trials = max_trials  # Số lần thử tối đa cho ILK (LKH)
    
    async def solve(self, formatted_problem, future_id: int, beam_width: int = 2) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        
        # Initialize population
        population = []
        initial_tour = self.generate_initial_tour(n)
        
        # Run Genetic Algorithm (GA)
        for run in range(self.runs):
            if self.future_tracker.get(future_id):
                return None  # Return early if the task is canceled
            
            # Apply LKH to improve the tour
            improved_tour = self.iterated_lin_kernighan(initial_tour, distance_matrix)
            improved_tour = self.itp(improved_tour, population, distance_matrix)
            
            # Add the new tour if no existing tour in the population has the same cost
            if not self.has_same_cost(improved_tour, population, distance_matrix):
                if len(population) < self.population_size:
                    population.append(improved_tour)
                else:
                    # Replace the worst tour if the new one is better
                    worst_tour = self.find_worst_tour(population, distance_matrix)
                    if self.cost(improved_tour, distance_matrix) < self.cost(worst_tour, distance_matrix):
                        self.replace_tour(improved_tour, worst_tour, population)
            
            # Select two parents and perform crossover to generate a child tour
            parent1, parent2 = self.select_parents(population, distance_matrix)
            child_tour = self.crossover(parent1, parent2)
            initial_tour = child_tour  # Set the child as the initial tour for the next run
        
        # Return the best tour from the population
        best_tour = self.find_best_tour(population, distance_matrix)
        return best_tour
    
    def generate_initial_tour(self, n):
        return list(range(n))
    
    def iterated_lin_kernighan(self, tour, distance_matrix):
        def cost(tour, distance_matrix):
            return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + distance_matrix[tour[-1]][tour[0]]
        
        def swap_two_edges(tour, i, j):
            new_tour = tour[:i+1] + list(reversed(tour[i+1:j+1])) + tour[j+1:]
            return new_tour
        
        best_tour = tour[:]
        best_cost = cost(best_tour, distance_matrix)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(tour)):
                for j in range(i + 2, len(tour)):
                    if i == 0 and j == len(tour) - 1:
                        continue
                    new_tour = swap_two_edges(best_tour, i, j)
                    new_cost = cost(new_tour, distance_matrix)
                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
        
        return best_tour
    
    def itp(self, tour, population, distance_matrix):
        improved_tour = self.iterated_lin_kernighan(tour, distance_matrix)
        return improved_tour
    
    def has_same_cost(self, tour, population, distance_matrix):
        for p in population:
            if self.cost(tour, distance_matrix) == self.cost(p, distance_matrix):
                return True
        return False

    def cost(self, tour, distance_matrix):
        return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + distance_matrix[tour[-1]][tour[0]]

    def find_worst_tour(self, population, distance_matrix):
        return max(population, key=lambda t: self.cost(t, distance_matrix))

    def replace_tour(self, new_tour, worst_tour, population):
        population.remove(worst_tour)
        population.append(new_tour)

    def select_parents(self, population, distance_matrix):
        sorted_population = sorted(population, key=lambda t: self.cost(t, distance_matrix))
        return sorted_population[0], sorted_population[1]

    def crossover(self, parent1, parent2):
        half = len(parent1) // 2
        child = parent1[:half] + [node for node in parent2 if node not in parent1[:half]]
        return child

    def find_best_tour(self, population, distance_matrix):
        return min(population, key=lambda t: self.cost(t, distance_matrix))
    
    def problem_transformations(self, problem: GraphProblem):
        return problem.edges  # Assume problem.edges gives the distance matrix
    
if __name__=='__main__':
    # runs the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHGeneticSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
