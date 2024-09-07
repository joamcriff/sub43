# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
from graphite.utils.graph_utils import timeout
import asyncio
import time

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')],
                 population_size=10, runs=20, max_trials=500):
        super().__init__(problem_types=problem_types)
        self.population_size = population_size  # Kích thước quần thể
        self.runs = runs  # Số lần chạy GA
        self.max_trials = max_trials  # Số lần thử tối đa cho ILK (LKH)
    
    async def solve(self, formatted_problem, future_id: int, beam_width: int = 2) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        
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
    
    # Các hàm hỗ trợ khác vẫn giữ nguyên như đã mô tả ở phần trước
    def generate_initial_tour(self, n):
        return list(range(n))

    def iterated_lin_kernighan(self, tour, distance_matrix):
        return tour

    def itp(self, tour, population, distance_matrix):
        return tour

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
    
