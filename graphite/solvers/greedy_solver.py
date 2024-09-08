from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import heapq
import numpy as np
import time
import asyncio
import random

class GeneticNearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')],
                 population_size=50, generations=100, mutation_rate=0.1):
        super().__init__(problem_types=problem_types)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)

        # Khởi tạo dân số ban đầu
        population = self.initialize_population(n)

        for generation in range(self.generations):
            if self.future_tracker.get(future_id):
                return None

            # Tính fitness
            fitness = [self.calculate_total_distance(individual, distance_matrix) for individual in population]
            
            # Lựa chọn
            selected_population = self.selection(population, fitness)
            
            # Giao phối
            offspring = self.crossover(selected_population)
            
            # Đột biến
            self.mutation(offspring)
            
            # Áp dụng 3-opt
            offspring = [self.three_opt(individual, distance_matrix) for individual in offspring]

            # Tạo dân số mới
            population = self.create_new_population(selected_population, offspring)

        # Chọn giải pháp tốt nhất từ dân số cuối cùng
        best_route = min(population, key=lambda ind: self.calculate_total_distance(ind, distance_matrix))
        return best_route

    def initialize_population(self, n):
        return [np.random.permutation(n).tolist() for _ in range(self.population_size)]

    def selection(self, population, fitness):
        sorted_population = [x for _, x in sorted(zip(fitness, population))]
        return sorted_population[:self.population_size // 2]

    def crossover(self, selected_population):
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = self.order_crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        return offspring

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child1 = [None] * size
        child2 = [None] * size

        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]

        fill_positions(child1, parent2, start, end)
        fill_positions(child2, parent1, start, end)

        return child1, child2

    def fill_positions(self, child, parent, start, end):
        pos = end + 1
        for gene in parent[end + 1:] + parent[:end + 1]:
            if gene not in child:
                if pos >= len(child):
                    pos = 0
                while child[pos] is not None:
                    pos += 1
                child[pos] = gene

    def mutation(self, offspring):
        for individual in offspring:
            if random.random() < self.mutation_rate:
                self.swap_mutation(individual)

    def swap_mutation(self, individual):
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]

    def create_new_population(self, selected_population, offspring):
        return selected_population + offspring[:self.population_size - len(selected_population)]

    def three_opt(self, route, distance_matrix):
        size = len(route)
        best_route = route[:]
        best_distance = self.calculate_total_distance(best_route, distance_matrix)
        improved = True

        while improved:
            improved = False
            for i in range(size - 3):
                for j in range(i + 2, size - 1):
                    for k in range(j + 2, size + (i > 0)):
                        if k == size: k = 0
                        new_route = self.apply_3opt(best_route, i, j, k)
                        new_distance = self.calculate_total_distance(new_route, distance_matrix)
                        if new_distance < best_distance:
                            best_route = new_route
                            best_distance = new_distance
                            improved = True

        return best_route

    def apply_3opt(self, route, i, j, k):
        if i > j or j > k or k >= len(route):
            return route

        new_route1 = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:k + 1][::-1] + route[k + 1:]
        new_route2 = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:k + 1] + route[k + 1:]
        new_route3 = route[:i + 1] + route[i + 1:j + 1] + route[j + 1:k + 1][::-1] + route[k + 1:]
        new_route4 = route[:i + 1] + route[i + 1:j + 1] + route[j + 1:k + 1] + route[k + 1:]
        return min([new_route1, new_route2, new_route3, new_route4], key=lambda r: self.calculate_total_distance(r, distance_matrix))

    def calculate_total_distance(self, route, distance_matrix):
        return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + distance_matrix[route[-1]][route[0]]

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == "__main__":
    # Runs the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = GeneticNearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time() - start_time}")