import numpy as np
import random
import time
import asyncio
from typing import List, Tuple, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        best_route = None
        best_cost = float('inf')

        for _ in range(10):  # Try multiple random starts
            route = self.nearest_neighbour(distance_matrix, future_id)
            route = self.two_opt(route, distance_matrix)
            cost = self.calculate_cost(route, distance_matrix)
            if cost < best_cost:
                best_cost = cost
                best_route = route
        
        return best_route

    def nearest_neighbour(self, distance_matrix: List[List[Union[int, float]]], future_id: int) -> List[int]:
        n = len(distance_matrix[0])
        visited = [False] * n
        route = []
        total_distance = 0

        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return None
            nearest_distance = np.inf
            nearest_node = None
            for j in range(n):
                if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                    nearest_distance = distance_matrix[current_node][j]
                    nearest_node = j

            route.append(nearest_node)
            visited[nearest_node] = True
            total_distance += nearest_distance
            current_node = nearest_node
        
        total_distance += distance_matrix[current_node][route[0]]
        route.append(route[0])
        return route

    def two_opt(self, tour: List[int], distance_matrix: List[List[Union[int, float]]]) -> List[int]:
        best_tour = tour
        best_cost = self.calculate_cost(best_tour, distance_matrix)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(best_tour) - 1):
                for j in range(i + 1, len(best_tour)):
                    if j - i == 1:
                        continue
                    new_tour = best_tour[:i] + best_tour[i:j][::-1] + best_tour[j:]
                    new_cost = self.calculate_cost(new_tour, distance_matrix)
                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        
        return best_tour

    def calculate_cost(self, tour: List[int], distance_matrix: List[List[Union[int, float]]]) -> float:
        return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges  

if __name__ == "__main__":
    # Run the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time() - start_time}")
