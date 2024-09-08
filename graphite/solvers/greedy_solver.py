from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import heapq
import numpy as np
import time
import asyncio

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        visited = [False] * n
        route = []
        total_distance = 0

        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return None

            nearest_node, nearest_distance = self.find_nearest_neighbor(current_node, visited, distance_matrix)

            if nearest_node != -1:
                route.append(nearest_node)
                visited[nearest_node] = True
                total_distance += nearest_distance
                current_node = nearest_node

        total_distance += distance_matrix[current_node][route[0]]
        route.append(route[0])

        # Áp dụng thuật toán 3-opt để cải thiện giải pháp
        route = self.three_opt(route, distance_matrix)
        return route

    def find_nearest_neighbor(self, current_node, visited, distance_matrix):
        n = len(visited)
        priority_queue = []

        for j in range(n):
            if not visited[j]:
                heapq.heappush(priority_queue, (distance_matrix[current_node][j], j))

        while priority_queue:
            nearest_distance, nearest_node = heapq.heappop(priority_queue)
            if not visited[nearest_node]:
                return nearest_node, nearest_distance

        return -1, np.inf

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
        # Tạo các biến thể của lộ trình với các cách kết hợp ba đoạn khác nhau
        new_route = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:k + 1][::-1] + route[k + 1:]
        return new_route

    def calculate_total_distance(self, route, distance_matrix):
        return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + distance_matrix[route[-1]][route[0]]

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == "__main__":
    # Runs the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time() - start_time}")
