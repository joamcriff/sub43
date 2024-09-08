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
import heapq
import numpy as np
import time
import asyncio

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = np.array(formatted_problem)
        n = len(distance_matrix)
        visited = np.zeros(n, dtype=bool)
        route = []
        total_distance = 0

        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
            if future_id:  # Dummy condition for future tracker
                return None

            nearest_node, nearest_distance = self.find_nearest_neighbor(current_node, visited, distance_matrix)

            if nearest_node != -1:
                route.append(nearest_node)
                visited[nearest_node] = True
                total_distance += nearest_distance
                current_node = nearest_node

        # Quay trở lại nút bắt đầu
        total_distance += distance_matrix[current_node][route[0]]
        route.append(route[0])

        # Áp dụng thuật toán 2-opt để cải thiện giải pháp
        route = self.two_opt(route, distance_matrix)
        return route

    def find_nearest_neighbor(self, current_node, visited, distance_matrix):
            # Sử dụng vector hóa NumPy để tìm hàng xóm gần nhất
            unvisited_distances = np.where(~visited, distance_matrix[current_node], np.inf)
            nearest_node = np.argmin(unvisited_distances)
            nearest_distance = unvisited_distances[nearest_node]
            return nearest_node, nearest_distance

    def two_opt(self, route, distance_matrix):
        size = len(route)
        best_route = route[:]
        best_distance = self.calculate_total_distance(best_route, distance_matrix)
        improved = True

        while improved:
            improved = False
            for i in range(1, size - 2):
                for j in range(i + 2, size):
                    if j - i == 1: continue
                    new_route = best_route[:i + 1] + best_route[i + 1:j][::-1] + best_route[j:]
                    new_distance = self.calculate_total_distance_partial(best_route, new_route, i, j, best_distance, distance_matrix)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True

        return best_route

    def calculate_total_distance_partial(self, old_route, new_route, i, j, old_distance, distance_matrix):
        # Giảm số lần tính toán lại toàn bộ khoảng cách bằng cách cập nhật khoảng cách thay đổi
        old_segment_distance = distance_matrix[old_route[i], old_route[i + 1]] + distance_matrix[old_route[j - 1], old_route[j]]
        new_segment_distance = distance_matrix[new_route[i], new_route[i + 1]] + distance_matrix[new_route[j - 1], new_route[j]]
        return old_distance - old_segment_distance + new_segment_distance

    def calculate_total_distance(self, route, distance_matrix):
        # Sử dụng NumPy để tính toán nhanh tổng khoảng cách
        indices = np.arange(len(route) - 1)
        return np.sum(distance_matrix[route[indices], route[indices + 1]])

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == "__main__":
    # runs the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time() - start_time}")
