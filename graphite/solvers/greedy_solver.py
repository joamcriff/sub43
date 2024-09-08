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
        # Chuyển đổi distance_matrix thành mảng NumPy
        distance_matrix = np.array(formatted_problem)
        n = distance_matrix.shape[0]
        
        visited = np.zeros(n, dtype=bool)
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

        # Quay trở lại nút bắt đầu
        total_distance += distance_matrix[current_node, route[0]]
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
        distance_matrix = np.array(distance_matrix)  # Chuyển đổi thành mảng NumPy
        best_distance = self.calculate_total_distance(best_route, distance_matrix)
        improved = True

        while improved:
            improved = False
            for i in range(1, size - 2):
                for j in range(i + 2, size):
                    if j - i == 1:
                        continue  # Bỏ qua các cạnh kề

                    # Chuyển đổi chỉ số thành số nguyên
                    i1, i2 = int(best_route[i]), int(best_route[i + 1])
                    j1, j2 = int(best_route[j - 1]), int(best_route[j])
                    
                    # Tính toán delta khoảng cách của việc hoán đổi
                    old_distance = (distance_matrix[i1, i2] +
                                    distance_matrix[j1, j2])
                    new_distance = (distance_matrix[i1, j1] +
                                    distance_matrix[i2, j2])
                    
                    delta = new_distance - old_distance
                    if delta < 0:
                        # Thực hiện hoán đổi 2-opt
                        best_route[i + 1:j] = reversed(best_route[i + 1:j])
                        best_distance += delta
                        improved = True

        return best_route

    def calculate_total_distance(self, route, distance_matrix):
        distance_matrix = np.array(distance_matrix)  # Chuyển đổi thành mảng NumPy
        route = np.array(route)  # Đảm bảo route là mảng NumPy
        indices = np.arange(len(route) - 1)
        total_distance = np.sum(distance_matrix[route[indices], route[indices + 1]])
        total_distance += distance_matrix[route[-1], route[0]]
        return total_distance

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
