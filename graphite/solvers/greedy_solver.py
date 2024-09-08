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
        distance_matrix = formatted_problem
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

            # Sử dụng hàng đợi ưu tiên để tìm hàng xóm gần nhất
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
        n = len(visited)
        priority_queue = []

        # Khởi tạo hàng đợi ưu tiên với tất cả các nút chưa được thăm
        for j in range(n):
            if not visited[j]:
                heapq.heappush(priority_queue, (distance_matrix[current_node][j], j))

        while priority_queue:
            nearest_distance, nearest_node = heapq.heappop(priority_queue)
            if not visited[nearest_node]:
                return nearest_node, nearest_distance

        return -1, np.inf

    def two_opt(self, route, distance_matrix):
        size = len(route)
        best_route = route[:]
        best_distance = self.calculate_total_distance(best_route, distance_matrix)
        improved = True

        while improved:
            improved = False
            for i in range(1, size - 2):
                for j in range(i + 2, size):
                    if j - i == 1: continue  # Skip adjacent edges

                    # Tính toán delta thay vì tính toàn bộ lại khoảng cách
                    old_distance = (
                        distance_matrix[best_route[i], best_route[i + 1]] +
                        distance_matrix[best_route[j - 1], best_route[j]]
                    )
                    new_distance = (
                        distance_matrix[best_route[i], best_route[j - 1]] +
                        distance_matrix[best_route[i + 1], best_route[j]]
                    )

                    delta = new_distance - old_distance
                    if delta < 0:  # Nếu cải thiện, đảo ngược đoạn giữa
                        best_route[i + 1:j] = best_route[i + 1:j][::-1]
                        best_distance += delta
                        improved = True

            # Thoát khi không còn cải thiện
        return best_route


    def calculate_total_distance(self, route, distance_matrix):
        return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + distance_matrix[route[-1]][route[0]]

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
