from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import time
import asyncio
import random
from typing import List, Union, Tuple
from numba import njit  # Import numba for JIT compilation

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')], max_2opt_iterations: int = 100):
        super().__init__(problem_types=problem_types)
        self.max_2opt_iterations = max_2opt_iterations

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = np.array(formatted_problem)
        n = len(distance_matrix[0])
        num_starts = max(n // 3, 1)  # Đảm bảo ít nhất 1 điểm bắt đầu

        best_route = None
        best_total_distance = float('inf')

        # Tìm kiếm nhiều hướng đồng thời
        start_nodes = random.sample(range(n), min(num_starts, n))  # Chọn num_starts điểm bắt đầu ngẫu nhiên
        routes = await asyncio.gather(
            *[self.find_route_from_start(distance_matrix, start_node, n, future_id) for start_node in start_nodes]
        )

        # Chọn đường đi tốt nhất
        for route, total_distance in routes:
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_route = route

        # Áp dụng 2-opt để tối ưu hóa đường đi
        optimized_route = self.two_opt(best_route, distance_matrix)

        return optimized_route

    async def find_route_from_start(self, distance_matrix: np.ndarray, start_node: int, n: int, future_id: int) -> Tuple[List[int], float]:
        visited = np.zeros(n, dtype=bool)
        route = [start_node]
        total_distance = 0

        current_node = start_node
        visited[current_node] = True

        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return route, float('inf')

            # Tìm điểm gần nhất chưa thăm bằng numpy
            remaining = np.where(~visited)[0]
            nearest_node = remaining[np.argmin(distance_matrix[current_node, remaining])]
            total_distance += distance_matrix[current_node][nearest_node]

            # Di chuyển đến điểm gần nhất
            route.append(nearest_node)
            visited[nearest_node] = True
            current_node = nearest_node

        # Trở về điểm xuất phát
        total_distance += distance_matrix[current_node][start_node]
        route.append(start_node)

        return route, total_distance

    def two_opt(self, route: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Thực hiện thuật toán 2-opt để tối ưu hóa tuyến đường với giới hạn lần lặp."""
        def calculate_total_distance(route):
            return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

        n = len(route)
        best_distance = calculate_total_distance(route)
        iteration = 0

        while iteration < self.max_2opt_iterations:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    new_distance = calculate_total_distance(new_route)

                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True

            if not improved:
                break
            iteration += 1

        return route

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges

if __name__ == "__main__":
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NearestNeighbourSolver(problem_types=[test_problem.problem_type], max_2opt_iterations=50)
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time() - start_time}")
