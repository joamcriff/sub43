from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import time
import asyncio
import random
from typing import List, Union, Tuple

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        num_starts = max(n//3, 1)  # Đảm bảo ít nhất 1 điểm bắt đầu

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

        # Áp dụng 3-opt để tối ưu hóa đường đi
        optimized_route = self.three_opt(best_route, distance_matrix)

        return optimized_route

    async def find_route_from_start(self, distance_matrix: List[List[Union[int, float]]], start_node: int, n: int, future_id: int) -> Tuple[List[int], float]:
        visited = [False] * n
        route = []
        total_distance = 0

        current_node = start_node
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return route, float('inf')

            # Tìm điểm gần nhất chưa thăm
            nearest_distance = np.inf
            nearest_node = None
            for j in range(n):
                if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                    nearest_distance = distance_matrix[current_node][j]
                    nearest_node = j

            if nearest_node is None:
                break

            # Di chuyển đến điểm gần nhất chưa thăm
            route.append(nearest_node)
            visited[nearest_node] = True
            total_distance += nearest_distance
            current_node = nearest_node

        # Trở về điểm xuất phát
        total_distance += distance_matrix[current_node][start_node]
        route.append(start_node)

        return route, total_distance

    def three_opt(self, route: List[int], distance_matrix: List[List[Union[int, float]]]) -> List[int]:
        """Thực hiện thuật toán 3-opt để tối ưu hóa tuyến đường."""
        def calculate_total_distance(route):
            return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

        def reverse_segment_if_better(route, i, j, k):
            """Thử tất cả các hoán đổi 3 cạnh có thể và chọn cách tốt nhất."""
            A, B, C = route[i - 1], route[i], route[j - 1], route[j], route[k - 1], route[k % len(route)]
            d0 = distance_matrix[A][B] + distance_matrix[C][route[k - 1]] + distance_matrix[route[k - 1]][route[k % len(route)]]
            d1 = distance_matrix[A][route[k - 1]] + distance_matrix[B][C] + distance_matrix[route[k - 1]][route[k % len(route)]]
            d2 = distance_matrix[A][C] + distance_matrix[route[k - 1]][B] + distance_matrix[route[k - 1]][route[k % len(route)]]
            d3 = distance_matrix[A][C] + distance_matrix[B][route[k - 1]] + distance_matrix[route[k - 1]][route[k % len(route)]]
            d4 = distance_matrix[route[i - 1]][route[k - 1]] + distance_matrix[B][C] + distance_matrix[route[k - 1]][route[k % len(route)]]
            d_min = min(d0, d1, d2, d3, d4)

            if d_min == d0:
                return route  # Không cần thay đổi
            elif d_min == d1:
                new_route = route[:i] + route[i:k][::-1] + route[k:]
                return new_route
            elif d_min == d2:
                new_route = route[:i] + route[i:k] + route[j:i][::-1] + route[k:]
                return new_route
            elif d_min == d3:
                new_route = route[:i] + route[i:k] + route[j + 1:i]
                return new_route
            else:
                new_route = route[:i] + route[i:k] + route[j:]
                return new_route

        n = len(route)
        improved = True

        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 2, n - 1):
                    for k in range(j + 2, n + 1):
                        new_route = reverse_segment_if_better(route, i, j, k)
                        if new_route != route:
                            route = new_route
                            improved = True

        return route

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges


if __name__ == "__main__":
    # Chạy solver trên bài toán MetricTSP thử nghiệm
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
