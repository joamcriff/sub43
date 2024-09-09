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

        n = len(route)
        best_distance = calculate_total_distance(route)
        improved = True

        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        # Thử tất cả các hoán đổi có thể với 3-opt
                        new_route = self.reverse_segment_if_better(route, i, j, k, distance_matrix)
                        new_distance = calculate_total_distance(new_route)

                        if new_distance < best_distance:
                            route = new_route
                            best_distance = new_distance
                            improved = True

        return route

    def reverse_segment_if_better(self, route: List[int], i: int, j: int, k: int, distance_matrix: List[List[Union[int, float]]]) -> List[int]:
        """Thử tất cả các hoán đổi 3 cạnh có thể và chọn cách tốt nhất."""
        A, B, C, D, E, F = route[i - 1], route[i], route[j - 1], route[j], route[k - 1], route[k % len(route)]

        # Tính toán khoảng cách hiện tại
        d0 = (distance_matrix[A][B] + distance_matrix[C][D] + distance_matrix[E][F])
        
        # 7 hoán đổi khác nhau có thể cho 3-opt
        d1 = (distance_matrix[A][C] + distance_matrix[B][D] + distance_matrix[E][F])
        d2 = (distance_matrix[A][B] + distance_matrix[C][F] + distance_matrix[D][E])
        d3 = (distance_matrix[A][D] + distance_matrix[E][B] + distance_matrix[C][F])
        d4 = (distance_matrix[A][C] + distance_matrix[E][D] + distance_matrix[B][F])
        d5 = (distance_matrix[A][E] + distance_matrix[B][C] + distance_matrix[D][F])
        d6 = (distance_matrix[A][E] + distance_matrix[D][B] + distance_matrix[C][F])
        
        # Chọn hoán đổi có khoảng cách ngắn nhất
        distances = [d0, d1, d2, d3, d4, d5, d6]
        best_distance = min(distances)

        if best_distance == d0:
            return route  # Không thay đổi
        elif best_distance == d1:
            return route[:i] + route[i:j][::-1] + route[j:k][::-1] + route[k:]
        elif best_distance == d2:
            return route[:i] + route[i:k][::-1] + route[k:]
        elif best_distance == d3:
            return route[:i] + route[i:j] + route[j:k][::-1] + route[k:]
        elif best_distance == d4:
            return route[:i] + route[j:k] + route[i:j][::-1] + route[k:]
        elif best_distance == d5:
            return route[:i] + route[j:k][::-1] + route[i:j] + route[k:]
        elif best_distance == d6:
            return route[:i] + route[k-1:j-1:-1] + route[i:j] + route[k:]

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
