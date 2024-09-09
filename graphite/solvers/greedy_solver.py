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
        num_starts = 10  # Đảm bảo ít nhất 1 điểm bắt đầu

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

        return best_route

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

            # Tìm điểm gần nhất đến toàn bộ các điểm đã thăm
            nearest_node = None
            min_total_increase = np.inf
            for j in range(n):
                if not visited[j]:
                    total_increase = self.calculate_total_increase(route, j, distance_matrix)
                    if total_increase < min_total_increase:
                        min_total_increase = total_increase
                        nearest_node = j

            if nearest_node is None:
                break

            # Chèn điểm gần nhất vào vị trí tốt nhất trong lộ trình
            best_insertion_index = self.find_best_insertion(route, nearest_node, distance_matrix)
            route.insert(best_insertion_index, nearest_node)
            visited[nearest_node] = True
            total_distance += min_total_increase

        # Trở về điểm xuất phát nếu chưa có
        if route[-1] != start_node:
            total_distance += distance_matrix[route[-1]][start_node]
            route.append(start_node)

        return route, total_distance

    def calculate_total_increase(self, route: List[int], new_node: int, distance_matrix: List[List[Union[int, float]]]) -> float:
        """Tính toán tổng mức tăng khoảng cách khi thêm new_node vào lộ trình."""
        min_increase = np.inf
        for i in range(1, len(route)):
            increase = (distance_matrix[route[i - 1]][new_node] +
                        distance_matrix[new_node][route[i]] -
                        distance_matrix[route[i - 1]][route[i]])
            if increase < min_increase:
                min_increase = increase
        return min_increase

    def find_best_insertion(self, route: List[int], new_node: int, distance_matrix: List[List[Union[int, float]]]) -> int:
        """Tìm vị trí tốt nhất để chèn new_node vào lộ trình."""
        best_index = 0
        min_increase = np.inf
        for i in range(1, len(route)):
            increase = (distance_matrix[route[i - 1]][new_node] +
                        distance_matrix[new_node][route[i]] -
                        distance_matrix[route[i - 1]][route[i]])
            if increase < min_increase:
                min_increase = increase
                best_index = i
        return best_index

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
