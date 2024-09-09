from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import time
import asyncio
import random
from typing import List, Union, Tuple

class LinKernighanSolver:
    def __init__(self, distance_matrix: List[List[Union[int, float]]]):
        self.distance_matrix = distance_matrix

    def solve(self, route: List[int]) -> List[int]:
        # Thực hiện thuật toán Lin-Kernighan để tối ưu hóa tuyến đường
        # Đây là một phiên bản đơn giản và bạn có thể cần cài đặt thực tế cho thuật toán LK
        return self.lin_kernighan(route)

    def lin_kernighan(self, route: List[int]) -> List[int]:
        # Giả định thực hiện thuật toán Lin-Kernighan (LK) trên tuyến đường
        # Đây là một phiên bản đơn giản hóa. Để thực hiện thuật toán LK thực sự, cần mã phức tạp hơn.
        # Để biết thêm chi tiết, bạn nên tham khảo tài liệu về LK.
        best_route = route
        best_distance = self.calculate_total_distance(route)
        improved = True

        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    new_route = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:]
                    new_distance = self.calculate_total_distance(new_route)
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
        
        return best_route

    def calculate_total_distance(self, route: List[int]) -> float:
        return sum(self.distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

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

        # Áp dụng thuật toán Lin-Kernighan để tối ưu hóa đường đi
        if best_route:
            lk_solver = LinKernighanSolver(distance_matrix)
            optimized_route = lk_solver.solve(best_route)
            # Đảm bảo tuyến đường hoàn chỉnh và hợp lệ
            if len(optimized_route) == n + 1 and optimized_route[0] == optimized_route[-1]:
                return optimized_route
            else:
                raise ValueError("Optimized route has an invalid number of cities.")
        else:
            raise ValueError("No valid route found.")

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

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges

        
if __name__ == "__main__":
    # Chạy solver trên bài toán MetricTSP thử nghiệm
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    try:
        route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
        print(f"{solver.__class__.__name__} Solution: {route}")
    except ValueError as e:
        print(f"Error: {e}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
