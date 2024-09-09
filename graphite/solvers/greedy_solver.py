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

        # Áp dụng thuật toán Nearest Insertion để tối ưu hóa đường đi
        optimized_route = self.nearest_insertion(distance_matrix)

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

    def nearest_insertion(self, distance_matrix: List[List[Union[int, float]]]) -> List[int]:
        """Thực hiện thuật toán Nearest Insertion để tối ưu hóa tuyến đường."""
        def calculate_total_distance(route):
            return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

        n = len(distance_matrix)
        visited = [False] * n
        route = []

        # Chọn điểm khởi đầu
        start_node = random.randint(0, n - 1)
        route.append(start_node)
        visited[start_node] = True

        # Tiến hành thêm các đỉnh
        while len(route) < n:
            best_insertion = None
            best_distance = float('inf')
            for i in range(n):
                if not visited[i]:
                    # Tìm điểm tốt nhất để chèn vào route
                    for j in range(len(route)):
                        distance_if_inserted = (distance_matrix[route[j]][i] + distance_matrix[i][route[(j + 1) % len(route)]])
                        if distance_if_inserted < best_distance:
                            best_distance = distance_if_inserted
                            best_insertion = (i, j)

            # Thực hiện chèn điểm vào route
            insert_node, pos = best_insertion
            route.insert((pos + 1) % len(route), insert_node)
            visited[insert_node] = True

        # Thêm đường trở về điểm xuất phát
        route.append(route[0])

        return route

    def problem_transformations(self, problem: GraphProblem):
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
