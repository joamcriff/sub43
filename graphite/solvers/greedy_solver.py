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
        n = len(distance_matrix)
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

        if best_route:
            # Thực hiện tối ưu hóa bằng Simulated Annealing
            best_route = self.simulated_annealing(distance_matrix, best_route)
            
            # Đảm bảo rằng đường đi bao gồm tất cả các điểm và trở về điểm xuất phát
            if len(set(best_route)) == n and best_route[0] == best_route[-1]:
                return best_route
            else:
                raise ValueError("Optimized route does not include all cities or does not return to the start")

        return []

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

    def simulated_annealing(self, distance_matrix: List[List[Union[int, float]]], initial_route: List[int], initial_temp: float = 1000.0, cooling_rate: float = 0.995, min_temp: float = 1.0) -> List[int]:
        """Thực hiện tối ưu hóa bằng Simulated Annealing."""
        current_route = initial_route
        current_distance = self.calculate_total_distance(distance_matrix, current_route)
        best_route = current_route
        best_distance = current_distance
        temperature = initial_temp

        while temperature > min_temp:
            # Tạo biến thể ngẫu nhiên của đường đi hiện tại
            new_route = self.swap_two_segments(current_route)
            new_distance = self.calculate_total_distance(distance_matrix, new_route)

            # Tính toán sự chấp nhận theo tiêu chí nhiệt độ
            if self.acceptance_probability(current_distance, new_distance, temperature) > random.random():
                current_route = new_route
                current_distance = new_distance

                # Cập nhật đường đi tốt nhất
                if current_distance < best_distance:
                    best_route = current_route
                    best_distance = current_distance

            # Giảm nhiệt độ
            temperature *= cooling_rate

        return best_route

    def swap_two_segments(self, route: List[int]) -> List[int]:
        """Tạo biến thể ngẫu nhiên bằng cách hoán đổi hai đoạn đường."""
        new_route = route[:]
        n = len(new_route)
        idx1, idx2 = random.sample(range(1, n-1), 2)  # Tránh hoán đổi điểm bắt đầu và kết thúc
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        new_route[idx1:idx2] = reversed(new_route[idx1:idx2])
        return new_route

    def acceptance_probability(self, old_cost: float, new_cost: float, temperature: float) -> float:
        """Tính toán xác suất chấp nhận dựa trên sự thay đổi chi phí và nhiệt độ."""
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / temperature)

    def calculate_total_distance(self, distance_matrix: List[List[Union[int, float]]], route: List[int]) -> float:
        """Tính tổng khoảng cách của đường đi."""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i + 1]]
        return total_distance


if __name__ == "__main__":
    # Chạy solver trên bài toán MetricTSP thử nghiệm
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = NearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    try:
        route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
        print(f"{solver.__class__.__name__} Solution: {route}")
        print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
    except ValueError as e:
        print(f"Error: {e}")
