from typing import List, Union, Tuple
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import time
import asyncio
import random

def path_cost_from_distance_matrix(distances: np.ndarray, path: List[int]) -> float:
    """Tính tổng khoảng cách của một lộ trình từ ma trận khoảng cách."""
    return np.sum([distances[path[i], path[i + 1]] for i in range(len(path) - 1)])

def two_opt_change(route: List[int], i: int, j: int) -> List[int]:
    """Thực hiện phép toán 2-opt để đảo ngược đoạn lộ trình giữa các chỉ số i và j."""
    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
    return new_route

def two_opt(path: List[int], distances: np.ndarray) -> Tuple[float, List[int]]:
    """Cải thiện lộ trình bằng thuật toán 2-opt."""
    best_distance = path_cost_from_distance_matrix(distances, path)
    present_route = path.copy()

    for i in range(1, len(path) - 2):  # Bắt đầu từ 1 để bỏ qua node 0
        for j in range(i + 1, len(path) - 1):
            new_route = two_opt_change(present_route, i, j)
            new_distance = path_cost_from_distance_matrix(distances, new_route)

            if new_distance < best_distance:
                present_route = new_route
                best_distance = new_distance

    return best_distance, present_route

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = np.array(formatted_problem)
        n = len(distance_matrix[0])
        visited = [False] * n
        route = []
        total_distance = 0

        # Khởi tạo lộ trình với Nearest Neighbour
        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return None
            # Tìm điểm gần nhất chưa thăm
            nearest_distance = np.inf
            nearest_node = random.choice([i for i, is_visited in enumerate(visited) if not is_visited])  # Lựa chọn ngẫu nhiên điểm chưa thăm
            for j in range(n):
                if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                    nearest_distance = distance_matrix[current_node][j]
                    nearest_node = j

            # Di chuyển đến điểm gần nhất chưa thăm
            route.append(nearest_node)
            visited[nearest_node] = True
            total_distance += nearest_distance
            current_node = nearest_node

        # Quay lại điểm bắt đầu
        total_distance += distance_matrix[current_node][route[0]]
        route.append(route[0])

        # Cải thiện lộ trình bằng thuật toán 2-opt
        best_distance, best_route = two_opt(route, distance_matrix)

        return best_route

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
