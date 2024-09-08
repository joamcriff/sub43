from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import time
import asyncio
import random

def path_cost_from_distance_matrix(distances: np.ndarray, path: List[int]) -> float:
    """Tính tổng khoảng cách của một lộ trình từ ma trận khoảng cách."""
    return np.sum([distances[path[i], path[i + 1]] for i in range(len(path) - 1)])

class NearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = np.array(formatted_problem)
        n = len(distance_matrix[0])

        best_route = None
        best_distance = np.inf
        start_nodes = random.sample(range(n), min(n, 10))
        # Khởi tạo từ nhiều điểm bắt đầu khác nhau
        for start_node in range(n):
            visited = [False] * n
            route = []
            total_distance = 0

            current_node = start_node
            route.append(current_node)
            visited[current_node] = True

            for _ in range(n - 1):
                if self.future_tracker.get(future_id):
                    return None
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

            # Quay lại điểm bắt đầu
            total_distance += distance_matrix[current_node][start_node]
            route.append(start_node)

            # Cập nhật lộ trình tốt nhất
            if total_distance < best_distance:
                best_distance = total_distance
                best_route = route

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
