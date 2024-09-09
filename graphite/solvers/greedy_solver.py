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
            # Tính toán min_distance_threshold
            max_distance = max(max(row) for row in distance_matrix)
            min_distance_threshold = max_distance / 5 

            # Cải thiện đường đi bằng cách chuyển các đoạn đường chéo thành các đoạn không chéo
            best_route = improve_route(best_route, distance_matrix, min_distance_threshold)
            
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

def improve_route(route: List[int], distance_matrix: List[List[Union[int, float]]], min_distance_threshold: float) -> List[int]:
    """Cải thiện đường đi bằng cách thay thế các đoạn đường chéo bằng các đoạn không chéo, chỉ xét các đoạn đường xa."""
    n = len(route)
    improved_route = route[:]
    improved = True

    while improved:
        improved = False
        # Xem xét từng cặp đoạn đường không liền kề
        for i in range(1, n - 2):
            for j in range(i + 2, n - (1 if i == 0 else 0)):
                if distance_matrix[improved_route[i]][improved_route[i+1]] > min_distance_threshold and \
                   distance_matrix[improved_route[j]][improved_route[(j+1) % len(improved_route)]] > min_distance_threshold:
                    if can_be_improved(improved_route, i, j, distance_matrix):
                        improved_route = apply_2opt(improved_route, i, j)
                        improved = True
                        break
            if improved:
                break

    return improved_route

def can_be_improved(route: List[int], i: int, j: int, distance_matrix: List[List[Union[int, float]]]) -> bool:
    """Kiểm tra xem việc hoán đổi đoạn đường từ i đến j có làm giảm khoảng cách không."""
    if j - i == 1:  # Không phải là đoạn chéo
        return False

    # Tính khoảng cách trước và sau khi hoán đổi
    d1 = distance_matrix[route[i - 1]][route[i]] + distance_matrix[route[j]][route[(j + 1) % len(route)]]
    d2 = distance_matrix[route[i - 1]][route[j]] + distance_matrix[route[i]][route[(j + 1) % len(route)]]

    return d2 < d1

def apply_2opt(route: List[int], i: int, j: int) -> List[int]:
    """Hoán đổi đoạn đường từ i đến j."""
    new_route = route[:]
    new_route[i:j + 1] = reversed(new_route[i:j + 1])
    return new_route

        
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
