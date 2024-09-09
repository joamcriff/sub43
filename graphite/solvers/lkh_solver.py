from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
from graphite.utils.graph_utils import timeout
import numpy as np
import time
import asyncio
import random

import bittensor as bt

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphProblem]=[GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem:List[List[Union[int, float]]], future_id:int, num_starts:int=10) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])

        # Tìm kiếm nhiều hướng đồng thời
        start_nodes = random.sample(range(n), min(num_starts, n))  # Chọn num_starts điểm bắt đầu ngẫu nhiên
        routes = await asyncio.gather(
            *[self.find_route_from_start(distance_matrix, start_node, n, future_id) for start_node in start_nodes]
        )

        # Chọn route ngắn nhất từ các route đã tìm
        best_route = min(routes, key=lambda route: self.calculate_total_distance(route, distance_matrix))
        return best_route

    async def find_route_from_start(self, distance_matrix:List[List[Union[int, float]]], start_node:int, n:int, future_id:int) -> List[int]:
        # Khởi tạo lộ trình với node bắt đầu
        route = [start_node, (start_node + 1) % n, start_node]  # Lộ trình bắt đầu với hai node đầu tiên và quay lại node khởi đầu
        visited = [False] * n
        visited[start_node], visited[(start_node + 1) % n] = True, True

        # Thêm các node chưa được thăm vào lộ trình
        while len(route) < n + 1:  # route phải dài hơn n vì quay lại node đầu tiên
            if self.future_tracker.get(future_id):
                return None

            # Tìm node chưa được thăm gần nhất
            nearest_node = -1
            nearest_distance = np.inf
            for i in range(n):
                if not visited[i]:
                    for node in route[:-1]:  # Không tính node cuối vì nó là node khởi đầu
                        if distance_matrix[node][i] < nearest_distance:
                            nearest_node = i
                            nearest_distance = distance_matrix[node][i]

            # Tìm vị trí tốt nhất để chèn node gần nhất vào lộ trình
            best_insertion_index = 0
            min_increase = np.inf
            for i in range(1, len(route)):
                increase = (distance_matrix[route[i - 1]][nearest_node] + 
                            distance_matrix[nearest_node][route[i]] - 
                            distance_matrix[route[i - 1]][route[i]])
                if increase < min_increase:
                    min_increase = increase
                    best_insertion_index = i

            # Chèn node vào vị trí tốt nhất
            route.insert(best_insertion_index, nearest_node)
            visited[nearest_node] = True

        return route

    def calculate_total_distance(self, route: List[int], distance_matrix: List[List[Union[int, float]]]) -> float:
        total_distance = 0
        for i in range(1, len(route)):
            total_distance += distance_matrix[route[i - 1]][route[i]]
        return total_distance

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__=="__main__":
    # Chạy solver với một bài toán MetricTSP thử nghiệm
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHGeneticSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"Giải pháp {solver.__class__.__name__}: {route}")
    print(f"Thời gian thực hiện cho {n_nodes} node: {time.time()-start_time}")
