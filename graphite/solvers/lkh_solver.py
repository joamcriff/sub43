from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import time
import asyncio
import random
from typing import List, Union, Tuple

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphProblem]=[GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem:List[List[Union[int, float]]], future_id:int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])

        # Khởi tạo với hai node, tạo cạnh đầu tiên
        route = [0, 1, 0]  # Lộ trình bắt đầu với hai node đầu tiên
        visited = [False] * n
        visited[0], visited[1] = True, True

        # Thêm các node còn lại vào lộ trình bằng cách sử dụng thuật toán Nearest Insertion
        while len(route) < n + 1:  # route phải dài hơn n vì quay lại node đầu tiên
            if self.future_tracker.get(future_id):
                return None

            # Tìm node chưa được thăm gần nhất với bất kỳ node nào trong lộ trình hiện tại
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

            # Chèn node gần nhất vào vị trí tốt nhất trong lộ trình
            route.insert(best_insertion_index, nearest_node)
            visited[nearest_node] = True

        return route

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