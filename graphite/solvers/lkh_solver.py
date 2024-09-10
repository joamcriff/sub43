from pyconcorde.tsp import TSPSolver
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import time
import asyncio
from typing import List, Union

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        
        # Flatten the distance matrix for Concorde input
        flattened_distances = []
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle is needed for undirected graphs
                flattened_distances.append(distance_matrix[i][j])
        
        # Khởi tạo solver Concorde với dữ liệu khoảng cách
        solver = TSPSolver.from_data(range(n), norm="GEO", dist=flattened_distances)
        
        # Giải quyết bài toán TSP
        solution = solver.solve()

        # Trả về lộ trình tối ưu với vòng khép kín (quay lại điểm xuất phát)
        optimal_route = solution.tour
        optimal_route.append(optimal_route[0])  # Thêm điểm khởi đầu vào cuối

        return optimal_route

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges

if __name__ == "__main__":
    # Chạy solver trên bài toán MetricTSP thử nghiệm
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHGeneticSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
