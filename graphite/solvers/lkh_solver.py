from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import subprocess
import asyncio
import time
import os
import random
from typing import List, Union, Tuple

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.lkh_path = os.path.abspath("/root/LKH-2.0.10/LKH")  # Đường dẫn đến tệp thực thi LKH

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        # Tạo giải pháp khởi đầu bằng Nearest Neighbour
        initial_route = await self.solve_with_nearest_neighbour(formatted_problem, future_id)
        
        # Tối ưu hóa giải pháp khởi đầu với LKH
        optimized_route = await self.solve_with_lkh(formatted_problem, future_id, initial_route)
        
        return optimized_route

    async def solve_with_nearest_neighbour(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)
        
        # Khởi tạo giải pháp với Nearest Neighbour
        visited = [False] * n
        route = []
        total_distance = 0

        current_node = random.randint(0, n - 1)  # Bắt đầu từ một node ngẫu nhiên
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
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
        total_distance += distance_matrix[current_node][route[0]]
        route.append(route[0])

        return route

    async def solve_with_lkh(self, formatted_problem: List[List[Union[int, float]]], future_id: int, initial_route: List[int]) -> List[int]:
        distance_matrix = np.array(formatted_problem)
        n = len(distance_matrix)

        # Tạo file .tsp cho đầu vào
        tsp_file = "problem.tsp"
        with open(tsp_file, "w") as f:
            f.write(f"NAME: tsp_problem\nTYPE: TSP\nDIMENSION: {n}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            for row in distance_matrix:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("EOF\n")
        
        # Tạo file .par (parameter file) cho LKH
        par_file = "problem.par"
        with open(par_file, "w") as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\n")
            f.write("OUTPUT_TOUR_FILE = solution.tour\n")
            f.write("RUNS = 5\n")  # Thực hiện nhiều lần chạy LKH

            # Thêm giải pháp khởi tạo vào file .par
            initial_route_str = " ".join(str(node + 1) for node in initial_route)
            f.write(f"INITIAL_TOUR_FILE = initial_solution.tour\n")

            # Tạo file .tour chứa giải pháp khởi tạo
            with open("initial_solution.tour", "w") as init_f:
                init_f.write("TOUR_SECTION\n")
                for node in initial_route:
                    init_f.write(f"{node + 1}\n")
                init_f.write("-1\n")

        # Chạy LKH solver thông qua dòng lệnh
        result = subprocess.run([self.lkh_path, par_file], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"LKH solver failed with exit code {result.returncode}")

        # Đọc file kết quả solution.tour
        tour_file = "solution.tour"
        route = []
        with open(tour_file, "r") as f:
            reading_tour = False
            for line in f:
                if "TOUR_SECTION" in line:
                    reading_tour = True
                    continue
                if reading_tour:
                    node = int(line.strip())
                    if node == -1:
                        break
                    route.append(node - 1)  # Chuyển đổi chỉ số từ 1-based sang 0-based

        # Đảm bảo chu trình khép kín bằng cách thêm lại node đầu tiên vào cuối
        if route[0] != route[-1]:
            route.append(route[0])
        
        # Xóa các file tạm
        os.remove(tsp_file)
        os.remove(par_file)
        os.remove(tour_file)
        os.remove("initial_solution.tour")

        return route

    def problem_transformations(self, problem: GraphProblem) -> List[List[Union[int, float]]]:
        return problem.edges

# Đoạn mã kiểm tra
if __name__ == "__main__":
    # Chạy solver trên bài toán MetricTSP thử nghiệm
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHGeneticSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
