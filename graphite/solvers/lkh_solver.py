from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import subprocess
import asyncio
import time
import os
from typing import List, Union

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.lkh_path = os.path.abspath("/root/LKH-2.0.10/LKH")  # Đường dẫn đến tệp thực thi LKH mà bạn đã cài đặt

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        # Chuyển đổi ma trận khoảng cách sang định dạng .tsp mà LKH yêu cầu
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
            f.write("RUNS = 1\n")  # Thực hiện 1 lần chạy LKH
        
        # Chạy LKH solver thông qua dòng lệnh
        subprocess.run([self.lkh_path, par_file], check=True)

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
                    route.append(node - 1)  # LKH sử dụng chỉ số 1-based, cần chuyển về 0-based
        return route

    def problem_transformations(self, problem: GraphProblem):
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
