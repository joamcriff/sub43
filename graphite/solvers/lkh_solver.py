from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np
import subprocess
import asyncio
import time
import os
from typing import List, Union, Tuple
import io

class LKHGeneticSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.lkh_path = os.path.abspath("/root/LKH-2.0.10/LKH")  # Đường dẫn đến tệp thực thi LKH

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
        # Chuyển đổi ma trận khoảng cách sang định dạng .tsp mà LKH yêu cầu
        distance_matrix = np.array(formatted_problem)
        n = len(distance_matrix)

        # Tạo file .tsp trong bộ nhớ tạm (in-memory)
        tsp_file = io.StringIO()
        tsp_file.write(f"NAME: tsp_problem\nTYPE: TSP\nDIMENSION: {n}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        for row in distance_matrix:
            tsp_file.write(" ".join(map(str, row)) + "\n")
        tsp_file.write("EOF\n")
        tsp_file.seek(0)  # Đặt con trỏ về đầu file

        # Tạo file .par (parameter file) cho LKH
        par_file = io.StringIO()
        par_file.write("PROBLEM_FILE = problem.tsp\n")
        par_file.write("OUTPUT_TOUR_FILE = solution.tour\n")
        par_file.write("RUNS = 3\n")  # Tăng số lần chạy để tăng chất lượng
        par_file.seek(0)  # Đặt con trỏ về đầu file

        # Chạy LKH solver thông qua dòng lệnh
        result = subprocess.run([self.lkh_path, par_file], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"LKH solver failed with exit code {result.returncode}")

        # Đọc file kết quả solution.tour trong bộ nhớ
        tour_file = io.StringIO(result.stdout)  # Giả sử output từ LKH được chuyển trực tiếp ra stdout
        route = []
        reading_tour = False
        for line in tour_file:
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
