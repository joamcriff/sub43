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
        # Khởi tạo giải pháp với Nearest Neighbour
        initial_route = await self.solve_with_nearest_neighbour(formatted_problem, future_id)
        
        # Tối ưu hóa giải pháp với LKH
        optimized_route = await self.solve_with_lkh(formatted_problem, future_id, initial_route)
        
        return optimized_route

    async def solve_with_nearest_neighbour(self, formatted_problem: List[List[Union[int, float]]], future_id: int) -> List[int]:
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

        return best_route

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

        return route, total_distance

    async def solve_with_lkh(self, formatted_problem: List[List[Union[int, float]]], future_id: int, initial_route: List[int]) -> List[int]:
        n = len(formatted_problem)

        # Tạo file .tsp cho LKH
        tsp_file = "problem.tsp"
        with open(tsp_file, "w") as f:
            f.write(f"NAME: tsp_problem\n")
            f.write(f"TYPE: TSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write(f"EDGE_WEIGHT_SECTION\n")
            for row in formatted_problem:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("EOF\n")

        # Tạo file .par (parameter file) cho LKH
        par_file = "problem.par"
        with open(par_file, "w") as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\n")
            f.write("OUTPUT_TOUR_FILE = solution.tour\n")
            f.write("RUNS = 10\n")  # Thực hiện nhiều lần chạy để cải thiện chất lượng giải pháp
            f.write("PATCHING_C = 0.5\n")  # Kích thước của các patching phases (tham số có thể thay đổi tùy theo nhu cầu)
            f.write("PATCHING_A = 0.2\n")  # Điều chỉnh kích thước của các patching phases (tham số có thể thay đổi tùy theo nhu cầu)
            f.write(f"INITIAL_TOUR_FILE = initial_solution.tour\n")



        # Tạo file .tour chứa giải pháp khởi tạo từ NN
        init_tour_file = "initial_solution.tour"
        with open(init_tour_file, "w") as init_f:
            init_f.write("TOUR_SECTION\n")
            for node in initial_route:
                init_f.write(f"{node + 1}\n")  # Chuyển đổi từ chỉ số 0-based sang 1-based
            init_f.write("-1\n")  # Kết thúc danh sách điểm tour

        # Chạy LKH solver thông qua dòng lệnh
        result = subprocess.run([self.lkh_path, par_file], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"LKH solver failed with exit code {result.returncode}\nError Output: {result.stderr}")

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
        if route and route[0] != route[-1]:
            route.append(route[0])

        # Xóa các file tạm
        os.remove(tsp_file)
        os.remove(par_file)
        os.remove(tour_file)
        os.remove(init_tour_file)

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
