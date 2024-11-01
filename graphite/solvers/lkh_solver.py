from typing import List, Union
import matplotlib.pyplot as plt
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.utils.graph_utils import timeout, get_multi_minmax_tour_distance
from graphite.solvers.greedy_solver_multi import NearestNeighbourMultiSolver
from graphite.data.dataset_utils import load_default_dataset
from graphite.utils.graph_utils import timeout
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
import numpy as np
import time
import asyncio
import random

import bittensor as bt
import os
import subprocess
import tempfile
from io import StringIO
# from greedy_solver import NearestNeighbourSolver

class LKHSolver(BaseSolver):
    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)
        self.lkh_path = "./LKH-3.0.11/LKH"
    
    def create_problem_file(self, distance_matrix):
        dimension = len(distance_matrix)
        problem_file_content = f"""NAME: ATSP
        TYPE: ATSP
        DIMENSION: {dimension}
        EDGE_WEIGHT_TYPE: EXPLICIT
        EDGE_WEIGHT_FORMAT: FULL_MATRIX
        EDGE_WEIGHT_SECTION
        """
        # Sử dụng StringIO và np.savetxt để tạo chuỗi cho ma trận khoảng cách
        buffer = StringIO()
        np.savetxt(buffer, distance_matrix, fmt='%d', delimiter=' ')
        matrix_string = buffer.getvalue().strip()
        problem_file_content += matrix_string + "\nEOF\n"
        return problem_file_content
    
    def create_parameter_file(self, problem_file_path, tour_file_path, salesmen=2, nodes=5000):
        trial = int(500 * 5000 / nodes)
        parameter_file_content = f"""PROBLEM_FILE = {problem_file_path}
        TOUR_FILE = {tour_file_path}
        SALESMEN = {salesmen}
        INITIAL_PERIOD = 100
        PRECISION = 1e-04
        RUNS = 1
        INITIAL_TOUR_ALGORITHM = GREEDY
        MAX_CANDIDATES = 6
        TRACE_LEVEL = 1
        OPTIMUM = 1183
        MTSP_OBJECTIVE = MINMAX
        """
        return parameter_file_content
    
        # KICK_TYPE = 15
        # KICKS = 20
        # POPULATION_SIZE = 20
        # MAX_TRIALS = {trial}
        # TIME_LIMIT = 20
        # TOTAL_TIME_LIMIT = 20
    
    async def solve(self, formatted_problem, future_id:int)->List[int]:
        with tempfile.NamedTemporaryFile('w+', prefix='problem_', suffix='.txt', delete=False) as problem_file, \
            tempfile.NamedTemporaryFile('w+', prefix='param_', suffix='.txt', delete=False) as parameter_file, \
            tempfile.NamedTemporaryFile('r+', prefix='tour_', suffix='.txt', delete=False) as tour_file:

            problem_file_content = self.create_problem_file(formatted_problem.edges)
            problem_file.write(problem_file_content)
            problem_file.flush()
            # Trích xuất thông tin về số lượng salesman, depot và kiểu depot
            salesmen = formatted_problem.n_salesmen
            
            parameter_file_content = self.create_parameter_file(
                problem_file.name, tour_file.name, salesmen, len(formatted_problem.edges)
            )
            parameter_file.write(parameter_file_content)
            parameter_file.flush()
            # Chạy LKH
            subprocess.run([self.lkh_path, parameter_file.name], check=True)
            
            # Đọc file tour
            tour_file.seek(0)
            tour = self.parse_tour_file(tour_file.name)

        # Xóa các file tạm
        os.remove(problem_file.name)
        os.remove(parameter_file.name)
        os.remove(tour_file.name)

        return tour
    
    def calculate_total_distance(self, tour, distance_matrix):
        total_distance = 0
        for i in range(len(tour)):
            total_distance += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return total_distance
    
    def parse_tour_file(self, tour_file_path):
        tour = []
        with open(tour_file_path, 'r') as file:
            in_tour_section = False
            for line in file:
                if line.strip() == 'TOUR_SECTION':
                    in_tour_section = True
                elif line.strip() == '-1':
                    break
                elif in_tour_section:
                    tour.append(int(line.strip()) - 1)  # LKH uses 1-based indexing
        tour.append(tour[0])
        return tour

    def problem_transformations(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
        return problem
    
if __name__ == "__main__":
    ## Test case for GraphV2Problem
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
    class Mock:
        def __init__(self) -> None:
            pass        

        def recreate_edges(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
            node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
            node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
            if problem.cost_function == "Geom":
                return geom_edges(node_coords)
            elif problem.cost_function == "Euclidean2D":
                return euc_2d_edges(node_coords)
            elif problem.cost_function == "Manhatten2D":
                return man_2d_edges(node_coords)
            else:
                return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
    mock = Mock()
    load_default_dataset(mock)

    n_nodes = 2000
    m = 10
    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, selected_ids=random.sample(list(range(100000)),n_nodes), dataset_ref="Asia_MSB", n_salesmen=m, depots=[0]*m)
    test_problem.edges = mock.recreate_edges(test_problem)

    lkh_solver = LKHSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(lkh_solver.solve_problem(test_problem))
    # total_distance = lkh_solver.calculate_total_distance(route, test_problem.edges)
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route)
    score1 = get_multi_minmax_tour_distance(test_synapse)

    solver2 = NearestNeighbourMultiSolver(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route2)
    score2 = get_multi_minmax_tour_distance(test_synapse)

    print(f"{lkh_solver.__class__.__name__} Tour: {route}")
    # print(f"Total distance of the tour: {total_distance}")
    print(f"{lkh_solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
    print(f"LKH scored: {score1} while Multi scored: {score2}")
