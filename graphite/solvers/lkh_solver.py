from typing import List, Union
import matplotlib.pyplot as plt
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.solvers.greedy_solver_multi_2 import NearestNeighbourMultiSolver2
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

class LKHSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphV2Problem] = [GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)
        self.lkh_path = "./LKH-3.0.11/LKH"
    
    def create_problem_file(self, distance_matrix, salesmen):
        dimension = len(distance_matrix)
        problem_file_content = f"""NAME: ATSP
        TYPE: ATSP
        DIMENSION: {dimension}
        VEHICLES : {salesmen}
        EDGE_WEIGHT_TYPE: EXPLICIT
        EDGE_WEIGHT_FORMAT: FULL_MATRIX
        EDGE_WEIGHT_SECTION
        """
        buffer = StringIO()
        np.savetxt(buffer, distance_matrix, fmt='%d', delimiter=' ')
        matrix_string = buffer.getvalue().strip()
        problem_file_content += matrix_string + "\nEOF\n"
        return problem_file_content
    
    def create_parameter_file(self, problem_file_path, tour_file_path, salesmen=2, nodes=5000):
        trial = int(500 * 5000 / nodes)
        parameter_file_content = f"""PROBLEM_FILE = {problem_file_path}
        TOUR_FILE = {tour_file_path}     
        KICK_TYPE = 15
        KICKS = 20
        POPULATION_SIZE = 20
        MAX_TRIALS = {trial}
        SALESMEN = {salesmen}
        INITIAL_PERIOD = 100
        PRECISION = 1e-04
        RUNS = 1
        INITIAL_TOUR_ALGORITHM = GREEDY
        MAX_CANDIDATES = 5
        TRACE_LEVEL = 1
        OPTIMUM = 1183
        MTSP_OBJECTIVE = MINMAX
        TIME_LIMIT = 17
        TOTAL_TIME_LIMIT = 17
        """
        return parameter_file_content

    async def solve(self, formatted_problem, future_id: int) -> List[List[int]]:
        with tempfile.NamedTemporaryFile('w+', prefix='problem_', suffix='.txt', delete=False) as problem_file, \
            tempfile.NamedTemporaryFile('w+', prefix='param_', suffix='.txt', delete=False) as parameter_file, \
            tempfile.NamedTemporaryFile('r+', prefix='tour_', suffix='.txt', delete=False) as tour_file:

            problem_file_content = self.create_problem_file(formatted_problem.edges, formatted_problem.n_salesmen)
            problem_file.write(problem_file_content)
            problem_file.flush()
            salesmen = formatted_problem.n_salesmen
            
            parameter_file_content = self.create_parameter_file(
                problem_file.name, tour_file.name, salesmen, len(formatted_problem.edges)
            )
            parameter_file.write(parameter_file_content)
            parameter_file.flush()
            subprocess.run([self.lkh_path, parameter_file.name], check=True)
            
            tour_file.seek(0)
            tour = self.parse_tour_file(tour_file.name)

        os.remove(problem_file.name)
        os.remove(parameter_file.name)
        os.remove(tour_file.name)

        # Transform the tour to match the output format of NearestNeighbourMultiSolver
        tours = self.split_into_sublists(tour[1:], formatted_problem.n_salesmen)
        
        # Debugging output to verify tours
        all_nodes = set()
        for idx, tour in enumerate(tours):
            print(f"Tour for salesman {idx + 1}: {tour}")
            all_nodes.update(tour)
        
        closed_tours = [[0] + tour + [0] for tour in tours]

        # Verify that all nodes except the depot are included exactly once
        expected_nodes = set(range(1, len(formatted_problem.edges)))  # Nodes 1 to n-1
        visited_nodes = set(node for tour in closed_tours for node in tour if node != 0)
        
        print("Visited nodes:", sorted(visited_nodes))
        print("Expected nodes:", sorted(expected_nodes))
        
        # Detailed difference check
        missing_nodes = expected_nodes - visited_nodes
        extra_nodes = visited_nodes - expected_nodes
        
        assert missing_nodes == set() and extra_nodes == set(), \
            f"Missing nodes: {missing_nodes}, Extra nodes: {extra_nodes}"
        
        return closed_tours
    
    def split_into_sublists(self, original_list, n_salesmen):
        n = len(original_list)
        sublist_size = n // n_salesmen
        remainder = n % n_salesmen

        sublists = []
        start_index = 0

        for i in range(n_salesmen):
            if i < remainder:
                size = sublist_size + 1
            else:
                size = sublist_size
                
            sublists.append(original_list[start_index:start_index + size])
            start_index += size

        return sublists

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
        tour.append(tour[0])  # Ensure the tour closes by returning to the start
        return tour

    def problem_transformations(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
        return problem
    
if __name__ == "__main__":
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
    m = 5
    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, selected_ids=random.sample(list(range(100000)),n_nodes), dataset_ref="Asia_MSB", n_salesmen=m, depots=[0]*m)
    test_problem.edges = mock.recreate_edges(test_problem)

    # lkh_solver = LKHSolver(problem_types=[test_problem])
    # start_time = time.time()
    # route = asyncio.run(lkh_solver.solve_problem(test_problem))
    # test_synapse = GraphV2Synapse(problem = test_problem, solution = route)
    # score1 = get_multi_minmax_tour_distance(test_synapse)

    solver1 = NearestNeighbourMultiSolver(problem_types=[test_problem])
    route1 = asyncio.run(solver1.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route1)
    score1 = get_multi_minmax_tour_distance(test_synapse)

    solver2 = NearestNeighbourMultiSolver2(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route2)
    score2 = get_multi_minmax_tour_distance(test_synapse)

    # print(f"{lkh_solver.__class__.__name__} Tour: {route}")
    # print(f"{lkh_solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
    print(f"LKH scored: {score1} while Multi scored: {score2}")