from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.utils.graph_utils import timeout
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
    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.lkh_path = "./LKH-3.0.11/LKH"
    
    def create_problem_file(self, distance_matrix):
        dimension = len(distance_matrix)
        problem_file_content = f"""NAME: TSP
        TYPE: TSP
        DIMENSION: {dimension}
        EDGE_WEIGHT_TYPE: EXPLICIT
        EDGE_WEIGHT_FORMAT: FULL_MATRIX
        EDGE_WEIGHT_SECTION
        """
        # Sử dụng StringIO và np.savetxt để tạo chuỗi
        buffer = StringIO()
        np.savetxt(buffer, distance_matrix, fmt='%d', delimiter=' ')
        matrix_string = buffer.getvalue().strip()
        problem_file_content += matrix_string + "\nEOF\n"
        return problem_file_content
    
    def create_parameter_file(self, problem_file_path, tour_file_path, nodes=5000):
        trial = 0
        parameter_file_content = f"""PROBLEM_FILE = {problem_file_path}
        TOUR_FILE = {tour_file_path}
        PRECISION = 1e-04
        RUNS = 1
        INITIAL_TOUR_ALGORITHM = GREEDY
        KICK_TYPE = 4
        KICKS = 10
        MAX_TRIALS = {trial}   
        TIME_LIMIT = 20
        TOTAL_TIME_LIMIT = 20
        """

        return parameter_file_content
    
    async def solve(self, formatted_problem, future_id:int)->List[int]:
        with tempfile.NamedTemporaryFile('w+', prefix='problem_', suffix='.txt', delete=False) as problem_file, \
            tempfile.NamedTemporaryFile('w+', prefix='param_', suffix='.txt', delete=False) as parameter_file, \
            tempfile.NamedTemporaryFile('r+', prefix='tour_', suffix='.txt', delete=False) as tour_file:

            problem_file_content = self.create_problem_file(formatted_problem)
            problem_file.write(problem_file_content)
            problem_file.flush()

            parameter_file_content = self.create_parameter_file(problem_file.name, tour_file.name, len(formatted_problem))
            parameter_file.write(parameter_file_content)
            parameter_file.flush()

            # Run LKH
            subprocess.run([self.lkh_path, parameter_file.name], check=True)
            # process = await asyncio.create_subprocess_exec(
            #     self.lkh_path, parameter_file.name,
            #     stdout=subprocess.PIPE, stderr=subprocess.PIPE
            # )
            # stdout, stderr = await process.communicate()

            # if process.returncode != 0:
            #     raise RuntimeError(f"LKH failed with error: {stderr.decode()}")

            # Read the tour file
            tour_file.seek(0)
            tour = self.parse_tour_file(tour_file.name)

        # Clean up temporary files
        os.remove(problem_file.name)
        os.remove(parameter_file.name)
        os.remove(tour_file.name)

        # total_distance = self.calculate_total_distance(tour, formatted_problem)

        # return tour
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

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges
    
if __name__ == "__main__":
    ## Test case for GraphV2Problem
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
    loaded_datasets = {}
    with np.load('dataset/Asia_MSB.npz') as f:
        loaded_datasets["Asia_MSB"] = np.array(f['data'])

    def recreate_edges(problem: GraphV2Problem):
        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            return geom_edges(node_coords)
        elif problem.cost_function == "Euclidean2D":
            return euc_2d_edges(node_coords)
        elif problem.cost_function == "Manhatten2D":
            return man_2d_edges(node_coords)
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
      
    n_nodes = 5000
    # randomly select n_nodes indexes from the selected graph
    selected_node_idxs = random.sample(range(26000000), n_nodes)
    test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB")
    
    if isinstance(test_problem, GraphV2Problem):
        test_problem.edges = recreate_edges(test_problem)
    
    print("edges", test_problem.edges)
    print("Problem", test_problem)

    lkh_solver = LKHSolver(problem_types=[test_problem])
    start_time = time.time()

    # Run the solver to get the tour
    route = asyncio.run(lkh_solver.solve_problem(test_problem))

    # Calculate total distance of the tour
    total_distance = lkh_solver.calculate_total_distance(route, test_problem.edges)

    print(f"{lkh_solver.__class__.__name__} Tour: {route}")
    print(f"Total distance of the tour: {total_distance}")
    print(f"{lkh_solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")
