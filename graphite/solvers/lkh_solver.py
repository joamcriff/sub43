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

class LKHSolver(BaseSolver):
    def __init__(self, problem_types: List[Union[GraphV1Problem, GraphV2Problem]] = [GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.concorde_path = "./concorde_build/TSP/concorde"

    def create_problem_file(self, distance_matrix):
        dimension = len(distance_matrix)
        problem_file_content = f"""NAME: TSP
        TYPE: TSP
        DIMENSION: {dimension}
        EDGE_WEIGHT_TYPE: EXPLICIT
        EDGE_WEIGHT_FORMAT: FULL_MATRIX
        EDGE_WEIGHT_SECTION
        """
        buffer = StringIO()
        np.savetxt(buffer, distance_matrix, fmt='%d', delimiter=' ')
        matrix_string = buffer.getvalue().strip()
        problem_file_content += matrix_string + "\nEOF\n"
        return problem_file_content

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        with tempfile.NamedTemporaryFile('w+', prefix='problem_', suffix='.tsp', delete=False) as problem_file, \
                tempfile.NamedTemporaryFile('r+', prefix='tour_', suffix='.sol', delete=False) as tour_file:

            problem_file_content = self.create_problem_file(formatted_problem)
            problem_file.write(problem_file_content)
            problem_file.flush()

            # Run Concorde
            subprocess.run([self.concorde_path, '-o', tour_file.name, problem_file.name], check=True)

            # Read the tour file
            tour_file.seek(0)
            tour = self.parse_tour_file(tour_file.name)

        # Clean up temporary files
        os.remove(problem_file.name)
        os.remove(tour_file.name)

        return tour

    def parse_tour_file(self, tour_file_path):
        tour = []
        with open(tour_file_path, 'r') as file:
            for line in file:
                if line.strip().isdigit():
                    tour.append(int(line.strip()))
        tour.append(tour[0])
        return tour

    def calculate_total_distance(self, tour, distance_matrix):
        total_distance = 0
        for i in range(len(tour)):
            total_distance += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return total_distance

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.edges

if __name__ == "__main__":
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

    n_nodes = 3000
    selected_node_idxs = random.sample(range(26000000), n_nodes)
    test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB")

    if isinstance(test_problem, GraphV2Problem):
        test_problem.edges = recreate_edges(test_problem)

    print("edges", test_problem.edges)
    print("Problem", test_problem)

    concorde_solver = LKHSolver(problem_types=[test_problem])
    start_time = time.time()

    route = asyncio.run(concorde_solver.solve_problem(test_problem))

    total_distance = concorde_solver.calculate_total_distance(route, test_problem.edges)

    print(f"{concorde_solver.__class__.__name__} Tour: {route}")
    print(f"Total distance of the tour: {total_distance}")
    print(f"{concorde_solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")