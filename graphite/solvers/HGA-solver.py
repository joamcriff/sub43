import asyncio
import json
import random
import subprocess
import time
from typing import List, Union
import numpy as np
from graphite.data.dataset_utils import load_default_dataset
from graphite.protocol import GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver_multi_2 import NearestNeighbourMultiSolver2
from graphite.utils.graph_utils import get_multi_minmax_tour_distance

class HGASolver(BaseSolver):
    def __init__(self, problem_types: List[GraphV2Problem] = [GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)
        self.HGA_path = "/home/lampham/julia-tsp/m-TSP/test/solve_mtsp.jl"

    def prepare_mtsp_input(n_vehicles, dist_mtx):
        # Prepare your data
        data = {
            "n_vehicles": n_vehicles,
            "dist_mtx": dist_mtx.tolist()
        }
        # Write to file
        with open('/home/lampham/PycharmProjects/sub43/graphite/output/mtsp_input.json', 'w') as f:
            json.dump(data, f)

    async def solve(self, formatted_problem) -> List[List[int]]:
        routes, lengths = self.solve_mtsp(formatted_problem.n_salesmen, formatted_problem.edges, node_coords)
        return routes

    def solve_mtsp(self,n_vehicles, dist_mtx, node_coords):
        # Save input data
        input_data = {
            "n_vehicles": n_vehicles,
            "dist_mtx": dist_mtx.tolist(),
            "coordinates": node_coords.tolist()
        }
        with open('/home/lampham/PycharmProjects/sub43/graphite/input/input.json', 'w') as f:
            json.dump(input_data, f)

        # Call Julia script
        subprocess.run(["julia", self.HGA_path])

        # Read results
        with open('/home/lampham/PycharmProjects/sub43/graphite/output/output.json', 'r') as f:
            results = json.load(f)
        tours = results["routes"]
        closed_tours = [[0] + tour + [0] for tour in tours]
        return closed_tours, results["lengths"]

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
                return geom_edges(node_coords),node_coords
            elif problem.cost_function == "Euclidean2D":
                return euc_2d_edges(node_coords),node_coords
            elif problem.cost_function == "Manhatten2D":
                return man_2d_edges(node_coords),node_coords
            else:
                return "Only Geom, Euclidean2D, and Manhatten2D supported for now."

    mock = Mock()
    load_default_dataset(mock)

    n_nodes = 200
    m = 3
    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, selected_ids=random.sample(list(range(100000)), n_nodes),
                                     dataset_ref="Asia_MSB", n_salesmen=m, depots=[0] * m)
    test_problem.edges, node_coords = mock.recreate_edges(test_problem)
    hga_solver = HGASolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(hga_solver.solve(test_problem))
    test_synapse = GraphV2Synapse(problem=test_problem, solution=route)
    score1 = get_multi_minmax_tour_distance(test_synapse)
    solver2 = NearestNeighbourMultiSolver2(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem=test_problem, solution=route2)
    score2 = get_multi_minmax_tour_distance(test_synapse)

    print(f"hga scored: {score1} while Multi2 scored: {score2}")