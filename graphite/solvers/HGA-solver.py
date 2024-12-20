import asyncio
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from graphite.data.dataset_utils import load_default_dataset
from graphite.protocol import GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver_multi_2 import NearestNeighbourMultiSolver2
from graphite.utils.graph_utils import get_multi_minmax_tour_distance


@dataclass
class ClusterData:
    depot_index: int
    n_salesmen: int
    distance_matrix: np.ndarray
    coordinates: np.ndarray
    distance_type: str = "Euclidean2D"


@dataclass
class ClusterResult:
    routes: List[List[int]]
    lengths: List[float]

class HGASolver(BaseSolver):
    def __init__(self, problem_types: List[GraphV2Problem] = [GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)
        self.HGA_path = Path("/home/lampham/PycharmProjects/Graphite-Subnet-v2/graphite/m-TSP-julia/test/solve_mtsp.jl")

    async def solve(self, formatted_problem) -> List[List[int]]:
        routes, lengths = self.solve_mtsp(formatted_problem.n_salesmen, formatted_problem.edges, node_coords, formatted_problem.cost_function)
        return routes

    async def solve_cluster(self, cluster_data: ClusterData, julia_path: str) -> ClusterResult:
        """
        Solve a single cluster using Julia HGA solver
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.json"
            output_path = Path(temp_dir) / "output.json"
            print('input_path', input_path)
            print('output_path', output_path)

            # Prepare cluster input
            cluster_input = {
                "n_vehicles": cluster_data.n_salesmen,
                "dist_mtx": cluster_data.distance_matrix.tolist(),
                "coordinates": cluster_data.coordinates.tolist(),
                "distance_type": cluster_data.distance_type
            }

            # Write input file
            with open(input_path, 'w') as f:
                json.dump(cluster_input, f)

            cmd = [
                "julia",
                str(julia_path),
                "--input", str(input_path),
                "--output", str(output_path)
            ]

            import subprocess
            try:
                # Run process and capture output in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,  # This makes output strings instead of bytes
                    bufsize=1  # Line buffered
                )

                # Print output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print("Julia:", output.strip())

                # Get any remaining stderr
                stderr = process.stderr.read()
                if stderr:
                    print("Julia stderr:", stderr)

                # Check return code
                retcode = process.poll()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, cmd)

                # Read results
                with open(output_path, 'r') as f:
                    results = json.load(f)
                print("solve_cluster done")
                return ClusterResult(
                    routes=results["routes"],
                    lengths=results["lengths"]
                )

            except subprocess.CalledProcessError as e:
                print(f"Julia process failed with return code {e.returncode}")
                print(f"Error output: {e.stderr}")
                raise
            except FileNotFoundError:
                raise RuntimeError("Julia solver failed to produce output file")
            except json.JSONDecodeError:
                raise RuntimeError("Invalid JSON output from Julia solver")

    def solve_mtsp(self, n_vehicles, dist_mtx, node_coords, cost_function):
        # Prepare input data
        cluster_data = ClusterData(
            depot_index=0,
            n_salesmen=n_vehicles,
            distance_matrix=np.array(dist_mtx),
            coordinates=np.array(node_coords),
            distance_type=cost_function
        )

        # Solve the cluster
        results = asyncio.run(self.solve_cluster(cluster_data, self.HGA_path))

        if not isinstance(results, ClusterResult):
            raise TypeError("Expected ClusterResult from solve_cluster")

        tours = results.routes
        closed_tours = [[0] + tour + [0] for tour in tours]
        return closed_tours, results.lengths

    def problem_transformations(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
        return problem


import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges

    total_start_time = time.time()


    class Mock:
        def __init__(self) -> None:
            pass

        def recreate_edges(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
            edge_start_time = time.time()
            node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
            node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])

            if problem.cost_function == "Geom":
                edges = geom_edges(node_coords)
            elif problem.cost_function == "Euclidean2D":
                edges = euc_2d_edges(node_coords)
            elif problem.cost_function == "Manhatten2D":
                edges = man_2d_edges(node_coords)
            else:
                return "Only Geom, Euclidean2D, and Manhatten2D supported for now."

            logger.info(f"Edge recreation took: {time.time() - edge_start_time:.2f} seconds")
            return edges, node_coords


    # Initialize mock and load data
    init_start_time = time.time()
    mock = Mock()
    load_default_dataset(mock)
    logger.info(f"Initialization and data loading took: {time.time() - init_start_time:.2f} seconds")

    # Problem setup
    setup_start_time = time.time()
    n_nodes = 2000
    m = 10
    test_problem = GraphV2ProblemMulti(
        n_nodes=n_nodes,
        selected_ids=random.sample(list(range(100000)), n_nodes),
        dataset_ref="Asia_MSB",
        n_salesmen=m,
        depots=[0] * m,
        cost_function="Euclidean2D"
    )
    test_problem.edges, node_coords = mock.recreate_edges(test_problem)
    logger.info(f"Problem setup took:   {time.time() - setup_start_time:.2f} seconds")

    # HGA Solver
    hga_start_time = time.time()
    hga_solver = HGASolver(problem_types=[test_problem])
    route = asyncio.run(hga_solver.solve(test_problem))
    test_synapse = GraphV2Synapse(problem=test_problem, solution=route)
    score1 = get_multi_minmax_tour_distance(test_synapse)
    logger.info(f"HGA Solver took: {time.time() - hga_start_time:.2f} seconds")

    # Nearest Neighbour Solver
    nn_start_time = time.time()
    solver2 = NearestNeighbourMultiSolver2(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse2 = GraphV2Synapse(problem=test_problem, solution=route2)
    score2 = get_multi_minmax_tour_distance(test_synapse2)
    logger.info(f"Nearest Neighbour Solver took: {time.time() - nn_start_time:.2f} seconds")

    # Final results
    logger.info(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
    print(f"hga scored: {score1} while Multi2 scored: {score2}")