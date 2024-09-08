# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
from graphite.utils.graph_utils import timeout
import asyncio
import time
import heapq

class BeamSearchSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    def heuristic_estimate(self, path, distance_matrix):
        if not isinstance(path, list):
            raise TypeError("path should be a list")
        last_node = path[-1]
        remaining_nodes = [i for i in range(len(distance_matrix)) if i not in path]
        if not remaining_nodes:
            return 0
        min_distance = min(distance_matrix[last_node][i] for i in remaining_nodes)
        return min_distance

    async def solve(self, formatted_problem, future_id: int, beam_width: int = 3) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix)

        beam = [(0, [0], 0)]  # (current_node, path, current_distance)
        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return None
            candidates = []

            for current_node, path, current_distance in beam:
                for next_node in range(n):
                    if next_node not in path:
                        new_path = path + [next_node]
                        new_distance = current_distance + distance_matrix[current_node][next_node]
                        heuristic = self.heuristic_estimate(new_path, distance_matrix)
                        candidates.append((new_distance + heuristic, next_node, new_path))

            candidates = heapq.nsmallest(beam_width, candidates)
            beam = [(dist, path, path) for dist, _, path in candidates]

        final_candidates = []
        for _, path, current_distance in beam:
            final_distance = current_distance + distance_matrix[path[-1]][0]
            final_candidates.append((path + [0], final_distance))

        best_path, best_distance = min(final_candidates, key=lambda x: x[1])

        return best_path

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == '__main__':
    # Run the solver on a test MetricTSP
    n_nodes = 100
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = BeamSearchSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve(test_problem.edges, future_id=1))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time() - start_time}")

