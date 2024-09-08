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

from typing import List, Dict
from graphite.solvers import *
from graphite.dataset.dataset_generator import MetricTSPGenerator, GeneralTSPGenerator
from graphite.protocol import GraphProblem, GraphSynapse
from graphite.utils.graph_utils import get_tour_distance
import pandas as pd
import tqdm
import time
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT_DIR = "tests"
SAVE_DIR = "evaluation_results"
N_PROBLEMS = 1000

def can_show_plot():
    # Check if running in a headless environment
    if os.name == 'posix':
        display = os.getenv('DISPLAY')
        if not display:
            return False

    # Check if the backend is suitable for interactive plotting
    backend = matplotlib.get_backend()
    if backend in ['agg', 'cairo', 'svg', 'pdf', 'ps']:
        return False

    return True

def compare_problems(solvers: List[BaseSolver], problems: List[GraphProblem]) -> Dict[str, List[float]]:
    problem_types = set([problem.problem_type for problem in problems])
    mock_synapses = [GraphSynapse(problem=problem) for problem in problems]
    results = {solver.__class__.__name__: [] for solver in solvers}
    run_times_dict = {solver.__class__.__name__: [] for solver in solvers}
    scores_dict = {solver.__class__.__name__: [] for solver in solvers}

    for i, solver in enumerate(solvers):
        total_run_time = 0
        scores = []
        print(f"Running Solver {i+1} - {solver.__class__.__name__}")
        
        for mock_synapse in tqdm.tqdm(mock_synapses, desc=f"{solver.__class__.__name__} solving {problem_types}"):
            # Start the timer and run the solver
            start_time = time.perf_counter()
            mock_synapse.solution = asyncio.run(solver.solve_problem(mock_synapse.problem))
            run_time = time.perf_counter() - start_time
            
            # Add the run time to the total run time
            total_run_time += run_time

            # Tính tổng độ dài hành trình của solution và thêm vào danh sách scores
            path_length = get_tour_distance(mock_synapse)
            scores.append(path_length)
        
        # Lưu thời gian chạy và điểm số vào dictionary
        run_times_dict[solver.__class__.__name__] = [total_run_time]
        scores_dict[solver.__class__.__name__] = scores
        
        # In ra tổng thời gian hoàn thành và tổng độ dài hành trình
        total_score = sum(scores)
        print(f"Solver {solver.__class__.__name__}: Total path length = {total_score:.2f}, Total Time Taken = {total_run_time:.2f} seconds")

    return run_times_dict, scores_dict


def compute_relative_scores(scores_df: pd.DataFrame, tolerance=1e-5):
    relative_scores = pd.DataFrame(index=scores_df.index)
    solvers = scores_df.columns.difference(['problem_size'])
    
    for solver in solvers:
        relative_scores[solver] = scores_df[solvers].apply(
            lambda row: sum(
                np.isclose(row[solver], row[other_solver], rtol=tolerance) or row[solver] < row[other_solver]
                for other_solver in solvers
            ), axis=1)
    
    # Normalize the relative scores to the range 0 to 1
    normalized_relative_scores = (relative_scores - relative_scores.min()) / (relative_scores.max() - relative_scores.min())
    return normalized_relative_scores

def main():
    if not os.path.exists(os.path.join(ROOT_DIR, SAVE_DIR)):
        os.makedirs(os.path.join(ROOT_DIR, SAVE_DIR))

    # Use MetricTSPGenerator to generate problems of various graph sizes
    metric_problems, metric_sizes = GeneralTSPGenerator.generate_n_samples(N_PROBLEMS)
    # NearestNeighbourSolver(),  HPNSolver()
    test_solvers = [BeamSearchSolver(), NearestNeighbourSolver(), LKHGeneticSolver()]

    run_times_dict, scores_dict = compare_problems(test_solvers, metric_problems)

    # Create DataFrames for run times and scores
    run_times_df = pd.DataFrame(run_times_dict)
    scores_df = pd.DataFrame(scores_dict)

    # Add the problem size classification
    run_times_df['problem_size'] = metric_sizes
    scores_df['problem_size'] = metric_sizes

    # Set the problem index
    run_times_df.index.name = 'problem_index'
    scores_df.index.name = 'problem_index'

    # Compute and normalize relative scores as the score of the solver compared to the best and the worst solver of the cohort
    relative_scores_df = compute_relative_scores(scores_df)

    # Save the data
    run_times_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "solver_run_times.csv"))
    scores_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "solver_scores.csv"))
    relative_scores_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "solver_relative_scores.csv"))

    # Compute the average relative scores grouped by problem size
    average_relative_scores = relative_scores_df.groupby(scores_df['problem_size']).mean()

    # Plotting the average relative scores
    average_relative_scores.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Relative Score of Each Solver by Problem Size')
    plt.xlabel('Problem Size')
    plt.ylabel('Average Relative Score (Normalized)')
    plt.legend(title='Solver', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(ROOT_DIR, SAVE_DIR), "relative_score.png"))
    if can_show_plot():
        plt.show()
    else:
        plt.close()

if __name__=="__main__":
    main()
