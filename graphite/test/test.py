import numpy as np
from typing import List, Tuple, Dict
import random


class MemeticMTSP:
    def __init__(self,
                 distances: np.ndarray,
                 num_salesmen: int,
                 depots: List[int] = None,
                 population_size: int = 30,
                 offspring_size: int = 1,
                 max_iterations: int = 4000):
        """
        Initialize the Memetic Algorithm solver

        Args:
            distances: Distance matrix between cities
            num_salesmen: Number of salesmen
            depots: List of depot indices (if None, assumes single depot at index 0)
            population_size: Size of population (μ)
            offspring_size: Number of offspring per generation (γ)
            max_iterations: Maximum number of iterations
        """
        self.distances = distances
        self.num_cities = len(distances)
        self.num_salesmen = num_salesmen
        self.depots = depots if depots else [0]
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.max_iterations = max_iterations
        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        """Generate initial population of solutions"""
        for _ in range(self.population_size):
            # Create random solution
            solution = self._generate_random_solution()
            # Improve it with VND
            solution = self._variable_neighborhood_descent(solution)
            self.population.append(solution)

            # Update best solution
            fitness = self._calculate_fitness(solution)
            if fitness < self.best_fitness:
                self.best_solution = solution.copy()
                self.best_fitness = fitness

    def _generate_random_solution(self) -> List[List[int]]:
        """Generate random initial solution"""
        unassigned = list(range(len(self.distances)))
        # Remove depots from unassigned cities
        for depot in self.depots:
            unassigned.remove(depot)

        solution = [[] for _ in range(self.num_salesmen)]

        # Assign each salesman to a depot
        for i, depot in enumerate(self.depots):
            solution[i % self.num_salesmen].append(depot)

        # Randomly assign remaining cities
        while unassigned:
            # Find shortest tour
            tour_lengths = [self._calculate_tour_length(tour) for tour in solution]
            shortest_tour_idx = tour_lengths.index(min(tour_lengths))

            # Pick random unassigned city
            city = random.choice(unassigned)
            unassigned.remove(city)

            # Add to shortest tour
            solution[shortest_tour_idx].append(city)

        # Close tours by returning to depots
        for i, tour in enumerate(solution):
            if tour[0] != tour[-1]:
                tour.append(tour[0])

        return solution

    def _calculate_tour_length(self, tour: List[int]) -> float:
        """Calculate length of a single tour"""
        if len(tour) < 2:
            return 0

        length = 0
        for i in range(len(tour) - 1):
            length += self.distances[tour[i]][tour[i + 1]]
        return length

    def _calculate_fitness(self, solution: List[List[int]]) -> float:
        """Calculate fitness (longest tour length) of a solution"""
        tour_lengths = [self._calculate_tour_length(tour) for tour in solution]
        return max(tour_lengths)

    def _variable_neighborhood_descent(self, solution: List[List[int]]) -> List[List[int]]:
        """Improve solution using Variable Neighborhood Descent"""
        # Implementation of VND with neighborhoods:
        # N1: Insert - remove city from longest tour and insert in another tour
        # N2: Swap - swap two cities between tours
        # N3: 2-opt* - special 2-opt move between tours
        neighborhoods = [self._insert_move, self._swap_move, self._two_opt_star]

        improved = True
        while improved:
            improved = False
            for neighborhood in neighborhoods:
                new_solution = neighborhood(solution)
                if self._calculate_fitness(new_solution) < self._calculate_fitness(solution):
                    solution = new_solution
                    improved = True
                    break

        return solution

    def _insert_move(self, solution: List[List[int]]) -> List[List[int]]:
        """Insert move: remove city from longest tour and insert in another tour"""
        new_solution = [tour.copy() for tour in solution]
        longest_tour_idx = max(range(len(solution)),
                               key=lambda i: self._calculate_tour_length(solution[i]))

        # Try each city in longest tour
        longest_tour = new_solution[longest_tour_idx]
        for i in range(1, len(longest_tour) - 1):  # Skip depot
            city = longest_tour[i]

            # Try inserting in each position of other tours
            best_pos = None
            best_tour_idx = None
            best_improvement = 0

            for tour_idx, tour in enumerate(new_solution):
                if tour_idx == longest_tour_idx:
                    continue

                for pos in range(1, len(tour)):  # Skip inserting at depot
                    # Calculate improvement
                    old_length = self._calculate_tour_length(longest_tour)
                    new_longest = longest_tour[:i] + longest_tour[i + 1:]
                    new_length = self._calculate_tour_length(new_longest)

                    tour_before = tour[:]
                    tour_after = tour[:pos] + [city] + tour[pos:]
                    old_other = self._calculate_tour_length(tour_before)
                    new_other = self._calculate_tour_length(tour_after)

                    improvement = max(old_length, old_other) - max(new_length, new_other)

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_pos = pos
                        best_tour_idx = tour_idx

            # Make best move if improvement found
            if best_improvement > 0:
                receiving_tour = new_solution[best_tour_idx]
                new_solution[longest_tour_idx] = longest_tour[:i] + longest_tour[i + 1:]
                new_solution[best_tour_idx] = (receiving_tour[:best_pos] +
                                               [city] +
                                               receiving_tour[best_pos:])
                return new_solution

        return new_solution

    def _swap_move(self, solution: List[List[int]]) -> List[List[int]]:
        """Swap move: swap cities between tours"""
        new_solution = [tour.copy() for tour in solution]
        longest_tour_idx = max(range(len(solution)),
                               key=lambda i: self._calculate_tour_length(solution[i]))

        # Try swapping each city in longest tour
        longest_tour = new_solution[longest_tour_idx]
        for i in range(1, len(longest_tour) - 1):  # Skip depot
            city1 = longest_tour[i]

            # Try swapping with each city in other tours
            for tour_idx, tour in enumerate(new_solution):
                if tour_idx == longest_tour_idx:
                    continue

                for j in range(1, len(tour) - 1):  # Skip depot
                    city2 = tour[j]

                    # Calculate improvement
                    old_length1 = self._calculate_tour_length(longest_tour)
                    old_length2 = self._calculate_tour_length(tour)

                    # Make swap
                    longest_tour[i] = city2
                    tour[j] = city1

                    new_length1 = self._calculate_tour_length(longest_tour)
                    new_length2 = self._calculate_tour_length(tour)

                    if max(new_length1, new_length2) < max(old_length1, old_length2):
                        return new_solution

                    # Undo swap
                    longest_tour[i] = city1
                    tour[j] = city2

        return new_solution

    def _two_opt_star(self, solution: List[List[int]]) -> List[List[int]]:
        """2-opt* move: exchange path segments between tours"""
        new_solution = [tour.copy() for tour in solution]
        longest_tour_idx = max(range(len(solution)),
                               key=lambda i: self._calculate_tour_length(solution[i]))

        longest_tour = new_solution[longest_tour_idx]

        # Try each pair of edges in longest tour
        for i in range(len(longest_tour) - 1):
            # Skip if involves depot
            if longest_tour[i] in self.depots or longest_tour[i + 1] in self.depots:
                continue

            # Try exchanging with each edge in other tours
            for tour_idx, tour in enumerate(new_solution):
                if tour_idx == longest_tour_idx:
                    continue

                for j in range(len(tour) - 1):
                    # Skip if involves depot
                    if tour[j] in self.depots or tour[j + 1] in self.depots:
                        continue

                    # Calculate improvement
                    old_length1 = self._calculate_tour_length(longest_tour)
                    old_length2 = self._calculate_tour_length(tour)

                    # Make exchange
                    new_tour1 = (longest_tour[:i + 1] +
                                 tour[j + 1:])
                    new_tour2 = (tour[:j + 1] +
                                 longest_tour[i + 1:])

                    new_length1 = self._calculate_tour_length(new_tour1)
                    new_length2 = self._calculate_tour_length(new_tour2)

                    if max(new_length1, new_length2) < max(old_length1, old_length2):
                        new_solution[longest_tour_idx] = new_tour1
                        new_solution[tour_idx] = new_tour2
                        return new_solution

        return new_solution

    def _edge_assembly_crossover(self, parent1: List[List[int]],
                                 parent2: List[List[int]]) -> List[List[int]]:
        """Edge Assembly Crossover operator"""
        # Create union of edges from both parents
        edges = set()
        for tour in parent1:
            for i in range(len(tour) - 1):
                edges.add((min(tour[i], tour[i + 1]), max(tour[i], tour[i + 1])))
        for tour in parent2:
            for i in range(len(tour) - 1):
                edges.add((min(tour[i], tour[i + 1]), max(tour[i], tour[i + 1])))

        # Create AB-cycles
        cycles = []
        remaining_edges = edges.copy()

        while remaining_edges:
            cycle = []
            edge = remaining_edges.pop()
            cycle.append(edge)
            current = edge[1]

            while True:
                # Find next edge
                next_edge = None
                for e in remaining_edges:
                    if current in e:
                        next_edge = e
                        break

                if not next_edge:
                    break

                remaining_edges.remove(next_edge)
                cycle.append(next_edge)
                current = next_edge[0] if next_edge[1] == current else next_edge[1]

                if current == edge[0]:
                    cycles.append(cycle)
                    break

        # Construct offspring using cycles
        offspring = [[] for _ in range(self.num_salesmen)]

        # Start with depots
        for i, depot in enumerate(self.depots):
            offspring[i % self.num_salesmen].append(depot)

        # Add cycles
        for cycle in cycles:
            # Find shortest tour
            tour_lengths = [self._calculate_tour_length(tour) for tour in offspring]
            shortest_tour_idx = tour_lengths.index(min(tour_lengths))

            # Add cycle to tour
            tour = offspring[shortest_tour_idx]
            for edge in cycle:
                if edge[0] not in tour:
                    tour.append(edge[0])
                if edge[1] not in tour:
                    tour.append(edge[1])

        # Close tours
        for tour in offspring:
            if tour[0] != tour[-1]:
                tour.append(tour[0])

        return offspring

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Solve the minmax mTSP using the memetic algorithm

        Returns:
            Tuple of (best solution, best fitness)
        """
        # Initialize population
        self.initialize_population()

        iteration = 0
        while iteration < self.max_iterations:
            # Select parents
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)

            # Generate offspring
            for _ in range(self.offspring_size):
                # Create offspring using edge assembly crossover
                offspring = self._edge_assembly_crossover(parent1, parent2)

                # Improve offspring using VND
                offspring = self._variable_neighborhood_descent(offspring)

                # Update best solution if improved
                fitness = self._calculate_fitness(offspring)
                if fitness < self.best_fitness:
                    self.best_solution = offspring.copy()
                    self.best_fitness = fitness

                # Add to population
                if len(self.population) >= self.population_size:
                    # Remove worst solution
                    worst_idx = max(range(len(self.population)),
                                    key=lambda i: self._calculate_fitness(self.population[i]))
                    self.population.pop(worst_idx)
                self.population.append(offspring)

            iteration += 1

        return self.best_solution, self.best_fitness


# Example usage
if __name__ == "__main__":
    # Example problem with 5 cities and 2 salesmen
    distances = np.array([
        [0, 2, 4, 3, 1],
        [2, 0, 3, 5, 2],
        [4, 3, 0, 1, 5],
        [3, 5, 1, 0, 4],
        [1, 2, 5, 4, 0]
    ])

    # Single depot case
    solver = MemeticMTSP(distances=distances,
                         num_salesmen=2)
    solution, fitness = solver.solve()
    print("Single depot solution:", solution)
    print("Objective value:", fitness)

    # Multi-depot case
    solver = MemeticMTSP(distances=distances,
                         num_salesmen=2,
                         depots=[0, 1])
    solution, fitness = solver.solve()
    print("\nMulti-depot solution:", solution)
    print("Objective value:", fitness)
