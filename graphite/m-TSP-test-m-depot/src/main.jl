using Clustering
using Distances
using Random
using StatsBase
using TSPLIB
using TSPSolvers
using LinearAlgebra

include("split.jl")
include("genetic_algorithm.jl")
include("initial.jl")
include("mutation.jl")
include("crossover.jl")
include("neighborhood.jl")
include("neighborhood_intra.jl")
include("costs.jl")
include("intersection.jl")
include("enrichment.jl")

"""
    solve_mTSP(n_vehicles, dist_mtx, coordinates, depot_indices; kwargs...)

Solves the Multiple Traveling Salesman Problem with multiple depots.
"""
function solve_mTSP(
    n_vehicles::Int,
    dist_mtx::Matrix{Float64},
    coordinates::Matrix{Float64},
    depot_indices::Vector{Int64};  # Default to first node as depot
    n_runs::Int=1,
    n_iterations::Int=100,
    time_limit::Float64=10.0,
    W::Int=1000,
    h::Float64=0.3,
    popsize::Tuple{Int,Int}=(10, 20),
    k_tournament::Int=2,
    mutation_chance::Float64=0.0,
    num_nei::Int=2,
    verbose::Bool=false
)::Tuple{Vector{Vector{Int}},Vector{Float64}}

    if n_vehicles == 1
        dist_mtx_int = round.(Int, dist_mtx)
        tour, tour_len = TSPSolvers.solve_tsp(dist_mtx_int; 
            algorithm="HGS", nbIter=n_iterations, timeLimit=time_limit)
        return [tour], [Float64(tour_len)]
    end

    if length(depot_indices) < 1
        error("At least one depot must be specified")
    end

    t0 = time()
    n_nodes = size(dist_mtx)[1]
    
    # Extract depot and customer coordinates
    depot_coordinates = coordinates[depot_indices, :]
    customer_indices = setdiff(1:n_nodes, depot_indices)
    customer_coordinates = coordinates[customer_indices, :]

    # Create expanded distance matrix with dummy depot
    n_total = n_nodes + 1
    dist_mtx_with_dummy = zeros(Float64, n_total, n_total)
    
    # Copy original distances
    dist_mtx_with_dummy[1:n_nodes, 1:n_nodes] = dist_mtx
    
    # Add dummy depot connections
    for i in 1:n_nodes
        if i in depot_indices
            dist_mtx_with_dummy[i, n_total] = 0.0
            dist_mtx_with_dummy[n_total, i] = 0.0
        else
            dist_mtx_with_dummy[i, n_total] = Inf
            dist_mtx_with_dummy[n_total, i] = Inf
        end
    end
    dist_mtx_with_dummy[n_total, n_total] = 0.0

    best = Inf
    worst = 0.0
    crossover_functions = Int[2, 3]
    avg = 0.0

    best_chrm = Chromosome(Int[], 0.0, 0.0, Tour[])
    all_chrms = Chromosome[]

    for _ in 1:n_runs
        time_limit_for_this_run = time_limit - (time() - t0)
        P, roullet = perform_genetic_algorithm(
            dist_mtx_with_dummy,
            n_vehicles,
            depot_indices,
            h,
            popsize,
            k_tournament,
            n_iterations,
            time_limit_for_this_run,
            mutation_chance,
            num_nei,
            crossover_functions,
            customer_coordinates,
            depot_coordinates,
            verbose=verbose
        )

        avg += P[1].fitness
        push!(all_chrms, P[1])
        if P[1].fitness < best
            best = P[1].fitness
            best_chrm = P[1]
        end
        if P[1].fitness > worst
            worst = P[1].fitness
        end
    end

    hgs_routes = [t.sequence for t in best_chrm.tours]
    hgs_route_lengths = [t.cost for t in best_chrm.tours]

    return hgs_routes, hgs_route_lengths
end