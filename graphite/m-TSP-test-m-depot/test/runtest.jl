# test/test_simple.jl

using HGSmTSP
using Test
using Random
using LinearAlgebra

"""
Basic test function for the mTSP solver
"""
function test_simple()
    n_nodes = 10  # Total nodes including depots
    n_vehicles = 3
    depots = [1, 5, 8]  # Multiple depot indices

    # Set random seed for reproducibility
    Random.seed!(42)

    # Create random coordinates
    coordinates = rand(n_nodes, 2) .* 1000

    # Calculate distance matrix
    dist_mtx = zeros(n_nodes, n_nodes)
    for i in 1:n_nodes
        for j in 1:n_nodes
            dist_mtx[i, j] = sqrt(sum((coordinates[i, :] - coordinates[j, :]).^2))
        end
    end

    # Solve
    println("Solving simple case...")
    routes, costs = solve_mTSP(
        n_vehicles, 
        dist_mtx, 
        coordinates, 
        depots;
        n_iterations=100,
        time_limit=10.0,
        verbose=true
    )

    # Validate solution
    @test length(routes) == n_vehicles
    @test all(route -> !isempty(route), routes)
    
    # Print results
    println("\nResults:")
    println("Number of routes: ", length(routes))
    for (i, route) in enumerate(routes)
        println("Route $i: $(route) (cost: $(round(costs[i], digits=2)))")
    end

    return routes, costs, coordinates, depots
end

"""
Validate a solution for the mTSP
"""
function validate_solution(routes::Vector{Vector{Int}}, costs::Vector{Float64}, 
    coordinates::Matrix{Float64}, depots::Vector{Int}, n_vehicles::Int)
    
    # Check number of routes
    @test length(routes) == n_vehicles "Number of routes doesn't match number of vehicles"

    # Check all customers are visited exactly once
    all_nodes = Set{Int}()
    for route in routes
        for node in route
            @test !(node in depots) "Depot $node found in route sequence"
            @test !(node in all_nodes) "Node $node visited multiple times"
            push!(all_nodes, node)
        end
    end

    # Check each route starts and ends near a depot
    for route in routes
        if !isempty(route)
            start_dist = minimum(d -> norm(coordinates[d,:] - coordinates[route[1],:]), depots)
            end_dist = minimum(d -> norm(coordinates[d,:] - coordinates[route[end],:]), depots)
            @test start_dist < 1e-10 "Route doesn't start from a depot"
            @test end_dist < 1e-10 "Route doesn't end at a depot"
        end
    end

    println("✅ Solution is valid!")
    return true
end

"""
Run all tests
"""
function run_tests()
    @testset "mTSP Tests" begin
        routes, costs, coordinates, depots = test_simple()
        validate_solution(routes, costs, coordinates, depots, length(routes))
    end
end

# Run tests when this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end