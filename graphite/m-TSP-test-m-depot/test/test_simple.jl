using HGSmTSP

# Create a simple test instance
function test_simple()
    n_nodes = 10  # Total nodes including depots
    n_vehicles = 3
    depots = [1, 5, 8]  # Multiple depot indices

    # Create random coordinates
    coordinates = rand(n_nodes, 2) .* 10

    #convert coordinates to Float64


    # Calculate distance matrix
    dist_mtx = zeros(n_nodes, n_nodes)
    for i in 1:n_nodes
        for j in 1:n_nodes
            dist_mtx[i, j] = sqrt(sum((coordinates[i, :] - coordinates[j, :]).^2))
        end
    end

    # Solve
    println("Solving simple case...")
    @time routes, costs = solve_mTSP(
        n_vehicles , 
        dist_mtx , 
        coordinates, 
        depots;
        n_iterations=100,
        time_limit=10.0,
        verbose=true
    )

    # Print results
    println("\nResults:")
    println("Number of routes: ", length(routes))
    for (i, route) in enumerate(routes)
        println("Route $i: $(route) (cost: $(round(costs[i], digits=2)))")
    end

    return routes, costs
end

@time test_simple()