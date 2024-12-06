module HGSmTSP

include("main.jl")

export solve_mTSP

# for precompilation of the module
function __init__()
    # n_nodes, n_vehicles = 10, 2
    # coordinates = rand(n_nodes, 2) .* 1000
    # dist_mtx = zeros(n_nodes, n_nodes)
    # for i in 1:n_nodes
    #     for j in 1:n_nodes
    #         dist_mtx[i, j] = norm(coordinates[i, :] - coordinates[j, :])
    #     end
    # end
    
    # # Example with single depot
    # solve_mTSP(n_vehicles, dist_mtx, coordinates)
    
    # # Example with multiple depots
    # depot_indices = [1, 5]  # Example depots
    # solve_mTSP(n_vehicles, dist_mtx, coordinates, depot_indices)
end

end