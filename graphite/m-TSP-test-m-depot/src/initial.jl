function greedy_insertion_tour(T::Matrix{Float64}, t1::Vector{Int}, depot_indices::Vector{Int})
    tour = Tour(Int[], 0.0)
    if isempty(t1)
        return tour
    end
    
    # Find best starting depot and city combination
    best_cost = Inf
    best_city_index = 0
    best_depot = first(depot_indices)
    
    for depot in depot_indices
        for (i, city) in enumerate(t1)
            if T[depot+1, city+1] < best_cost
                best_cost = T[depot+1, city+1]
                best_city_index = i
                best_depot = depot
            end
        end
    end
    
    push!(tour.sequence, t1[best_city_index])
    tour.cost = best_cost
    tour.start_depot = best_depot
    deleteat!(t1, best_city_index)
    
    # Build rest of tour
    while !isempty(t1)
        current_node = tour.sequence[end]
        best_cost = Inf
        best_city_index = 0
        
        for (i, city) in enumerate(t1)
            if T[current_node+1, city+1] < best_cost
                best_cost = T[current_node+1, city+1]
                best_city_index = i
            end
        end
        
        push!(tour.sequence, t1[best_city_index])
        tour.cost += best_cost
        deleteat!(t1, best_city_index)
    end
    
    # Find best end depot
    best_end_cost = Inf
    best_end_depot = first(depot_indices)
    for depot in depot_indices
        end_cost = T[tour.sequence[end]+1, depot+1]
        if end_cost < best_end_cost
            best_end_cost = end_cost
            best_end_depot = depot
        end
    end
    
    tour.end_depot = best_end_depot
    tour.cost += best_end_cost
    return tour
end

function move_farthest_median(medians, depot_coordinates)
    distances = [minimum([euclidean(median, depot) for depot in eachrow(depot_coordinates)]) 
                for median in medians]
    farthest_median = argmax(distances)
    closest_depot = argmin([euclidean(medians[farthest_median], depot) 
                          for depot in eachrow(depot_coordinates)])
    medians[farthest_median] += 0.1 * (depot_coordinates[closest_depot, :] - 
                                      medians[farthest_median])
    return medians
end

function k_median(customers::Matrix{Float64}, depot_coordinates::Matrix{Float64}, k::Int)
    data = customers
    n = size(data, 1)
    
    # Choose k initial cluster centers (medians)
    medians = [data[i, :] for i in sample(1:n, k, replace=false)]
    
    # Set threshold for difference between max and min distances
    threshold = 0.1
    assignments_ = Int[]
    
    # Run modified k-Median clustering algorithm
    for _ in 1:100
        # Assign each node to nearest median
        assignments_ = [argmin([euclidean(data[p, :], m) for m in medians]) for p in 1:n]
        
        # Update medians
        for j in 1:k
            cluster_points = data[findall(==(j), assignments_), :]
            if isempty(cluster_points)
                medians[j] = depot_coordinates[1, :]  # Use first depot if cluster empty
            else
                medians[j] = vec(median(cluster_points, dims=1))
            end
        end
        
        # Balance clusters relative to depots
        distances = [minimum([euclidean(median, depot) for depot in eachrow(depot_coordinates)]) 
                    for median in medians]
        if maximum(distances) - minimum(distances) > threshold
            medians = move_farthest_median(medians, depot_coordinates)
        end
    end
    return assignments_
end

function initial_kmedian_solution(T::Matrix{Float64}, customers::Matrix{Float64}, 
    depot_coordinates::Matrix{Float64}, K::Int, depot_indices::Vector{Int})
    
    k1, k2 = size(customers)
    customers_ = k2 == 2 ? transpose(customers) : customers
    
    # Get cluster assignments
    assignments_ = if rand() < 0.5
        result = kmeans(customers_, K)
        copy(result.assignments)
    else
        k_median(customers, depot_coordinates, K)
    end
    
    # Create tours
    tours = Vector{Tour}(undef, K)
    genes = Int[]
    obj = 0.0
    
    for i in 1:K
        t1 = findall(==(i), assignments_)
        if length(t1) == 1
            # Single city tour
            cost = minimum(d -> T[d+1, t1[1]+1] + T[t1[1]+1, d+1], depot_indices)
            tours[i] = Tour(t1, cost)
            push!(genes, t1[1])
            obj = max(obj, cost)
        else
            # Multi-city tour
            tour = greedy_insertion_tour(T, t1, depot_indices)
            tours[i] = tour
            append!(genes, tour.sequence)
            obj = max(obj, tour.cost)
        end
    end
    
    return Chromosome(genes, obj, 0.0, tours)
end

function initial_random_solution(T::Matrix{Float64}, K::Int, n_nodes::Int, depot_indices::Vector{Int})
    customer_nodes = setdiff(1:n_nodes, depot_indices)
    assignments_ = vcat(rand(1:K, length(customer_nodes) - K), 1:K)
    shuffle!(assignments_)
    
    tours = Tour[]
    genes = Int[]
    obj = 0.0
    
    for i in 1:K
        t1 = findall(==(i), assignments_)
        if length(t1) == 1
            cost = minimum(d -> T[d+1, t1[1]+1] + T[t1[1]+1, d+1], depot_indices)
            push!(tours, Tour(t1, cost))
            push!(genes, t1[1])
            obj = max(obj, cost)
        else
            tour = greedy_insertion_tour(T, copy(t1), depot_indices)
            push!(tours, tour)
            append!(genes, tour.sequence)
            obj = max(obj, tour.cost)
        end
    end
    
    return Chromosome(genes, obj, 0.0, tours)
end

# Additional helper functions for initial solution generation...
# [Continue with remaining functions, updating them to handle depot_indices]