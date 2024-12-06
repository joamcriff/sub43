function two_opt_mutation(chrm::Chromosome, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    r1 = rand() < 0.5 ? argmax([tour.cost for tour in chrm.tours]) : rand(1:length(chrm.tours))
    tour1 = chrm.tours[r1].sequence
    
    if length(tour1) <= 2
        return chrm
    end
    
    # Try 2-opt move avoiding depot nodes
    cost1 = chrm.tours[r1].cost
    k1, k2 = sort(sample(1:length(tour1), 2, replace=false))
    
    new_tour = copy(tour1)
    new_tour[k1:k2] = reverse(tour1[k1:k2])
    new_cost = find_tour_length(new_tour, T, depot_indices)
    
    if new_cost < cost1
        tour1[k1:k2] = reverse(tour1[k1:k2])
        chrm.tours[r1].cost = new_cost
        
        # Update chromosome
        chrm.genes = Int[]
        chrm.fitness = maximum(tour.cost for tour in chrm.tours)
        for tour in chrm.tours
            append!(chrm.genes, tour.sequence)
        end
    end
    
    return chrm
end

function scatter_mutation(chrm::Chromosome, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    moving_nodes = Int[]

    # Remove random nodes from each tour
    for tour in chrm.tours
        if !isempty(tour.sequence)
            c = sort(sample(1:length(tour.sequence), rand(1:min(3, length(tour.sequence))), replace=false))
            cities = copy(tour.sequence[c])
            deleteat!(tour.sequence, c)
            tour.cost = find_tour_length(tour.sequence, T, depot_indices)
            append!(moving_nodes, cities)
        end
    end

    # Reinsert moved nodes
    for city in moving_nodes
        if !(city in depot_indices)
            put_city_in_tour(chrm.tours, city, T, n_nodes, depot_indices)
        end
    end

    # Update chromosome
    chrm.genes = Int[]
    chrm.fitness = maximum(tour.cost for tour in chrm.tours)
    for tour in chrm.tours
        append!(chrm.genes, tour.sequence)
    end
    
    return chrm
end

function mix_neighbors_mutation(chrm::Chromosome, customers::Matrix{Float64}, 
    depot_coordinates::Matrix{Float64}, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    
    m = length(chrm.tours)
    r1 = rand() < 0.5 ? argmax([tour.cost for tour in chrm.tours]) : rand(1:length(chrm.tours))
    
    tour1 = chrm.tours[r1].sequence
    if length(tour1) <= 4
        return chrm
    end
    
    # Find nearest neighbor tour
    tour_neighbors = find_tour_neighbors(chrm.tours, customers, depot_coordinates, m)
    r2 = tour_neighbors[r1][1]
    tour2 = chrm.tours[r2].sequence
    
    if length(tour2) <= 4
        return chrm
    end

    # Perform crossover on subsections
    if length(tour1) <= length(tour2)
        idx1, idx2 = sort(sample(2:length(tour1)-1, 2, replace=false))
        t1 = vcat(tour2[1:idx1-1], tour1[idx1:idx2], tour2[idx2+1:length(tour2)])
        t2 = vcat(tour1[1:idx1-1], tour2[idx1:idx2], tour1[idx2+1:length(tour1)])
    else
        idx1, idx2 = sort(sample(2:length(tour2)-1, 2, replace=false))
        t1 = vcat(tour1[1:idx1-1], tour2[idx1:idx2], tour1[idx2+1:length(tour1)])
        t2 = vcat(tour2[1:idx1-1], tour1[idx1:idx2], tour2[idx2+1:length(tour2)])
    end
    
    # Update tours with new assignments
    chrm.tours[r1] = Tour(t1, find_tour_length(t1, T, depot_indices))
    chrm.tours[r2] = Tour(t2, find_tour_length(t2, T, depot_indices))
    
    # Update chromosome
    chrm.genes = Int[]
    chrm.fitness = maximum(tour.cost for tour in chrm.tours)
    for tour in chrm.tours
        append!(chrm.genes, tour.sequence)
    end
    
    return chrm
end

function rearange_nodes(chrm::Chromosome, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    c = deepcopy(chrm.tours)
    deleted_nodes = Int[]
    
    # Remove random nodes
    for tour in c
        dlt_idx = Int[]
        for (i, node) in enumerate(tour.sequence)
            if rand() < 0.05 && !(node in depot_indices)
                push!(dlt_idx, i)
            end
        end

        if !isempty(dlt_idx)
            append!(deleted_nodes, tour.sequence[dlt_idx])
            if length(dlt_idx) == length(tour.sequence)
                tour.cost = 0.0
                tour.sequence = Int[]
            else
                remove_cities_from_one_tour(tour, dlt_idx, T, n_nodes, depot_indices)
                deleteat!(tour.sequence, dlt_idx)
            end
        end
    end

    # Reinsert deleted nodes
    if !isempty(deleted_nodes)
        for node in deleted_nodes
            if !(node in depot_indices)
                put_city_in_short_tour(c, node, T, n_nodes, depot_indices)
            end
        end
    end

    # Create new chromosome
    chrm = Chromosome(Int[], 0.0, 0.0, c)
    chrm.fitness = maximum(tour.cost for tour in c)
    for tour in c
        append!(chrm.genes, tour.sequence)
    end
    
    return chrm
end

function destroy_and_build(chrm::Chromosome, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    c = deepcopy(chrm.tours)
    r1 = argmax([tour.cost for tour in c])
    deleted_nodes = copy(c[r1].sequence)
    c[r1].sequence = Int[]
    c[r1].cost = 0.0
    
    if !isempty(deleted_nodes)
        for node in deleted_nodes
            if !(node in depot_indices)
                put_city_in_short_tour(c, node, T, n_nodes, depot_indices)
            end
        end
    end

    # Create new chromosome
    chrm = Chromosome(Int[], 0.0, 0.0, c)
    chrm.fitness = maximum(tour.cost for tour in c)
    for tour in c
        append!(chrm.genes, tour.sequence)
    end
    
    return chrm
end

function mutate(chrm::Chromosome, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    new_chrm = deepcopy(chrm)
    if rand() < 0.5
        return destroy_and_build(new_chrm, T, n_nodes, depot_indices)
    else
        return rearange_nodes(new_chrm, T, n_nodes, depot_indices)
    end
end