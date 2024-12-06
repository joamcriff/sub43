function N1!(chrm::Chromosome, TT::Matrix{Float64}, close_nodes::Matrix{Bool}, 
    n_nodes::Int, depot_indices::Vector{Int})   #Shift(0,1)
    
    r1 = argmax([chrm.tours[i].cost for i in 1:length(chrm.tours)])
    routes = [i for i in 1:length(chrm.tours)]
    r2 = rand(setdiff(routes, r1))
    
    tour1 = chrm.tours[r1].sequence
    tour2 = chrm.tours[r2].sequence
    cost1 = chrm.tours[r1].cost
    cost2 = chrm.tours[r2].cost
    
    if isempty(tour1)
        return
    end
    
    # Find valid city for shifting
    k1 = rand(1:length(tour1))
    city1 = tour1[k1]
    
    if city1 in depot_indices
        return
    end
    
    # Find valid insertion positions
    candidates = Int[]
    nt = length(tour2)
    
    if nt == 0
        candidates = [1]
    else
        if close_nodes[n_nodes+1, city1]
            push!(candidates, 1)
        end
        for i = 2:nt
            if close_nodes[city1, tour2[i-1]] || close_nodes[city1, tour2[i]]
                push!(candidates, i)
            end
        end
        if close_nodes[n_nodes+1, city1]
            push!(candidates, nt + 1)
        end
    end
    
    if isempty(candidates)
        return
    end
    
    k2 = rand(candidates)
    new_cost2 = calculate_new_cost_add_one(tour2, cost2, city1, k2, TT, n_nodes, depot_indices)
    new_cost1 = calculate_new_cost_remove_one(tour1, cost1, k1, TT, n_nodes, depot_indices)
    
    if new_cost2 >= cost1
        return
    end
    
    # Perform the move
    insert!(tour2, k2, city1)
    deleteat!(tour1, k1)
    chrm.tours[r1].cost = new_cost1
    chrm.tours[r2].cost = new_cost2
    chrm.fitness = maximum(tour.cost for tour in chrm.tours)
    
    # Update chromosome genes
    update_chromosome_genes!(chrm)
end

function N2!(chrm::Chromosome, TT::Matrix{Float64}, close_nodes::Matrix{Bool}, 
    n_nodes::Int, depot_indices::Vector{Int})   #Swap(1,1)
    
    r1 = argmax([chrm.tours[i].cost for i in 1:length(chrm.tours)])
    routes = [i for i in 1:length(chrm.tours)]
    r2 = rand(setdiff(routes, r1))
    
    tour1 = chrm.tours[r1].sequence
    tour2 = chrm.tours[r2].sequence
    cost1 = chrm.tours[r1].cost
    cost2 = chrm.tours[r2].cost
    
    if isempty(tour1) || isempty(tour2)
        return
    end
    
    k1 = rand(1:length(tour1))
    city1 = tour1[k1]
    
    if city1 in depot_indices
        return
    end
    
    candidates = Int[]
    for (i, city2) in enumerate(tour2)
        if !(city2 in depot_indices) && 
            (close_nodes[city1, city2] || close_nodes[city2, city1])
            push!(candidates, i)
        end
    end
    
    if isempty(candidates)
        return
    end
    
    k2 = rand(candidates)
    city2 = tour2[k2]
    
    new_cost1, new_cost2 = calculate_new_cost_swap_one(
        tour1, cost1, city1, k1, tour2, cost2, city2, k2, TT, n_nodes, depot_indices)
    
    if new_cost1 >= cost1 || new_cost2 >= cost1
        return
    end
    
    # Perform the swap
    tour1[k1] = city2
    tour2[k2] = city1
    chrm.tours[r1].cost = new_cost1
    chrm.tours[r2].cost = new_cost2
    chrm.fitness = maximum(tour.cost for tour in chrm.tours)
    
    update_chromosome_genes!(chrm)
end

# Similar updates for N3 through N7 functions...

function update_chromosome_genes!(chrm::Chromosome)
    chrm.genes = Int[]
    for tour in chrm.tours
        append!(chrm.genes, tour.sequence)
    end
end

function find_tour_neighbors(tours::Vector{Tour}, T::Matrix{Float64}, depot_indices::Vector{Int}, m::Int)
    distances = ones(m, m) * -1
    
    for i in 1:m-1
        for j = i+1:m
            if isempty(tours[i].sequence) || isempty(tours[j].sequence)
                distances[i,j] = distances[j,i] = Inf
                continue
            end
            
            # Calculate minimum distance between any pair of cities in the tours
            min_dist = Inf
            for city1 in tours[i].sequence
                for city2 in tours[j].sequence
                    if !(city1 in depot_indices) && !(city2 in depot_indices)
                        dist = T[city1+1, city2+1]
                        min_dist = min(min_dist, dist)
                    end
                end
            end
            distances[i,j] = distances[j,i] = min_dist
        end
    end
    
    return [sortperm(distances[i,:])[2:min(m,5)] for i in 1:m]
end

function cross_exchange!(chrm::Chromosome, TT::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    r1 = argmax([tour.cost for tour in chrm.tours])
    routes = [i for i in 1:length(chrm.tours)]
    r2 = rand(setdiff(routes, r1))
    
    tour1 = chrm.tours[r1].sequence
    tour2 = chrm.tours[r2].sequence
    
    if length(tour1) < 2 || length(tour2) < 2
        return
    end
    
    # Select segments that don't include depots
    valid_segments1 = find_valid_segments(tour1, depot_indices)
    valid_segments2 = find_valid_segments(tour2, depot_indices)
    
    if isempty(valid_segments1) || isempty(valid_segments2)
        return
    end
    
    seg1 = rand(valid_segments1)
    seg2 = rand(valid_segments2)
    
    # Try exchange
    new_tour1 = create_exchanged_tour(tour1, tour2, seg1, seg2)
    new_tour2 = create_exchanged_tour(tour2, tour1, seg2, seg1)
    
    new_cost1 = find_tour_length(new_tour1, TT, depot_indices)
    new_cost2 = find_tour_length(new_tour2, TT, depot_indices)
    
    if max(new_cost1, new_cost2) < chrm.fitness
        chrm.tours[r1].sequence = new_tour1
        chrm.tours[r2].sequence = new_tour2
        chrm.tours[r1].cost = new_cost1
        chrm.tours[r2].cost = new_cost2
        chrm.fitness = max(new_cost1, new_cost2)
        
        update_chromosome_genes!(chrm)
    end
end

function find_valid_segments(tour::Vector{Int}, depot_indices::Vector{Int})
    segments = Tuple{Int,Int}[]
    start_idx = 1
    
    while start_idx < length(tour)
        if tour[start_idx] in depot_indices
            start_idx += 1
            continue
        end
        
        end_idx = start_idx + 1
        while end_idx <= length(tour) && !(tour[end_idx] in depot_indices)
            push!(segments, (start_idx, end_idx))
            end_idx += 1
        end
        start_idx = end_idx
    end
    
    return segments
end

function create_exchanged_tour(base_tour::Vector{Int}, other_tour::Vector{Int}, 
    base_seg::Tuple{Int,Int}, other_seg::Tuple{Int,Int})
    
    result = copy(base_tour)
    exchanged_segment = other_tour[other_seg[1]:other_seg[2]]
    result[base_seg[1]:base_seg[2]] = exchanged_segment
    return result
end