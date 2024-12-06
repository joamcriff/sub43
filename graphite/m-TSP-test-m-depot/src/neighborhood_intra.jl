function Ni1!(chrm::Chromosome, TT::Matrix{Float64}, close_nodes::Matrix{Bool}, 
    n_nodes::Int, depot_indices::Vector{Int})   #Reinsert
    
    r1 = rand() < 0.5 ? argmax([tour.cost for tour in chrm.tours]) : rand(1:length(chrm.tours))
    tour1 = chrm.tours[r1].sequence
    cost1 = chrm.tours[r1].cost

    if length(tour1) <= 1
        return
    end

    # Select a non-depot city
    valid_positions = findall(i -> !(tour1[i] in depot_indices), 1:length(tour1))
    if isempty(valid_positions)
        return
    end
    
    k1 = rand(valid_positions)
    city1 = tour1[k1]
    candidates = Int[]
    
    if length(tour1) == 2
        candidates = [1, 2]
    else
        if close_nodes[n_nodes+1, city1]
            push!(candidates, 1)
        end
        for i in 2:length(tour1)-1
            if i > k1
                if close_nodes[city1, tour1[i]] || close_nodes[city1, tour1[i+1]]
                    push!(candidates, i)
                end
            elseif i < k1
                if close_nodes[city1, tour1[i]] || close_nodes[city1, tour1[i-1]]
                    push!(candidates, i)
                end
            end
        end
        if close_nodes[n_nodes+1, city1]
            push!(candidates, length(tour1))
        end
    end
    
    candidates = setdiff(collect(Set(candidates)), [k1])
    if isempty(candidates)
        return
    end

    k2 = rand(candidates)
    new_cost1 = calculate_new_cost_exchange_one(tour1, cost1, city1, k1, k2, TT, n_nodes, depot_indices)

    if new_cost1 >= cost1
        return
    end

    deleteat!(tour1, k1)
    insert!(tour1, k2, city1)

    chrm.tours[r1].cost = new_cost1
    chrm.fitness = maximum(tour.cost for tour in chrm.tours)
    update_chromosome_genes!(chrm)
end

function Ni2!(chrm::Chromosome, TT::Matrix{Float64}, close_nodes::Matrix{Bool}, 
    n_nodes::Int, depot_indices::Vector{Int})   #Exchange (permutation)
    
    r1 = rand() < 0.5 ? argmax([tour.cost for tour in chrm.tours]) : rand(1:length(chrm.tours))
    tour1 = chrm.tours[r1].sequence
    
    if length(tour1) <= 1
        return
    end
    
    cost1 = chrm.tours[r1].cost
    
    # Find valid positions (non-depot cities)
    valid_positions = findall(i -> !(tour1[i] in depot_indices), 1:length(tour1))
    if length(valid_positions) < 2
        return
    end
    
    k1 = rand(valid_positions)
    city1 = tour1[k1]
    
    candidates = Int[]
    for pos in valid_positions
        if pos != k1
            if pos == 1
                if close_nodes[n_nodes+1, city1] || close_nodes[city1, tour1[2]]
                    push!(candidates, 1)
                end
            elseif pos == length(tour1)
                if close_nodes[n_nodes+1, city1] || close_nodes[city1, tour1[end-1]]
                    push!(candidates, length(tour1))
                end
            else
                if close_nodes[city1, tour1[pos-1]] || close_nodes[city1, tour1[pos+1]]
                    push!(candidates, pos)
                end
            end
        end
    end
    
    if isempty(candidates)
        return
    end
    
    k2 = rand(candidates)
    city2 = tour1[k2]
    
    new_cost1 = calculate_new_cost_exchange_two(tour1, cost1, city1, k1, city2, k2, TT, n_nodes, depot_indices)
    
    if new_cost1 >= cost1
        return
    end
    
    tour1[k1] = city2
    tour1[k2] = city1
    
    chrm.tours[r1].cost = new_cost1
    chrm.fitness = maximum(tour.cost for tour in chrm.tours)
    update_chromosome_genes!(chrm)
end

function Ni3!(chrm::Chromosome, T::Matrix{Float64}, close_nodes::Matrix{Bool}, 
    n_nodes::Int, depot_indices::Vector{Int})   #Or-opt2
    
    r1 = rand() < 0.5 ? argmax([tour.cost for tour in chrm.tours]) : rand(1:length(chrm.tours))
    tour1 = chrm.tours[r1].sequence
    
    if length(tour1) <= 2
        return
    end
    
    # Find valid segments (non-depot cities)
    valid_segments = find_valid_segments(tour1, depot_indices)
    if isempty(valid_segments)
        return
    end
    
    # Select a random valid segment
    k1, k2 = rand(valid_segments)
    if k2 - k1 < 1
        return
    end
    
    cost1 = chrm.tours[r1].cost
    new_tour = copy(tour1)
    segment = tour1[k1:k2]
    
    # Try all possible positions
    best_cost = cost1
    best_position = k1
    for i in 1:length(tour1)-1
        if i != k1 && i != k2
            test_tour = copy(tour1)
            deleteat!(test_tour, k1:k2)
            splice!(test_tour, i:i-1, segment)
            new_cost = find_tour_length(test_tour, T, depot_indices)
            if new_cost < best_cost
                best_cost = new_cost
                best_position = i
                new_tour = test_tour
            end
        end
    end
    
    if best_cost < cost1
        chrm.tours[r1].sequence = new_tour
        chrm.tours[r1].cost = best_cost
        chrm.fitness = maximum(tour.cost for tour in chrm.tours)
        update_chromosome_genes!(chrm)
    end
end

function Ni4!(chrm::Chromosome, T::Matrix{Float64}, close_nodes::Matrix{Bool}, 
    n_nodes::Int, depot_indices::Vector{Int})   #Or-opt3
    
    r1 = rand() < 0.5 ? argmax([tour.cost for tour in chrm.tours]) : rand(1:length(chrm.tours))
    tour1 = chrm.tours[r1].sequence
    
    if length(tour1) <= 3
        return
    end
    
    # Find valid segments of length 3 (non-depot cities)
    valid_segments = find_valid_segments_of_length(tour1, depot_indices, 3)
    if isempty(valid_segments)
        return
    end
    
    k1 = rand(valid_segments)
    segment = tour1[k1:k1+2]
    cost1 = chrm.tours[r1].cost
    
    # Try all possible positions
    best_cost = cost1
    best_position = k1
    new_tour = copy(tour1)
    
    for i in 1:length(tour1)-2
        if i != k1 && i != k1+1 && i != k1+2
            test_tour = copy(tour1)
            deleteat!(test_tour, k1:k1+2)
            splice!(test_tour, i:i-1, segment)
            new_cost = find_tour_length(test_tour, T, depot_indices)
            if new_cost < best_cost
                best_cost = new_cost
                best_position = i
                new_tour = test_tour
            end
        end
    end
    
    if best_cost < cost1
        chrm.tours[r1].sequence = new_tour
        chrm.tours[r1].cost = best_cost
        chrm.fitness = maximum(tour.cost for tour in chrm.tours)
        update_chromosome_genes!(chrm)
    end
end

function Ni5!(chrm::Chromosome, T::Matrix{Float64}, close_nodes::Matrix{Bool}, 
    n_nodes::Int, depot_indices::Vector{Int})   #2-opt
    
    r1 = rand() < 0.5 ? argmax([tour.cost for tour in chrm.tours]) : rand(1:length(chrm.tours))
    tour1 = chrm.tours[r1].sequence
    
    if length(tour1) <= 2
        return
    end
    
    # Find segments between depots
    depot_segments = find_segments_between_depots(tour1, depot_indices)
    if isempty(depot_segments)
        return
    end
    
    # Select a random segment and try 2-opt within it
    seg_start, seg_end = rand(depot_segments)
    if seg_end - seg_start < 2
        return
    end
    
    cost1 = chrm.tours[r1].cost
    best_cost = cost1
    best_tour = copy(tour1)
    
    for i in seg_start:seg_end-1
        for j in i+1:seg_end
            new_tour = copy(tour1)
            new_tour[i:j] = reverse(tour1[i:j])
            new_cost = find_tour_length(new_tour, T, depot_indices)
            if new_cost < best_cost
                best_cost = new_cost
                best_tour = new_tour
            end
        end
    end
    
    if best_cost < cost1
        chrm.tours[r1].sequence = best_tour
        chrm.tours[r1].cost = best_cost
        chrm.fitness = maximum(tour.cost for tour in chrm.tours)
        update_chromosome_genes!(chrm)
    end
end

# Helper functions
function find_segments_between_depots(tour::Vector{Int}, depot_indices::Vector{Int})
    segments = Tuple{Int,Int}[]
    start_idx = 1
    
    while start_idx < length(tour)
        while start_idx <= length(tour) && tour[start_idx] in depot_indices
            start_idx += 1
        end
        
        if start_idx > length(tour)
            break
        end
        
        end_idx = start_idx
        while end_idx < length(tour) && !(tour[end_idx+1] in depot_indices)
            end_idx += 1
        end
        
        if end_idx > start_idx
            push!(segments, (start_idx, end_idx))
        end
        
        start_idx = end_idx + 1
    end
    
    return segments
end

function find_valid_segments_of_length(tour::Vector{Int}, depot_indices::Vector{Int}, length::Int)
    valid_starts = Int[]
    for i in 1:length(tour)-length+1
        valid = true
        for j in 0:length-1
            if tour[i+j] in depot_indices
                valid = false
                break
            end
        end
        if valid
            push!(valid_starts, i)
        end
    end
    return valid_starts
end

function update_chromosome_genes!(chrm::Chromosome)
    chrm.genes = Int[]
    for tour in chrm.tours
        append!(chrm.genes, tour.sequence)
    end
end