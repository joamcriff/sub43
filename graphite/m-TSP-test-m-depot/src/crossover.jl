function crossover_POS(parent1::Vector{Int64}, parent2::Vector{Int64}, n_nodes::Int64, depot_indices::Vector{Int})
    child = zeros(Int64, n_nodes)
    num_pos = rand(1:n_nodes-1)
    selected_pos = sample(setdiff(1:n_nodes, depot_indices), 
                         Weights(ones(n_nodes-length(depot_indices))), 
                         num_pos, replace=false)
    selected_p1 = parent1[selected_pos]
    child[selected_pos] = selected_p1

    for i in parent2
        if !(i in selected_p1) && !(i in depot_indices)
            child[findfirst(x -> x == 0, child)] = i
        end
    end
    return child
end

function crossover_HX(TT::Matrix{Float64}, parent1::Vector{Int64}, parent2::Vector{Int64}, 
    n_nodes::Int64, depot_indices::Vector{Int})
    
    remaining_cities = setdiff(1:n_nodes, depot_indices)
    current_city = rand(remaining_cities)
    child = Int[current_city]
    deleteat!(remaining_cities, findfirst(==(current_city), remaining_cities))
    
    while !isempty(remaining_cities)
        pos1 = findfirst(==(current_city), parent1)
        pos2 = findfirst(==(current_city), parent2)
        next_city = n_nodes + 2
        min_edge = Inf

        for pos in [pos1, pos2]
            if !isnothing(pos)
                # Consider connections to cities and depots
                for candidate in remaining_cities
                    if TT[current_city+1, candidate+1] < min_edge
                        next_city = candidate
                        min_edge = TT[current_city+1, candidate+1]
                    end
                end
            end
        end

        if next_city == n_nodes + 2
            next_city = rand(remaining_cities)
        end

        current_city = next_city
        push!(child, current_city)
        deleteat!(remaining_cities, findfirst(==(current_city), remaining_cities))
    end
    
    return child
end

function tour_crossover2(parent1::Chromosome, parent2::Chromosome, T::Matrix{Float64}, 
    n_nodes::Int64, depot_indices::Vector{Int})
    
    P1_tours = deepcopy(parent1.tours)
    P2_tours = deepcopy(parent2.tours)
    c = Tour[]
    m = length(P1_tours)

    for i in 1:m
        tour1 = P1_tours[i].sequence
        cost1 = P1_tours[i].cost
        cost2 = 0.0
        max_intersection = -1
        tour2 = Int[]
        r2 = 0
        
        for j in 1:length(P2_tours)
            inter = length(intersect(tour1, P2_tours[j].sequence))
            if inter > max_intersection
                max_intersection = inter
                tour2 = P2_tours[j].sequence
                cost2 = P2_tours[j].cost
                r2 = j
            end
        end

        deleteat!(P2_tours, r2)
        
        if length(tour1) <= length(tour2)
            if length(tour1) <= 4
                push!(c, Tour(tour2, cost2))
            else
                idx1, idx2 = sort(sample(2:length(tour1)-1, 2, replace=false))
                cc = vcat(tour2[1:idx1-1], tour1[idx1:idx2], tour2[idx2+1:length(tour2)])
                push!(c, Tour(cc, find_tour_length(cc, T, depot_indices)))
            end
        else
            if length(tour2) <= 4
                push!(c, Tour(tour1, cost1))
            else
                idx1, idx2 = sort(sample(2:length(tour2)-1, 2, replace=false))
                cc = vcat(tour1[1:idx1-1], tour2[idx1:idx2], tour1[idx2+1:length(tour1)])
                push!(c, Tour(cc, find_tour_length(cc, T, depot_indices)))
            end
        end
    end

    # Handle overlapping cities
    counters = zeros(n_nodes)
    outsiders = Int[]
    for tour in c
        delete_indices = Int[]
        for (j, node) in enumerate(tour.sequence)
            if counters[node] > 0 || node in depot_indices
                push!(delete_indices, j)
            else
                counters[node] += 1
            end
        end
        
        if length(delete_indices) == length(tour.sequence)
            tour.cost = 0.0
            tour.sequence = Int[]
        elseif !isempty(delete_indices)
            remove_cities_from_one_tour(tour, delete_indices, T, n_nodes, depot_indices)
            deleteat!(tour.sequence, delete_indices)
        end
    end
    
    sort!(c, by=x -> x.cost, rev=true)
    outsiders = findall(x -> x == 0, counters)
    for city in outsiders
        if !(city in depot_indices)
            put_city_in_tour(c, city, T, n_nodes, depot_indices)
        end
    end

    child = Int[]
    for tour in c
        append!(child, tour.sequence)
    end

    return child
end

# Implement similar depot-aware modifications for other crossover functions
# (tour_crossover3, tour_crossover4, tour_crossover5)
# Each needs to consider depot_indices when manipulating tours

function remove_cities_from_one_tour(tour_::Tour, cities::Vector{Int}, T::Matrix{Float64}, 
    n_nodes::Int, depot_indices::Vector{Int})
    
    nt = length(tour_.sequence)
    index = 1
    i = 1
    seq = Int[]
    seqs = Vector{Vector{Int}}()
    
    while i <= cities[length(cities)]
        if i == cities[index]
            push!(seq, i)
            if i == cities[length(cities)]
                push!(seqs, seq)
            end
            i += 1
            index += 1
        else
            if !isempty(seq)
                push!(seqs, seq)
                seq = Int[]
            end
            i += 1
        end
    end
    
    tour = tour_.sequence
    cost = tour_.cost
    
    for seq in seqs
        ns = length(seq)
        new_cost = find_tour_length(filter(x -> !(x in seq), tour), T, depot_indices)
        tour_.cost = new_cost
    end
end

function put_city_in_tour(tours::Vector{Tour}, city::Int, T::Matrix{Float64}, 
    n_nodes::Int, depot_indices::Vector{Int})
    
    least_increase = Inf
    best_tour = 0
    best_position = 0
    
    for i in eachindex(tours)
        tour = tours[i].sequence
        current_cost = tours[i].cost
        
        for pos in 1:length(tour)+1
            temp_tour = copy(tour)
            insert!(temp_tour, pos, city)
            new_cost = find_tour_length(temp_tour, T, depot_indices)
            increase = new_cost - current_cost
            
            if increase < least_increase
                least_increase = increase
                best_tour = i
                best_position = pos
            end
        end
    end
    
    insert!(tours[best_tour].sequence, best_position, city)
    tours[best_tour].cost += least_increase
end