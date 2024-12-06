function segments_intersect_orient(A::AbstractVector{Float64}, B::AbstractVector{Float64}, 
    C::AbstractVector{Float64}, D::AbstractVector{Float64})
    # Unpack coordinates
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D

    # Calculate orientations
    o1 = orientation((x1, y1), (x2, y2), (x3, y3))
    o2 = orientation((x1, y1), (x2, y2), (x4, y4))
    o3 = orientation((x3, y3), (x4, y4), (x1, y1))
    o4 = orientation((x3, y3), (x4, y4), (x2, y2))

    return o1 != o2 && o3 != o4
end

function orientation(p, q, r)
    val = (q[2] - p[2]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[2] - q[2])
    if val == 0
        return 0  # Collinear
    end
    return val > 0 ? 1 : 2  # Clockwise or Counterclockwise
end

function find_intersections(t1::Vector{Int}, t2::Vector{Int}, customers::Matrix{Float64}, 
    depot_coordinates::Matrix{Float64}, depot_indices::Vector{Int})
    
    for i in 0:length(t1)
        if i == 0 || i == length(t1)
            loopset = 1:length(t2)-1
        else
            loopset = 0:length(t2)
        end

        # Get current segment's start point
        node1 = if i == 0
            get_depot_coordinates(t1[1], depot_coordinates, depot_indices)
        elseif i == length(t1)
            @view customers[t1[i], :]
        else
            @view customers[t1[i], :]
        end

        # Get current segment's end point
        node2 = if i == 0
            @view customers[t1[1], :]
        elseif i == length(t1)
            get_depot_coordinates(t1[end], depot_coordinates, depot_indices)
        else
            @view customers[t1[i+1], :]
        end

        for j in loopset
            # Get test segment's start point
            node3 = if j == 0
                get_depot_coordinates(t2[1], depot_coordinates, depot_indices)
            elseif j == length(t2)
                @view customers[t2[j], :]
            else
                @view customers[t2[j], :]
            end

            # Get test segment's end point
            node4 = if j == 0
                @view customers[t2[1], :]
            elseif j == length(t2)
                get_depot_coordinates(t2[end], depot_coordinates, depot_indices)
            else
                @view customers[t2[j+1], :]
            end

            if segments_intersect_orient(node1, node2, node3, node4)
                return i, j, true
            end
        end
    end
    return 0, 0, false
end

function get_depot_coordinates(city::Int, depot_coordinates::Matrix{Float64}, depot_indices::Vector{Int})
    # Find closest depot to the city
    closest_depot_idx = argmin([norm(depot_coordinates[i,:] - city) for i in 1:size(depot_coordinates,1)])
    return @view depot_coordinates[closest_depot_idx, :]
end

function solve_one_intersections(t1::Vector{Int}, t2::Vector{Int}, T::Matrix{Float64}, 
    k1::Int, k2::Int, depot_indices::Vector{Int})
    
    n1 = length(t1)
    n2 = length(t2)
    
    # Split tours at intersection points
    a = k1 == 0 ? Int[] : t1[1:k1]
    b = k2 == n2 ? Int[] : t2[k2+1:n2]
    c = k2 == 0 ? Int[] : t2[1:k2]
    d = k1 == n1 ? Int[] : t1[k1+1:n1]
    
    # Create new tour combinations
    tour11 = vcat(a, b)
    tour12 = vcat(c, d)
    tour21 = vcat(a, reverse(c))
    tour22 = vcat(reverse(b), d)
    
    # Calculate costs with multiple depot consideration
    cost11 = find_tour_length(tour11, T, depot_indices)
    cost12 = find_tour_length(tour12, T, depot_indices)
    cost21 = find_tour_length(tour21, T, depot_indices)
    cost22 = find_tour_length(tour22, T, depot_indices)
    
    # Return best combination
    if max(cost11, cost12) < max(cost21, cost22)
        return tour11, tour12, cost11, cost12
    else
        return tour21, tour22, cost21, cost22
    end
end

function solve_all_intersections!(chrm::Chromosome, customers::Matrix{Float64}, 
    depot_coordinates::Matrix{Float64}, T::Matrix{Float64}, depot_indices::Vector{Int})
    
    m = length(chrm.tours)
    n_nodes = length(chrm.genes)
    intersected_ = true
    
    while intersected_
        intersected_ = false
        for i in 1:m-1
            for j = i+1:m
                tour1 = chrm.tours[i].sequence
                tour2 = chrm.tours[j].sequence
                
                if !isempty(tour1) && !isempty(tour2)
                    intersected = true
                    while intersected
                        k1, k2, intersected = find_intersections(tour1, tour2, customers, 
                                                               depot_coordinates, depot_indices)
                        if intersected
                            t1, t2, c1, c2 = solve_one_intersections(tour1, tour2, T, k1, k2, 
                                                                   depot_indices)
                            tour1 = copy(t1)
                            tour2 = copy(t2)
                            chrm.tours[i].cost = c1
                            chrm.tours[j].cost = c2
                            chrm.tours[i].sequence = tour1
                            chrm.tours[j].sequence = tour2
                            intersected_ = true
                        end
                    end
                end
            end
        end
        
        for tour in chrm.tours
            optimize_tour_depots!(tour, T, n_nodes, depot_indices)
        end
    end

    if rand() < 0.1
        improve_after_removing_intersections(chrm.tours, T, n_nodes, m, customers, 
                                          depot_coordinates, depot_indices)
    end
    
    # Update chromosome
    chrm.genes = Int[]
    chrm.fitness = 0.0
    for tour in chrm.tours
        append!(chrm.genes, tour.sequence)
        chrm.fitness = max(chrm.fitness, tour.cost)
    end
end

function optimize_tour_depots!(tour::Tour, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    if isempty(tour.sequence)
        return
    end
    
    # Try all depot combinations
    best_cost = Inf
    best_start = first(depot_indices)
    best_end = first(depot_indices)
    
    for start_depot in depot_indices
        for end_depot in depot_indices
            cost = T[start_depot+1, tour.sequence[1]+1]  # Start depot to first city
            for i in 1:length(tour.sequence)-1
                cost += T[tour.sequence[i]+1, tour.sequence[i+1]+1]
            end
            cost += T[tour.sequence[end]+1, end_depot+1]  # Last city to end depot
            
            if cost < best_cost
                best_cost = cost
                best_start = start_depot
                best_end = end_depot
            end
        end
    end
    
    tour.cost = best_cost
    tour.start_depot = best_start
    tour.end_depot = best_end
end