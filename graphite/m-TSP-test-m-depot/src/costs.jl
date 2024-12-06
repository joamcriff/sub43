function calculate_new_cost_add_one(tour::Vector{Int}, cost::Float64, city::Int, position::Int, 
    T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    nt = length(tour)
    if nt == 0
        # For empty tour, find best depot pair
        return minimum(d -> T[d+1, city+1] + T[city+1, d+1], depot_indices)
    else
        if position == 1
            # Try all possible start depots
            best_start = minimum(d -> T[d+1, city+1] + T[city+1, tour[1]+1], depot_indices)
            current_start = minimum(d -> T[d+1, tour[1]+1], depot_indices)
            cost += best_start - current_start
        elseif position == nt + 1
            # Try all possible end depots
            best_end = minimum(d -> T[tour[nt]+1, city+1] + T[city+1, d+1], depot_indices)
            current_end = minimum(d -> T[tour[nt]+1, d+1], depot_indices)
            cost += best_end - current_end
        else
            cost += T[tour[position-1]+1, city+1] + T[city+1, tour[position]+1] - 
                   T[tour[position-1]+1, tour[position]+1]
        end
    end
    return cost
end

function calculate_new_cost_add_two(tour::Vector{Int}, cost::Float64, city1::Int, city2::Int, 
    position::Int, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    nt = length(tour)
    cost1 = cost
    cost2 = cost
    
    if nt == 0
        # For empty tour, find best depot pair for both orientations
        cost1 = minimum(d -> T[d+1, city1+1] + T[city1+1, city2+1] + T[city2+1, d+1], depot_indices)
        cost2 = minimum(d -> T[d+1, city2+1] + T[city2+1, city1+1] + T[city1+1, d+1], depot_indices)
    else
        if position == 1
            cost1 += minimum(d -> T[d+1, city1+1], depot_indices) + T[city1+1, city2+1] + 
                    T[city2+1, tour[1]+1] - minimum(d -> T[d+1, tour[1]+1], depot_indices)
            cost2 += minimum(d -> T[d+1, city2+1], depot_indices) + T[city2+1, city1+1] + 
                    T[city1+1, tour[1]+1] - minimum(d -> T[d+1, tour[1]+1], depot_indices)
        elseif position == nt + 1
            cost1 += T[tour[nt]+1, city1+1] + T[city1+1, city2+1] + 
                    minimum(d -> T[city2+1, d+1], depot_indices) - 
                    minimum(d -> T[tour[nt]+1, d+1], depot_indices)
            cost2 += T[tour[nt]+1, city2+1] + T[city2+1, city1+1] + 
                    minimum(d -> T[city1+1, d+1], depot_indices) - 
                    minimum(d -> T[tour[nt]+1, d+1], depot_indices)
        else
            cost1 += T[tour[position-1]+1, city1+1] + T[city1+1, city2+1] + 
                    T[city2+1, tour[position]+1] - T[tour[position-1]+1, tour[position]+1]
            cost2 += T[tour[position-1]+1, city2+1] + T[city2+1, city1+1] + 
                    T[city1+1, tour[position]+1] - T[tour[position-1]+1, tour[position]+1]
        end
    end
    if cost1 < cost2
        return cost1, true
    else
        return cost2, false
    end
end

function calculate_new_cost_add_three(tour::Vector{Int}, cost::Float64, city1::Int, city2::Int, 
    city3::Int, position::Int, T::Matrix{Float64}, n_nodes::Int, depot_indices::Vector{Int})
    nt = length(tour)
    cost1 = cost
    cost2 = cost
    
    if nt == 0
        cost1 = minimum(d -> T[d+1, city1+1] + T[city1+1, city2+1] + T[city2+1, city3+1] + 
                T[city3+1, d+1], depot_indices)
        cost2 = minimum(d -> T[d+1, city3+1] + T[city3+1, city2+1] + T[city2+1, city1+1] + 
                T[city1+1, d+1], depot_indices)
    else
        if position == 1
            cost1 += minimum(d -> T[d+1, city1+1], depot_indices) + T[city1+1, city2+1] + 
                    T[city2+1, city3+1] + T[city3+1, tour[1]+1] - 
                    minimum(d -> T[d+1, tour[1]+1], depot_indices)
            cost2 += minimum(d -> T[d+1, city3+1], depot_indices) + T[city3+1, city2+1] + 
                    T[city2+1, city1+1] + T[city1+1, tour[1]+1] - 
                    minimum(d -> T[d+1, tour[1]+1], depot_indices)
        elseif position == nt + 1
            cost1 += T[tour[nt]+1, city1+1] + T[city1+1, city2+1] + T[city2+1, city3+1] + 
                    minimum(d -> T[city3+1, d+1], depot_indices) - 
                    minimum(d -> T[tour[nt]+1, d+1], depot_indices)
            cost2 += T[tour[nt]+1, city3+1] + T[city3+1, city2+1] + T[city2+1, city1+1] + 
                    minimum(d -> T[city1+1, d+1], depot_indices) - 
                    minimum(d -> T[tour[nt]+1, d+1], depot_indices)
        else
            cost1 += T[tour[position-1]+1, city1+1] + T[city1+1, city2+1] + 
                    T[city2+1, city3+1] + T[city3+1, tour[position]+1] - 
                    T[tour[position-1]+1, tour[position]+1]
            cost2 += T[tour[position-1]+1, city3+1] + T[city3+1, city2+1] + 
                    T[city2+1, city1+1] + T[city1+1, tour[position]+1] - 
                    T[tour[position-1]+1, tour[position]+1]
        end
    end
    if cost1 < cost2
        return cost1, true
    else
        return cost2, false
    end
end

# Additional cost calculation functions...
# [Previous functions from calculate_new_cost_remove_one through calculate_new_cost_cross_upgraded remain 
# similar but need to be updated with depot_indices parameter and depot consideration logic]

function find_tour_length(tt::Vector{Int}, T::Matrix{Float64}, depot_indices::Vector{Int})
    if isempty(tt)
        return 0.0
    end
    
    # Find best depot pair
    best_cost = Inf
    for start_depot in depot_indices
        for end_depot in depot_indices
            cost = T[start_depot+1, tt[1]+1]  # Start depot to first city
            for i in 1:length(tt)-1
                cost += T[tt[i]+1, tt[i+1]+1]
            end
            cost += T[tt[end]+1, end_depot+1]  # Last city to end depot
            
            if cost < best_cost
                best_cost = cost
            end
        end
    end
    
    return best_cost
end