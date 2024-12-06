mutable struct Label
    Ri::Vector{Int}
    Vir::Vector{Float64}
    Pir::Vector{Int}
    Cir::Vector{Float64}
    depot_assignments::Vector{Int}  # Added to track depot assignments
end

function SPLIT(TT::Matrix{Float64}, K::Int, S::Vector{Int}, depot_indices::Vector{Int})
    n = length(S)
    labels = Vector{Label}(undef, n)
    
    # Initialize labels
    for i in 1:n
        R = if i == n
            [j for j in 1:K]
        else
            [j for j in 1:min(i, K-1)]
        end
        V = fill(Inf, length(R))
        P = fill(n + 1, length(R))
        C = fill(Inf, length(R))
        D = fill(0, length(R))  # Depot assignments
        labels[i] = Label(R, V, P, C, D)
    end

    # Dynamic programming
    for i in 1:n
        R = i > 1 ? labels[i-1].Ri : [0]
        
        for r in R
            Current_V = i > 1 ? labels[i].Vir[r] : 0.0
            
            if Current_V < Inf
                load = 0
                t = zero(eltype(TT))
                j = i
                
                while j <= n
                    # Try all possible depot combinations
                    if i == j
                        t = minimum(d -> TT[d+1, S[j]+1] + TT[S[j]+1, d+1], depot_indices)
                        best_depot = argmin(d -> TT[d+1, S[j]+1] + TT[S[j]+1, d+1], depot_indices)
                    else
                        prev_t = t
                        t = minimum(d1 -> minimum(d2 -> 
                            prev_t - TT[S[j-1]+1, d1+1] + 
                            TT[S[j-1]+1, S[j]+1] + 
                            TT[S[j]+1, d2+1], 
                            depot_indices), 
                            depot_indices)
                    end
                    
                    if r + 1 in labels[j].Ri
                        old_t = i > 1 ? labels[i-1].Vir[r] : 0.0
                        new_t = max(old_t, t)
                        
                        if new_t < labels[j].Vir[r+1]
                            labels[j].Vir[r+1] = new_t
                            labels[j].Pir[r+1] = i - 1
                            labels[j].Cir[r+1] = t
                            labels[j].depot_assignments[r+1] = best_depot
                        end
                    end
                    j += 1
                end
            end
        end
    end

    # Reconstruct solution
    rs = K
    trips = Vector{Tour}(undef, rs)
    for i in 1:rs
        trips[i] = Tour(Int[], 0.0, 0, 0)  # Initialize with start/end depots
    end
    
    tt = rs
    j = n
    
    while tt > 0
        i = labels[j].Pir[tt]
        trips[tt].cost = labels[j].Cir[tt]
        trips[tt].start_depot = labels[j].depot_assignments[tt]
        trips[tt].end_depot = labels[j].depot_assignments[tt]
        
        for k = i+1:j
            push!(trips[tt].sequence, S[k])
        end
        tt -= 1
        j = i
    end

    # Find optimal depot assignments
    for trip in trips
        if !isempty(trip.sequence)
            best_cost = Inf
            best_start = first(depot_indices)
            best_end = first(depot_indices)
            
            for start_depot in depot_indices
                for end_depot in depot_indices
                    cost = TT[start_depot+1, trip.sequence[1]+1]
                    for i in 1:length(trip.sequence)-1
                        cost += TT[trip.sequence[i]+1, trip.sequence[i+1]+1]
                    end
                    cost += TT[trip.sequence[end]+1, end_depot+1]
                    
                    if cost < best_cost
                        best_cost = cost
                        best_start = start_depot
                        best_end = end_depot
                    end
                end
            end
            
            trip.cost = best_cost
            trip.start_depot = best_start
            trip.end_depot = best_end
        end
    end

    obj = minimum(labels[n].Vir)
    return obj, trips
end

mutable struct Tour
    sequence::Vector{Int}
    cost::Float64
    start_depot::Int
    end_depot::Int

    # Constructors
    Tour(sequence::Vector{Int}, cost::Float64) = new(sequence, cost, 0, 0)
    Tour(sequence::Vector{Int}, cost::Float64, start_depot::Int, end_depot::Int) = 
        new(sequence, cost, start_depot, end_depot)
end

function validate_solution(trips::Vector{Tour}, depot_indices::Vector{Int}, K::Int)
    # Ensure number of routes matches K
    if length(trips) != K
        return false, "Number of routes doesn't match K"
    end
    
    # Check depot assignments
    for trip in trips
        if !isempty(trip.sequence)
            if !(trip.start_depot in depot_indices) || !(trip.end_depot in depot_indices)
                return false, "Invalid depot assignment"
            end
        end
    end
    
    # Check for depot nodes in sequences
    for trip in trips
        if any(node in depot_indices for node in trip.sequence)
            return false, "Depot found in tour sequence"
        end
    end
    
    # Check that each customer is visited exactly once
    visited = Dict{Int,Int}()
    for trip in trips
        for node in trip.sequence
            visited[node] = get(visited, node, 0) + 1
        end
    end
    
    for (node, count) in visited
        if count > 1
            return false, "Customer $node visited multiple times"
        end
    end
    
    return true, "Valid solution"
end