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

mutable struct Chromosome
    genes::Vector{Int64}
    fitness::Float64
    power::Float64
    tours::Vector{Tour}
end

function parent_selection_RWS(population::Vector{Chromosome}, total_p::Float64, popsize::Int64)
    r = -rand() * total_p
    summ = 0
    for i in 1:popsize
        summ -= population[i].power
        if r > summ
            return population[i]
        end
    end
    return population[end]  # Fallback
end

function parent_selection_TS(population::Vector{Chromosome}, k::Int64, popsize::Int64)
    idx = sample(1:popsize, k, replace=false)
    return population[idx[argmin(idx)]]
end

function parent_selection_RkS(population::Vector{Chromosome}, ranks::Vector{Int64}, 
    total_r::Int64, popsize::Int64)
    r = rand() * total_r
    summ = 0
    for i in 1:popsize
        summ += ranks[i]
        if r < summ
            return population[i]
        end
    end
    return population[end]  # Fallback
end

function select_parents(population::Vector{Chromosome}, k_tournament::Int64, popsize::Int64)
    return parent_selection_TS(population, k_tournament, popsize),
           parent_selection_TS(population, k_tournament, popsize)
end

function reproduce(TT::Matrix{Float64}, parent1::Chromosome, parent2::Chromosome, 
    n_nodes::Int64, depot_indices::Vector{Int}, crossover_functions::Vector{Int})::Vector{Int}
    r::Int = rand(crossover_functions)
    if r == 1
        return crossover_HX(TT, parent1.genes, parent2.genes, n_nodes, depot_indices)
    elseif r == 2
        return tour_crossover2(parent1, parent2, TT, n_nodes, depot_indices)
    elseif r == 3
        return tour_crossover3(parent1, parent2, TT, n_nodes, depot_indices)
    elseif r == 4
        return tour_crossover4(parent1, parent2, TT, n_nodes, depot_indices)
    elseif r == 5
        return tour_crossover5(parent1, parent2, TT, n_nodes, depot_indices)
    end
end

function find_difference(c1::Vector{Int64}, c2::Vector{Int64})
    diff1 = 0
    diff2 = 0
    c3 = reverse(c2)
    for i in 1:length(c1)
        if c1[i] != c2[i]
            diff1 += 1
        end
        if c1[i] != c3[i]
            diff2 += 1
        end
    end
    return min(diff1, diff2) / length(c1)
end

function find_difference(c1::Chromosome, c2::Chromosome)
    m = length(c1.tours)
    n = length(c1.genes)
    A = zeros(Int, m, m)
    for i in 1:m
        for j in 1:m
            A[i, j] = length(intersect(Set(c1.tours[i].sequence), Set(c2.tours[j].sequence)))
        end
    end
    summ = 0
    while true
        idx = argmax(A)
        if A[idx] == 0
            break
        end
        i = idx[1]
        j = idx[2]
        summ += A[i, j]
        A[i, :] .= 0
        A[:, j] .= 0
    end
    return 1 - summ / n
end

function sort_based_on_power!(population::Vector{Chromosome}, num_nei::Int)
    popsize = length(population)
    for i in 1:popsize
        neighbors = find_neighbors(popsize, i, num_nei)
        diff = mean(find_difference(population[i].genes, population[j].genes) 
            for j in neighbors)
        population[i].power = population[i].fitness * 0.8^diff
    end
    sort!(population, by=x -> x.power)
end

function perform_survival_plan!(population::Vector{Chromosome}, mu::Int64, sigma::Int64)
    if length(population) >= mu + sigma
        # Remove duplicates
        del_idx = Int[]
        for i in 1:length(population)-1
            for j in i+1:length(population)
                if population[i].genes == population[j].genes
                    push!(del_idx, j)
                end
            end
        end
        unique!(del_idx)
        deleteat!(population, sort(del_idx))
        
        # Remove worst solutions if still too many
        while length(population) > mu
            pop!(population)
        end
    end
end

function perform_genetic_algorithm(
    TT::Matrix{Float64},
    K::Int,
    depot_indices::Vector{Int},
    h::Float64,
    popsize::Tuple{Int64,Int64},
    k_tournament::Int64,
    num_iter::Int64,
    time_limit::Float64,
    mutation_chance::Float64,
    num_nei::Int,
    crossover_functions::Vector{Int},
    customers::Matrix{Float64},
    depot_coordinates::Matrix{Float64};
    verbose::Bool=false
)
    mu, sigma = popsize
    n_nodes = size(TT)[1] - 2
    
    # Initialize population
    population, old_best = generate_initial_population(
        TT, K, mu, depot_indices, customers, depot_coordinates)
    
    improve_count = 0
    gen_num = 0
    t0 = time()
    roullet = ones(Int, 4) * 100
    
    while improve_count < num_iter && (time() - t0) < time_limit
        sort_based_on_power!(population, num_nei)
        
        # Create offspring
        if rand() < mutation_chance
            offspring = mutate(rand(population[1:5]), TT, n_nodes, depot_indices)
        else
            parent1, parent2 = select_parents(population, k_tournament, length(population))
            child = reproduce(TT, parent1, parent2, n_nodes, depot_indices, crossover_functions)
            obj, trips = SPLIT(TT, K, child, depot_indices)
            offspring = Chromosome(child, obj, 0.0, trips)
        end
        
        # Improve offspring and add to population
        educate_and_add_the_offspring!(offspring, population, TT, depot_indices,
            customers, depot_coordinates, old_best, roullet, n_nodes, improve_count)
        
        # Selection
        perform_survival_plan!(population, mu, sigma)
        
        # Update statistics
        new_best = population[1].fitness
        if round(old_best, digits=3) > round(new_best, digits=3)
            old_best = new_best
            improve_count = 0
        else
            improve_count += 1
        end
        
        if verbose && gen_num % 1000 == 0
            println("Generation ", gen_num, " best objective: ", old_best,
                " time left: $(round(t0+time_limit-time())) seconds")
        end
        gen_num += 1
    end
    
    return population, roullet
end