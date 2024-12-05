using JSON
include(joinpath(@__DIR__, "../src/main.jl"))

# Read the input JSON file
println("Reading input data...")
input_data = JSON.parsefile("/home/lampham/PycharmProjects/sub43/graphite/input/input.json")

# Extract data
n_vehicles = input_data["n_vehicles"]
dist_mtx_raw = input_data["dist_mtx"]
coordinates_raw = input_data["coordinates"]

# Convert nested arrays to 2D array of Float64
dist_mtx = [Float64(x) for x in vcat(dist_mtx_raw...)]
dist_mtx = reshape(dist_mtx, length(dist_mtx_raw), length(dist_mtx_raw))

# Convert coordinates to 2D array of Float64
coordinates = [Float64(x) for x in vcat(coordinates_raw...)]
coordinates = reshape(coordinates, length(coordinates_raw), 2)

println("Solving mTSP...")
# Solve the problem
routes, lengths = solve_mTSP(n_vehicles, dist_mtx, coordinates,n_iterations=100, time_limit=5.0)

println("Writing output...")
# Prepare output
output_data = Dict(
    "routes" => routes,
    "lengths" => lengths
)

# Write the results to output JSON file
open("/home/lampham/PycharmProjects/sub43/graphite/output/output.json", "w") do f
    JSON.print(f, output_data)
end

println("Done!")