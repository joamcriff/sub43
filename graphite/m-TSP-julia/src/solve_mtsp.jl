using JSON
using HGSmTSP

# Read the input JSON file
println("Reading input data...")
input_data = JSON.parsefile("/home/lampham/PycharmProjects/sub43/graphite/input/input.json")

# Extract data
n_vehicles = input_data["n_vehicles"]
dist_mtx = input_data["dist_mtx"]
coordinates = input_data["coordinates"]

# Convert arrays to correct type if needed
dist_mtx = Float64.(dist_mtx)
coordinates = Float64.(coordinates)

println("Solving mTSP...")
# Solve the problem
routes, lengths = solve_mTSP(n_vehicles, dist_mtx, coordinates)

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