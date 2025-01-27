include("../src/simulation.jl")
include("../systems/particles/main.jl")

seed = 42
rng = Xoshiro(seed)
N = 300
M = 1
Length = 3
d = 3
temperature = 2.0
density = 1.2
box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
position_1 = [box .* @SVector rand(rng, d) for i in 1:Int(N // Length), m in 1:M]
position = []
for r in position_1
    push!(position, r)
    push!(position, r .+ @SVector [0.1, 0.1, 0.1])
    push!(position, r .- @SVector [0.1, 0.1, 0.1])
end
position = Vector{SVector{3,Float64}}(position)
mol_species = Vector{Tuple{Int, Int, Int, Int}}()
species = Vector{Tuple{Int, Int}}()
for i in 1:Int(N // Length)
    push!(species, (1, i))
    push!(species, (2, i))
    push!(species, (3, i))
    push!(mol_species, (1, 3, 3 * (i - 1) + 1, 3 * i))
end

function create_bond_matrix(N::Int)
    # Create a vector to store the SVectors, each containing a pair of integers
    matrix = SVector{2,Int}[]
    # Populate the matrix with pairs according to the specified pattern
    for i in 2:3:N
        push!(matrix, SVector(i, i + 1))
        push!(matrix, SVector(i - 1, i + 1))
        push!(matrix, SVector(i - 1, i))
    end
    return matrix
end

bonds = create_bond_matrix(N)
epsilon = [1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0]
sigma = [0.9 0.95 1.0; 0.95 1.0 1.05; 1.0 1.05 1.1]
k = [0.0 33.241 30.0; 33.241 0.0 27.210884; 30.0 27.210884 0.0]
r0 = [0.0 1.425 1.5; 1.425 0.0 1.575; 1.5 1.575 0.0]
model = GeneralKG(epsilon, sigma, k, r0)
chains = [System(position, species, mol_species, density, temperature, model, bonds) for _ in 1:M]
## Define moves and combine them into M independent pools
pswap = 0.2
displacement_policy = SimpleGaussian()
displacement_parameters = ComponentArray(σ=0.05)
pools = [(
    Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),
) for _ in 1:M]
## Define the simulation struct
steps = 1000
burn = 1000
block = [0, 1, 2, 4, 8, 16, 32, 64, 128]
sampletimes = scheduler(steps, burn, block)
path = "data/test/particles/Molecules/T$temperature/N$N/M$M/seed$seed"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, store_trajectory=true, parallel=false, verbose=true, path=path)
callbacks = (callback_energy, callback_acceptance)
## Run the simulation :)
run!(simulation, callbacks...)