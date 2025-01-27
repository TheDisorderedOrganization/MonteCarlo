include("../src/simulation.jl")
include("../systems/particles/main.jl")

# GERHARD MODEL
## Define M independent chains of a multicomponent system of N particles in d dimensions
seed = 42
rng = Xoshiro(seed)
NA = 20
NB = 11
NC = 12
N = NA + NB + NC
M = 8
d = 2
temperature = 0.5
density = 1.1920748468939728
box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
position = [[box .* @SVector rand(rng, d) for i in 1:N] for m in 1:M]
species = [shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB), 3ones(Int, NC))) for _ in 1:M]
model = JBB()
chains = [System(position[m], species[m], density, temperature, model) for m in 1:M]
## Define moves and combine them into M independent pools
pswap = 0.2
displacement_policy = SimpleGaussian()
displacement_parameters = ComponentArray(σ=0.065)
swap_policy = DoubleUniform()
swap_parameters = Vector{Float64}()
pools = [(
    Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1 - pswap),
    Move(DiscreteSwap(0, 0, (1, 3), (NA, NC)), swap_policy, swap_parameters, pswap / 2),
    Move(DiscreteSwap(0, 0, (2, 3), (NB, NC)), swap_policy, swap_parameters, pswap / 2),
    ) for _ in 1:M]
## Define the simulation struct
steps = 50000
burn = 10000
# block = [0, 1, 2, 4, 8, 16, 32, 64, 128]
block = [0, 10000]
sampletimes = scheduler(steps, burn, block)
path = "data/test/particles/KA2D/T$temperature/N$N/M$M/seed$seed"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, store_trajectory=true, parallel=false, verbose=true, path=path)
callbacks = (callback_energy, callback_acceptance)
## Run the simulation :)
run!(simulation, callbacks...)

# :)