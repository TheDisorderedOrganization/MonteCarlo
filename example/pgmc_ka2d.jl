include("../src/simulation.jl")
include("../systems/particles/main.jl")
include("../src/pgmc/pgmc_simulation.jl")

seed = 42
rng = Xoshiro(seed)
NA = 20
NB = 11
NC = 12
N = NA + NB + NC
M = 1
d = 2
temperature = 0.5
density = 1.1920748468939728
box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
position = [[box .* @SVector rand(rng, d) for i in 1:N] for m in 1:M]
species = [shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB), 3ones(Int, NC))) for _ in 1:M]
model = JBB()
chains = [System(position[m], species[m], density, temperature, model) for m in 1:M]
pswap = 0.2
displacement_policy = SimpleGaussian()
displacement_parameters = ComponentArray(σ=0.05)
swap_AC_policy = EnergyBias()
swap_BC_policy = EnergyBias()
swap_AC_parameters = ComponentArray(θ₁=0.0, θ₂=0.0)
swap_BC_parameters = ComponentArray(θ₁=0.0, θ₂=0.0)
pools = [(
    Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1 - pswap),
    Move(DiscreteSwap(0, 0, (1, 3), (NA, NC)), swap_AC_policy, swap_AC_parameters, pswap / 2),
    Move(DiscreteSwap(0, 0, (2, 3), (NB, NC)), swap_BC_policy, swap_BC_parameters, pswap / 2),
) for _ in 1:M]
optimisers = (BLANPG(1e-5), BLAPG(1e-4), BLAPG(1e-4))
steps = 10000
burn = 10000
block = [0, 1, 2, 4, 8, 16, 32, 64, 128]
sampletimes = scheduler(steps, burn, block)
q_batch_size = 50
sweeps_per_gradient = 1
sweeps_per_learning = 1
path = "data/pgmc/particles/KA2D/T$temperature/N$N/M$M/seed$seed"
simulation = Simulation(chains, pools, optimisers, steps;
    sweepstep=N, sampletimes=sampletimes, seed=seed,
    q_batch_size=q_batch_size, sweeps_per_gradient=sweeps_per_gradient, sweeps_per_learning=sweeps_per_learning,
    store_trajectory=true, store_parameters=true, parallel=false, verbose=true, path=path, ad_backend=Zygote_Backend())
callbacks = (callback_energy, callback_acceptance)
run!(simulation, callbacks...)

# :)