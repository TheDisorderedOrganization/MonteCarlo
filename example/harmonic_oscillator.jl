include("../src/simulation.jl")
include("../systems/particle_1d/particle_1d.jl")

function potential(x)
    return x^2
end

# MONTE CARLO
seed = 42
rng = Xoshiro(seed)
β = 2.0
M = 100
chains = [Particle(4rand(rng) - 2, β) for _ in 1:M]
pools = [(Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),) for _ in 1:M]
steps = 10^5
burn = 1000
block = [0, 10]
sampletimes = scheduler(steps, burn, block)
path = "data/particle_1d/Harmonic/beta$β/M$M/seed$seed"
simulation = Simulation(chains, pools, steps; sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, path=path)
callbacks = (callback_energy, callback_acceptance)
run!(simulation, callbacks...)


# PGMC
include("../src/pgmc/pgmc_simulation.jl")
seed = 42
rng = Xoshiro(seed)
β = 2.0
M = 10
chains = [Particle(4rand(rng) - 2, β) for _ in 1:M]
pools = [(
    Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 0.6),
    Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 0.4),
    ) for _ in 1:M]
optimisers = (Static(), VPG(0.1))
steps = 10^5
burn = 1000
block = [0, 10]
sampletimes = scheduler(steps, burn, block)
path = "data/pgmc/particle_1d/Harmonic/beta$β/M$M/seed$seed"
simulation = Simulation(chains, pools, optimisers, steps; 
    sampletimes=sampletimes, seed=seed, store_trajectory=true, store_parameters=true, parallel=false, verbose=true, path=path)
callbacks = (callback_energy, callback_acceptance)
run!(simulation, callbacks...)