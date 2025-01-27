# TEST WITH GERHARD'S CONFIG
using MCMC
using MCMC.ParticlesModule
using StaticArrays
using Distributions
using Random
using ComponentArrays

data = readlines("test/config_0.xyz")
N = parse(Int, data[4])
gbox = parse.(Float64, split(data[6], " "))
L = gbox[2] - gbox[1]
box = @SVector [L, L]
frame = data[10:end]
species = zeros(Int, N)
position = Vector{SVector{2,Float64}}(undef, N)
for i in eachindex(frame)
    species[i] = parse(Int, split(frame[i], " ")[1])
    position[i] = fold_back(SVector{2,Float64}(parse.(Float64, split(frame[i], " ")[2:3])), box)
end
temperature = 0.231
density = N / L^2
system = System(position, species, density, temperature, JBB())
system_ll = System(position, species, density, temperature, JBB(); list_type=LinkedList)
# GERHARD: -2.676832
@show mean(system.local_energy) / 2
@show mean(system_ll.local_energy) / 2
chains = [system]
chains_bkp = deepcopy(chains)
chains_ll = [system_ll]
chains_ll_bkp = deepcopy(chains_ll)

M = 1
seed = 10
rng = Xoshiro(seed)
sp1 = findall(isequal(1), species)
sp2 = findall(isequal(2), species)
sp3 = findall(isequal(3), species)
NA = length(sp1)
NB = length(sp2)
NC = length(sp3)
steps = 5000
burn = 0
block = [0, 1, 2, 4, 8]
sampletimes = scheduler(steps, burn, block)
callbacks = (callback_energy, callback_acceptance)

# NO SWAPS
pswap = 0.0
displacement_policy = SimpleGaussian()
displacement_parameters = ComponentArray(σ=0.05)
pools = [(
    Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1 - pswap),
) for _ in 1:M]
## Empty List
chains = deepcopy(chains_bkp)
path = "data/test/particles/KA2D_distribution/N$N/T$temperature/pswap$pswap/M$M"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, path=path)
#run!(simulation, callbacks...)
## Linked List
chains = deepcopy(chains_ll_bkp)
path = "data/test/particles/KA2D_distribution_LL/N$N/T$temperature/pswap$pswap/M$M"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, path=path)
run!(simulation, callbacks...)

# SWAPS
pswap = 0.2
displacement_policy = SimpleGaussian()
displacement_parameters = ComponentArray(σ=0.05)
swap_policy = DoubleUniform()
swap_parameters = Vector{Float64}()
pools = [(
    Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1 - pswap),
    Move(DiscreteSwap(0, 0, (1, 3), (NA, NC)), swap_policy, swap_parameters, pswap / 2),
    Move(DiscreteSwap(0, 0, (2, 3), (NB, NC)), swap_policy, swap_parameters, pswap / 2),
) for _ in 1:M]
## Empty List
chains = deepcopy(chains_bkp)
path = "data/test/particles/KA2D_distribution/N$N/T$temperature/pswap$pswap/M$M"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, path=path)
run!(simulation, callbacks...)
## Linked List
chains = deepcopy(chains_ll_bkp)
path = "data/test/particles/KA2D_distribution_LL/N$N/T$temperature/pswap$pswap/M$M"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, path=path)
run!(simulation, callbacks...)

# EB SWAPS
pswap = 0.1
displacement_policy = SimpleGaussian()
displacement_parameters = ComponentArray(σ=0.05)
swap_AC_policy = MCMC.EnergyBias()
swap_BC_policy = MCMC.EnergyBias()
swap_AC_parameters = ComponentArray(θ₁=1.0, θ₂=0.5)
swap_BC_parameters = ComponentArray(θ₁=0.5, θ₂=4.0)
pools = [(
    Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1 - pswap),
    Move(DiscreteSwap(0, 0, (1, 3), (NA, NC)), swap_AC_policy, swap_AC_parameters, pswap / 2),
    Move(DiscreteSwap(0, 0, (2, 3), (NB, NC)), swap_BC_policy, swap_BC_parameters, pswap / 2),
) for _ in 1:M]
## Empty List
chains = deepcopy(chains_bkp)
path = "data/test/particles/KA2D_distribution/N$N/T$temperature/ebswap$pswap/M$M"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, path=path)
run!(simulation, callbacks...)
## Linked List
chains = deepcopy(chains_ll_bkp)
path = "data/test/particles/KA2D_distribution_LL/N$N/T$temperature/ebswap$pswap/M$M"
simulation = Simulation(chains, pools, steps; sweepstep=N, sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, path=path)
run!(simulation, callbacks...)