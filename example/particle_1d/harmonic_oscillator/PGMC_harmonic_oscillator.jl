# EXAMPLE: Harmonic Oscillator
include("../particle_1d.jl")

potential(x) = x^2
###############################################################################

## RUN THE SIMULATION
seed = 42
rng = Xoshiro(seed)
β = 2.0
M = 10
chains = [System(4rand(rng) - 2, β) for _ in 1:M]
pools = [(
    Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.2), 0.6),
    Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 0.4),
) for _ in 1:M]
optimisers = (Static(), VPG(0.001))
steps = 10^5
burn = 1000
block = [0, 10]
sampletimes = build_schedule(steps, burn, block)
path = "data/PGMC/particle_1d/Harmonic/beta$β/M$M/seed$seed"

metropolis = Metropolis(chains, pools; seed=seed, parallel=false)
pge = PolicyGradientEstimator(chains, pools, optimisers)
pgu = PolicyGradientUpdate(chains, pge)
learn_ids = [k for k in eachindex(optimisers) if !isa(optimisers[k], Static)]

algorithms = (
    metropolis,
    pge,
    pgu,
    StoreCallbacks((callback_energy, callback_acceptance), path),
    StoreTrajectories(chains, path),
    StoreLastFrames(chains, path),
    PrintTimeSteps(),
    StoreParameters(pools[1], path; ids=learn_ids),
)
schedulers = [build_schedule(steps, 0, 1), build_schedule(steps, 0, 1), build_schedule(steps, 0, 2), sampletimes, sampletimes, [0, steps], build_schedule(steps, burn, steps ÷ 10), sampletimes]
simulation = Simulation(chains, algorithms, steps; schedulers=schedulers, path=path, verbose=true)

run!(simulation)


## PLOT RESULTS
using Plots, Statistics, Measures, DelimitedFiles
default(tickfontsize=15, guidefontsize=15, titlefontsize=15, legendfontsize=15,
    grid=false, size=(500, 500), minorticks=5)

energies = readdlm(joinpath(path, "energy.dat"))[:, 2]
@show mean(energies), std(energies)

prms_data = readlines(joinpath(path, "parameters", "2", "parameters.dat"))
steps_data = parse.(Int, getindex.(split.(prms_data, " "), 1))
time_steps = steps_data .- steps_data[1]
prms = parse.(Float64, replace.(getindex.(split.(prms_data, " "), 2), r"\[|\]" => ""))
plot(xlabel="t", ylabel="σ(t)", xscale=:log10, legend=false, title="β=$β, M=$M, η=$(optimisers[2].η)")
plot!(time_steps[2:end], prms[2:end], lw=2)
savefig("example/particle_1d/harmonic_oscillator/learning.png")