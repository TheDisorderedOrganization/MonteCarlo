# EXAMPLE: Harmonic Oscillator
include("../particle_1d.jl")

potential(x) = x^2
###############################################################################

## RUN THE SIMULATION
seed = 42
rng = Xoshiro(seed)
β = 2.0
M = 10
chains = [Particle(4rand(rng) - 2, β) for _ in 1:M]
pools = [(Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),) for _ in 1:M]
steps = 10^5
burn = 1000
block = [0, 10]
sampletimes = scheduler(steps, burn, block)
path = "data/MC/particle_1d/Harmonic/beta$β/M$M/seed$seed"
simulation = Simulation(chains, pools, steps; sampletimes=sampletimes, seed=seed, parallel=false, verbose=true, store_trajectory=true, path=path)
callbacks = (callback_energy, callback_acceptance)
run!(simulation, callbacks...)

## PLOT RESULTS
using Plots, Statistics, Measures, DelimitedFiles
default(tickfontsize=15, guidefontsize=15, titlefontsize=15, legendfontsize=15,
    grid=false, size=(500, 500), minorticks=5)

energies = readdlm(joinpath(path, "energy.dat"))[:, 2]
@show mean(energies), std(energies)

target_density(x, β) = exp(-β * x^2) * sqrt(β / pi)
xx = LinRange(-2.0, 2.0, 1000)
target = target_density.(xx, β)
plot(xlabel="x", ylabel="p(x)", title="β=$β, M=$M", legend=:bottomright)
plot!(xx, target, lw=3, label="Target density", c=:red)

trj_files = [joinpath(dir, "trajectory.xyz") for dir in readdir(joinpath(path, "trajectories"), join=true)]
trajectories = map(file -> readdlm(file)[:, 2], trj_files)
positions = vcat(trajectories...)

stephist!(positions, normalize=:pdf, lw=3, label="Simulation", c=1)
savefig("example/particle_1d/harmonic_oscillator/density.png")


