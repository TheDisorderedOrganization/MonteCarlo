using MonteCarlo
using Test
using DelimitedFiles

include("../example/particle_1d/particle_1d.jl")

potential(x) = x^2
@testset "Harmonic oscillator distribution" begin
    seed = 42
    rng = Xoshiro(seed)
    M = 100
    steps = 10^6
    burn = 1000
    block = [0, 10]
    sampletimes = scheduler(steps, burn, block)
    for β in [2.0, 2.5, 3.0]
        chains = [System(4rand(rng) - 2, β) for _ in 1:M]
        pools = [(Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),) for _ in 1:M]
        path = "data/MC/particle_1d/Harmonic/beta$β/M$M/seed$seed"
        simulation = Simulation(chains, pools, steps; sampletimes=sampletimes, seed=seed, parallel=false, verbose=false, store_trajectory=true, path=path)
        callbacks = (callback_energy, callback_acceptance)
        run!(simulation, callbacks...)
        μ⁺ = 0.0
        σ⁺ = 1 / sqrt(2β)
        trj_files = [joinpath(dir, "trajectory.xyz") for dir in readdir(joinpath(path, "trajectories"), join=true)]
        trajectories = map(file -> readdlm(file)[:, 2], trj_files)
        positions = vcat(trajectories...)
        @test isapprox(mean(positions), μ⁺, atol=1e-3)
        @test isapprox(std(positions), σ⁺, atol=1e-3)
    end
end