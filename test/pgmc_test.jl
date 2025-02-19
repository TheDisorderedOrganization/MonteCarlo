using MonteCarlo
using MonteCarlo.PolicyGuided
using Test
using DelimitedFiles

include("../example/particle_1d/particle_1d.jl")

potential(x) = x^2

@testset "Displacement optimisation" begin
    seed = 42
    rng = Xoshiro(seed)
    β = 2.0
    M = 10
    chains = [System(4rand(rng) - 2, β) for _ in 1:M]
    σ₀ = 0.2
    pools = [(
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.2),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.2),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.2),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.2),
        Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=σ₀), 0.2),
    ) for _ in 1:M]
    optimisers = (Static(), VPG(0.001), BLPG(0.001), BLAPG(1e-6, 1e-6), NPG(1e-2, 1e-6))
    steps = 10^5
    burn = 1000
    block = [0, 10]
    sampletimes = build_schedule(steps, burn, block)
    path = "data/PGMC/particle_1d/Harmonic/beta$β/M$M/seed$seed"
    algorithm_list = (
        (algorithm=Metropolis, pools=pools, seed=seed, parallel=false),
        (algorithm=PolicyGradientEstimator, pools=pools, optimisers=optimisers, q_batch_size=10, seed=seed, parallel=true),
        (algorithm=PolicyGradientUpdate, dependencies=(PolicyGradientEstimator,), scheduler=build_schedule(steps, burn, 2)),
        (algorithm=StoreCallbacks, callbacks=(callback_energy, callback_acceptance), scheduler=sampletimes),
        (algorithm=StoreTrajectories, scheduler=sampletimes),
        (algorithm=StoreParameters, pools=pools, scheduler=sampletimes),
        (algorithm=StoreLastFrames, scheduler=[steps]),
        (algorithm=PrintTimeSteps, scheduler=build_schedule(steps, burn, steps ÷ 10)),
    )
    simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
    run!(simulation)
    energies = readdlm(joinpath(path, "energy.dat"))[:, 2]
    @test isapprox(mean(energies), 0.25, atol=5e-2)
    prms_path = joinpath(path, "parameters")
    for (opt, dir) in zip(optimisers, readdir(prms_path))
        prms_data = readlines(joinpath(prms_path, dir, "parameters.dat"))
        sigmas = parse.(Float64, replace.(getindex.(split.(prms_data, " "), 2), r"\[|\]" => ""))
        @test (isa(opt, Static) && sigmas[end] == σ₀) || isapprox(sigmas[end], 1.2, atol=2e-1)
    end
end