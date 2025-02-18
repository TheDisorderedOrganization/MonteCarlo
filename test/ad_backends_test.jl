using MonteCarlo
using Test

include("../example/particle_1d/particle_1d.jl")

potential(x) = x^2

seed = 42
rng = Xoshiro(seed)
β = 2.0
system = System(4rand(rng) - 2, β)
action = Displacement(0.0)
policy = StandardGaussian()
parameters = ComponentArray(σ=0.2)

∇logq_Zygote = zero(parameters)
∇logq_Enzyme = zero(parameters)
∇logq_FD = zero(parameters)

logq_Zygote = MonteCarlo.PolicyGuided.withgrad_log_proposal_density!(∇logq_Zygote, action, policy, parameters, system, MonteCarlo.PolicyGuided.Zygote_Backend(); shadow=deepcopy(system))
logq_Enzyme = MonteCarlo.PolicyGuided.withgrad_log_proposal_density!(∇logq_Enzyme, action, policy, parameters, system, MonteCarlo.PolicyGuided.Enzyme_Backend(); shadow=deepcopy(system))
logq_FD = MonteCarlo.PolicyGuided.withgrad_log_proposal_density!(∇logq_FD, action, policy, parameters, system, MonteCarlo.PolicyGuided.ForwardDiff_Backend(); shadow=deepcopy(system))

@test isapprox(logq_Zygote, logq_Enzyme, atol=1e-10) && isapprox(logq_Zygote, logq_FD, atol=1e-10)
@test isapprox(∇logq_Zygote, ∇logq_Enzyme, atol=1e-10) && isapprox(∇logq_Zygote, ∇logq_FD, atol=1e-10)

