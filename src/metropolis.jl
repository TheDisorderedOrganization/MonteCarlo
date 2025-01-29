using Random, Distributions, Statistics, LinearAlgebra


sample_action!(action::Action, policy::Policy, parameters, system, rng) = println("Error: No sample_action! is defined")
log_proposal_density(action, policy, parameters, system) = println("Error: No log_proposal_density is defined")
perform_action!(system, action::Action) = println("Error: No perform_action! is defined")
delta_log_target_density(x1, x2, system) = println("Error: No delta_log_target_density is defined")
invert_action!(action::Action, system) = println("Error: No invert_action is defined")
perform_action_cached!(system, action::Action) = perform_action!(system, action)

mutable struct Move{A<:Action,P<:Policy,V<:AbstractArray,T<:AbstractFloat}
    action::A
    policy::P
    parameters::V
    weight::T
    total_calls::Int
    accepted_calls::Int
end

function Move(action, policy, parameters, weight)
    return Move(action, policy, parameters, weight, 0, 0)
end

function mc_step!(system, action::Action, policy::Policy, parameters::AbstractArray{T}, rng) where {T<:AbstractFloat}
    sample_action!(action, policy, parameters, system, rng)
    logq_forward = log_proposal_density(action, policy, parameters, system)
    x₁, x₂ = perform_action!(system, action)
    Δlogp = delta_log_target_density(x₁, x₂, system)
    invert_action!(action, system)
    logq_backward = log_proposal_density(action, policy, parameters, system)
    α = min(one(T), exp(Δlogp + logq_backward - logq_forward))
    if α > rand(rng)
        return 1
    else
        perform_action_cached!(system, action)
        return 0
    end
end

function mc_sweep!(system, pool, rng; mc_steps=1)
    weights = [move.weight for move in pool]
    for _ in 1:mc_steps
        id = rand(rng, Categorical(weights))
        move = pool[id]
        move.accepted_calls += mc_step!(system, move.action, move.policy, move.parameters, rng)
        move.total_calls += 1
    end
    return nothing
end

nothing