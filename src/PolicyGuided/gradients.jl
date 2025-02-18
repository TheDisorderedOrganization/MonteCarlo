abstract type AD_Backend end

struct Enzyme_Backend <: AD_Backend end

struct Zygote_Backend <: AD_Backend end

struct ForwardDiff_Backend <: AD_Backend end

reward(action::Action, system) = MonteCarlo.raise_error("reward")

function withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system, ::Enzyme_Backend;
    shadow=deepcopy(system)) where {T<:AbstractArray}
    _, logq = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        log_proposal_density,
        Enzyme.Const(action),
        Enzyme.Const(policy),
        Enzyme.Duplicated(parameters, ∇logq),
        Enzyme.Duplicated(system, shadow)
    )
    return logq
end

function withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system, ::Zygote_Backend;
    shadow=deepcopy(system)) where {T<:AbstractArray}
    logq, gd = Zygote.withgradient(x -> log_proposal_density(action, policy, x, system), parameters)
    ∇logq .= gd[1]
    return logq
end

function withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system, ::ForwardDiff_Backend;
    shadow=deepcopy(system)) where {T<:AbstractArray}
    logq = log_proposal_density(action, policy, parameters, system)
    ∇logq .= ForwardDiff.gradient(p -> log_proposal_density(action, policy, p, system), parameters)
    return logq
end

struct GradientData{T<:AbstractFloat, V<:AbstractArray, V2<:AbstractArray}
    j::T                # Objective function
    ∇j::V               # Gradient of the objective function
    ∇logq_forward::V    # Gradient of the log proposal density of the forward action
    g::V2               # Metric tensor for natural policy gradient
    n::Int              # Counter for averaging
end

function initialise_gradient_data(parameters::AbstractArray)
    j = zero(eltype(parameters))
    ∇j = zero(parameters)
    ∇logq_forward = zero(parameters)
    g = ∇j * ∇j'
    n = 0
    return GradientData(j, ∇j, ∇logq_forward, g, n)
end

function Base.:+(gd1::GradientData, gd2::GradientData)
    return GradientData(
        gd1.j + gd2.j,
        gd1.∇j + gd2.∇j,
        gd1.∇logq_forward + gd2.∇logq_forward,
        gd1.g + gd2.g,
        gd1.n + gd2.n
        )
end

function average(gd::GradientData)
    return GradientData(gd.j / gd.n, gd.∇j ./ gd.n, gd.∇logq_forward ./ gd.n, gd.g ./ gd.n, gd.n)
end

function pgmc_estimate(action::Action, policy::Policy, parameters::AbstractArray{T}, system;
    ∇logq_forward=zero(parameters), ∇logq_backward=zero(parameters), shadow=deepcopy(system), ad_backend::AD_Backend=Enzyme_Backend()) where {T<:AbstractFloat}
    ∇logq_forward .= zero(parameters)
    ∇logq_backward .= zero(parameters)
    logq_forward = withgrad_log_proposal_density!(∇logq_forward, action, policy, parameters, system, ad_backend; shadow=shadow)
    x₁, x₂ = perform_action!(system, action)
    Δlogp = delta_log_target_density(x₁, x₂, system)
    r = reward(action, system)
    invert_action!(action, system)
    logq_backward = withgrad_log_proposal_density!(∇logq_backward, action, policy, parameters, system, ad_backend; shadow=shadow)
    perform_action_cached!(system, action)
    α = min(one(T), exp(Δlogp + logq_backward - logq_forward))
    j = r * α
    ∇j = j .* (isone(α) ? ∇logq_forward : ∇logq_backward)
    g = ∇logq_forward * ∇logq_forward'
    return GradientData(j, ∇j, ∇logq_forward, g, 1)
end

function sample_gradient_data(action::Action, policy::Policy, parameters::AbstractArray, system, rng;
    ∇logq_forward=zero(parameters), ∇logq_backward=zero(parameters), shadow=deepcopy(system), ad_backend::AD_Backend=Enzyme_Backend())
    sample_action!(action, policy, parameters, system, rng)
    return pgmc_estimate(action, policy, parameters, system; ∇logq_forward=∇logq_forward, ∇logq_backward=∇logq_backward, shadow=shadow, ad_backend=ad_backend)
end

nothing