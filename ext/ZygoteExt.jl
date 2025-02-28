module ZygoteExt

using Arianna
using Zygote

struct Zygote_Backend <: Arianna.PolicyGuided.AD_Backend end

function Arianna.PolicyGuided.withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system::Arianna.AriannaSystem, ::Zygote_Backend;
    shadow=missing) where {T<:AbstractArray}
    logq, gd = Zygote.withgradient(x -> log_proposal_density(action, policy, x, system), parameters)
    ∇logq .= gd[1]
    return logq
end

end