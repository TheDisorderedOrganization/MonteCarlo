module EnzymeExt

using Arianna
using Enzyme

struct Enzyme_Backend <: Arianna.PolicyGuided.AD_Backend end

function Arianna.PolicyGuided.withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system::Arianna.AriannaSystem, ::Enzyme_Backend;
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

end