module EnzymeExt

using MonteCarlo
using Enzyme

struct Enzyme_Backend <: MonteCarlo.PolicyGuided.AD_Backend end

function MonteCarlo.PolicyGuided.withgrad_log_proposal_density!(∇logq::T, action::Action, policy::Policy, parameters::T, system, ::Enzyme_Backend;
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