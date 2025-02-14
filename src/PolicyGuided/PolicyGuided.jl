module PolicyGuided

using ..MonteCarlo: Action, Policy, Algorithm, Simulation
import ..MonteCarlo: make_step!, sample_action!, perform_action!, delta_log_target_density, log_proposal_density, invert_action!, perform_action_cached!
using Random
using LinearAlgebra
using Transducers
using Enzyme: autodiff, ReverseWithPrimal, Const, Duplicated
using Zygote: withgradient

include("gradients.jl")
include("learning.jl")
include("estimator.jl")
include("update.jl")

export Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG, reward
export PolicyGradientEstimator, PolicyGradientUpdate

end