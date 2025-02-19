module PolicyGuided

using ..MonteCarlo: Action, Policy, Algorithm, Simulation
import ..MonteCarlo: make_step!, write_algorithm, sample_action!, perform_action!, delta_log_target_density, log_proposal_density, invert_action!, perform_action_cached!, raise_error
using Random
using LinearAlgebra
using Transducers
using ForwardDiff

include("gradients.jl")
include("learning.jl")
include("estimator.jl")
include("update.jl")

export Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG, reward
export PolicyGradientEstimator, PolicyGradientUpdate

end