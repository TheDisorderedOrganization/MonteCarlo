module MonteCarlo

using Random
using Distributions
using Statistics
using LinearAlgebra
using Transducers
using Dates

include("simulation.jl")
include("utils.jl")
include("metropolis.jl")

export Action, Policy, Move
export sample_action!, perform_action!, perform_action_cached!, invert_action!
export log_proposal_density, delta_log_target_density
export mc_step!, mc_sweep!
export Metropolis, callback_acceptance, StoreParameters
export build_schedule, StoreCallbacks, StoreTrajectories, StoreLastFrames, PrintTimeSteps
export Simulation, run!

include("PolicyGuided/PolicyGuided.jl")
using .PolicyGuided: Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG, reward, PolicyGradientEstimator, PolicyGradientUpdate
export Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG, reward
export PolicyGradientEstimator, PolicyGradientUpdate

end