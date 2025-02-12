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

using Enzyme: autodiff, ReverseWithPrimal, Const, Duplicated
using Zygote: withgradient

include("pgmc/gradients.jl")
include("pgmc/learning.jl")
include("pgmc/pgmc.jl")

export Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG, reward
export PolicyGradientEstimator, PolicyGradientUpdate

end