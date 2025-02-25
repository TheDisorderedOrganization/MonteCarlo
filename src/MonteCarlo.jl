"""
    module MonteCarlo

The `MonteCarlo` module provides a flexible and extensible framework for performing Monte Carlo simulations.
"""
module MonteCarlo

using Random
using Distributions
using Statistics
using LinearAlgebra
using Transducers
using Dates
using Printf

include("simulation.jl")
include("algorithms.jl")
include("metropolis.jl")

export Action, Policy, Move
export sample_action!, perform_action!, perform_action_cached!, invert_action!
export log_proposal_density, delta_log_target_density
export mc_step!, mc_sweep!
export Metropolis, callback_acceptance, StoreParameters
export build_schedule, StoreCallbacks, StoreTrajectories, StoreLastFrames, StoreBackups, PrintTimeSteps
export Simulation, run!
export TXT, DAT

include("PolicyGuided/PolicyGuided.jl")

end