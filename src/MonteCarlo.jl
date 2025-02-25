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

include("metropolis.jl")
export Action, Policy, Move
export sample_action!, perform_action!, perform_action_cached!, invert_action!
export log_proposal_density, delta_log_target_density
export mc_step!, mc_sweep!
export Metropolis, callback_acceptance, StoreParameters

include("algorithms.jl")
export StoreCallbacks, StoreTrajectories, StoreLastFrames, StoreBackups, PrintTimeSteps
export TXT, DAT

include("simulation.jl")
export Simulation, build_schedule, run!

include("PolicyGuided/PolicyGuided.jl")

end