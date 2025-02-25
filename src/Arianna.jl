"""
    module Arianna

Arianna is a flexible and extensible framework for Monte Carlo simulations. Instead of acting as a black-box simulator, it
  provides a modular structure where users define their own system and Monte Carlo "moves". 
"""
module Arianna

using Random
using Distributions
using Statistics
using LinearAlgebra
using Transducers
using Dates
using Printf

include("simulation.jl")
export Simulation, build_schedule, run!

include("algorithms.jl")
export Algorithm, StoreCallbacks, StoreTrajectories, StoreLastFrames, StoreBackups, PrintTimeSteps
export TXT, DAT

include("metropolis.jl")
export Action, Policy, Move
export sample_action!, perform_action!, perform_action_cached!, invert_action!
export log_proposal_density, delta_log_target_density
export mc_step!, mc_sweep!
export Metropolis, callback_acceptance, StoreParameters

include("PolicyGuided/PolicyGuided.jl")

end
