module MonteCarlo

abstract type Action end
abstract type Policy end

export Action, Policy

include("metropolis.jl")
include("utils.jl")
include("simulation.jl")
include("pgmc/gradients.jl")
include("pgmc/learning.jl")
include("pgmc/pgmc_simulation.jl")


export sample_action!,perform_action!, perform_action_cached!, invert_action!
export log_proposal_density, delta_log_target_density 
export callback_acceptance,  callback_temperature
export Simulation, MonteCarloSimulation, PolicyGuidedMonteCarloSimulation
export scheduler, store_trajectory, write_summary, save_data, run!, InitialiseSummary
export PolicyGradient, Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG
export GradientData, initialise_gradient_data, sample_gradient_data, pgmc_estimate, AD_Backend, reward
export Move, mc_step!, mc_sweep!

end
