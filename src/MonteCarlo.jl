module MonteCarlo

include("metropolis.jl")
include("utils.jl")
include("simulation.jl")

export Action, Policy, Move
export sample_action!, perform_action!, perform_action_cached!, invert_action!
export log_proposal_density, delta_log_target_density
export mc_step!, mc_sweep!
export callback_acceptance
export Simulation, MonteCarloSimulation
export scheduler, run!

include("pgmc/gradients.jl")
include("pgmc/learning.jl")
include("pgmc/pgmc_simulation.jl")

export PolicyGradient, Static, VPG, BLPG, BLAPG, NPG, ANPG, BLANPG
export PolicyGuidedMonteCarloSimulation, GradientData, pgmc_estimate, reward

end
