using ConcreteStructs, StatsBase, Transducers, Distributions

abstract type Simulation end

@concrete mutable struct MonteCarloSimulation <: Simulation
    chains                      # Vector of independent systems
    pools                       # Vector of independent pools (one for each system)
    steps::Int                  # Number of MC sweeps
    sampletimes::Vector{Int}    # Time steps at which we store data 
    sweepstep::Int              # Number of mc steps per mc sweep
    path::String                # Simulation path
    seed::Int                   # Random number seed
    store_trajectory::Bool      # Flag to store trajectories at each measurement
    parallel::Bool              # Flag to parallelise over different threads
    verbose::Bool               # Flag for verbose
end

function Simulation(
    chains,
    pools,
    steps::Int;
    sampletimes::Vector{Int}=scheduler(steps, 0, 1),
    sweepstep::Int=1,
    path::AbstractString="data",
    seed::Int=1,
    store_trajectory::Bool=false,
    parallel::Bool=false,
    verbose::Bool=false
)
    @assert length(chains) == length(pools)
    @assert all(k -> all(move -> move.parameters == getindex.(pools, k)[1].parameters, getindex.(pools, k)), eachindex(pools[1]))
    @assert all(k -> all(move -> move.weight == getindex.(pools, k)[1].weight, getindex.(pools, k)), eachindex(pools[1]))
    return MonteCarloSimulation(chains, pools, steps, sampletimes, sweepstep, path, seed, store_trajectory, parallel, verbose)
end

function save_data(trj_files, cb_files, callbacks, t::Int, simulation)
    simulation.store_trajectory && for c in eachindex(chains)
        store_trajectory(trj_files[c], simulation.chains[c], t)
    end
    for k in eachindex(cb_files)
        println(cb_files[k], "$t $(callbacks[k](simulation))")
    end
    return nothing
end

function callback_acceptance(simulation)
    return mean([[move.accepted_calls / move.total_calls for move in pool] for pool in simulation.pools])
end

function run!(simulation::MonteCarloSimulation, callbacks...)
    # INISIALISATION
    simulation.verbose && println("INISIALISATION")
    ## Define random number generator
    seeds = [simulation.seed + c - 1 for c in eachindex(simulation.chains)]
    rngs = [Xoshiro(s) for s in seeds]
    ## Define transducers reducer and collecter
    reducer = simulation.parallel ? Transducers.foldxt : Transducers.foldxl
    collecter = simulation.parallel ? Transducers.tcollect : collect
    ## Create simulation path
    mkpath(simulation.path)
    ## Initialise summary
    write_summary(simulation, InitialiseSummary())
    ## Burn initial configurations
    simulation.verbose && println("Burn-in...")
    burn_time = @elapsed collecter(
        eachindex(simulation.chains) |> Map(c -> begin
            for t in 1:simulation.sampletimes[1]
                mc_sweep!(simulation.chains[c], simulation.pools[c], rngs[c]; mc_steps=simulation.sweepstep)
            end
        end)
    )
    simulation.verbose && println("Burn-in completed in $burn_time s")
    ## Update summary with burn-in time
    write_summary(simulation, UpdateBurnTime(burn_time))
    ## Create files for trajectories and callbacks
    simulation.verbose && println("Opening files...")
    trj_paths = joinpath.(simulation.path, "trajectories", ["$c" for c in eachindex(simulation.chains)])
    simulation.store_trajectory && mkpath.(trj_paths)
    trj_files = simulation.store_trajectory ? open.(joinpath.(trj_paths, "trajectory.xyz"), "w") : nothing
    simulation.verbose && simulation.store_trajectory && println("$(length(trj_files)) trajectory files created")
    cb_paths = joinpath.(simulation.path, [replace(string(cb), "callback_" => "") * ".dat" for cb in callbacks])
    cb_files = open.(cb_paths, "w")
    simulation.verbose && println("$(length(cb_files)) callback files created")
    try
        ## Initial measurement
        save_data(trj_files, cb_files, callbacks, simulation.sampletimes[1], simulation)
        ## Initialise schedulers 
        n = 2
        # MAIN LOOP
        simulation.verbose && println("RUN...")
        sim_time = @elapsed for t in simulation.sampletimes[1]+1:simulation.steps+simulation.sampletimes[1]
            ## One mc sweep for each chain
            collecter(
                eachindex(simulation.chains) |> Map(c -> begin
                    mc_sweep!(simulation.chains[c], simulation.pools[c], rngs[c]; mc_steps=simulation.sweepstep)
                end)
            )
            ## Save data when scheduled
            if t == simulation.sampletimes[n]
                simulation.verbose && println("t = $t")
                save_data(trj_files, cb_files, callbacks, t, simulation)
                n += 1
            end
        end
        ## Update summary
        simulation.verbose && println("Simulation completed in $sim_time s")
        write_summary(simulation, UpdateSimTime(sim_time))
    finally
        ## Make sure to close all files
        simulation.store_trajectory && close.(trj_files)
        close.(cb_files)
        # Save last snapshots
        simulation.store_trajectory && for c in eachindex(simulation.chains)
            open(joinpath(trj_paths[c], "lastframe.xyz"), "w") do trj
                store_trajectory(trj, simulation.chains[c], simulation.steps)
            end
        end
        ## Finalise
        simulation.verbose && println("DONE")
        write_summary(simulation, FinalReport())
    end
end

nothing
