struct PolicyGradientEstimator{P,O,VPL<:AbstractArray,VPR<:AbstractArray,VO<:AbstractArray,VG<:AbstractArray,VC<:AbstractArray,VL<:AbstractArray,ADB<:AD_Backend,R<:AbstractRNG,C<:Function} <: Algorithm
    pools::Vector{P}            # Vector of independent pools (one for each system)
    optimisers::O               # List of optimisers (one for each move)
    learn_ids::Vector{Int}      # List of learnable moves
    q_batch_size::Int           # Number of independent samples generated from proposal distributions
    policy_list::VPL            # List of policies (one for each move)
    parameters_list::VPR        # List of current parameters values (one array for each move)
    objectives::VO              # Cache for estimated objective functions Ĵ (one for each move)
    gradients_data::VG          # Gradient information (one for each move)
    chains_shadow::VC           # Copy of chains (for Enzyme)
    ∇logqs_forward::VL          # Preallocated forward gradients (one array for each move)
    ∇logqs_backward::VL         # Preallocated backward gradients (one array for each move)
    ad_backend::ADB             # Backend for automatic differentiation (Enzyme or Zygote)
    seed::Int                   # Random number seed
    rngs::Vector{R}             # Vector of random number generators
    parallel::Bool              # Flag to parallelise over different threads
    reducer::C                  # Transducer to reduce results from parallelised loops

    function PolicyGradientEstimator(
        chains::Vector{S},
        pools::Vector{P},
        optimisers::O;
        q_batch_size::Int=1,
        ad_backend::AD_Backend=Enzyme_Backend(),
        seed::Int=1,
        R::DataType=Xoshiro,
        parallel::Bool=false
    ) where {S,P,O}
        # Safety checks
        @assert length(chains) == length(pools)
        @assert all(k -> all(move -> move.parameters == getindex.(pools, k)[1].parameters, getindex.(pools, k)), eachindex(pools[1]))
        @assert all(k -> all(move -> move.weight == getindex.(pools, k)[1].weight, getindex.(pools, k)), eachindex(pools[1]))
        @assert length(optimisers) == length(pools[1])
        # Find learnable actions
        learn_ids = [k for k in eachindex(optimisers) if !isa(optimisers[k], Static)]
        #Make sure that all policies and parameters across chains refer to the same objects
        policy_list = [move.policy for move in pools[1]]
        parameters_list = [move.parameters for move in pools[1]]
        for pool in pools
            for k in eachindex(policy_list)
                pool[k].policy = policy_list[k]
                pool[k].parameters = parameters_list[k]
            end
        end
        # Create objectives and gradients caches
        objectives = zeros(eltype(pools[1][1].parameters), length(learn_ids))
        gradients_data = map(k -> initialise_gradient_data(parameters_list[k]), learn_ids)
        # Create shadows for Enzyme
        chains_shadow = deepcopy(chains)
        ∇logqs_forward = map(zero, parameters_list)
        ∇logqs_backward = map(zero, parameters_list)
        # Handle randomness
        seeds = [seed + c - 1 for c in eachindex(chains)]
        rngs = [R(s) for s in seeds]
        # Handle parallelism
        reducer = parallel ? Transducers.foldxt : Transducers.foldxl
        return new{P,O,typeof(policy_list),typeof(parameters_list),typeof(objectives),typeof(gradients_data),typeof(chains_shadow),typeof(∇logqs_forward),typeof(ad_backend),R,typeof(reducer)}(
            pools, optimisers, learn_ids, q_batch_size, policy_list, parameters_list, objectives, gradients_data,
            chains_shadow, ∇logqs_forward, ∇logqs_backward, ad_backend, seed, rngs, parallel, reducer
            )
    end

end

function make_step!(simulation::Simulation, algorithm::PolicyGradientEstimator)
    for (k, lid) in enumerate(algorithm.learn_ids)
        gd = algorithm.reducer(+,
            eachindex(simulation.chains) |> Map(c -> begin
                1:algorithm.q_batch_size |> Map(_ -> begin
                    sample_gradient_data(
                        algorithm.pools[c][lid].action,
                        algorithm.policy_list[lid],
                        algorithm.parameters_list[lid],
                        simulation.chains[c],
                        algorithm.rngs[c];
                        ∇logq_forward=algorithm.∇logqs_forward[lid],
                        ∇logq_backward=algorithm.∇logqs_backward[lid],
                        shadow=algorithm.chains_shadow[c],
                        ad_backend=algorithm.ad_backend
                        )
                end)
            end) |> Cat()
        )
        algorithm.gradients_data[k] = algorithm.gradients_data[k] + gd
        algorithm.objectives[k] = algorithm.gradients_data[k].j / algorithm.gradients_data[k].n
    end
    return nothing
end

function write_algorithm(io, algorithm::PolicyGradientEstimator, scheduler)
    println(io, "\tPolicyGradientEstimator")
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
    println(io, "\t\tLearnable moves: $(algorithm.learn_ids)")
    println(io, "\t\tQ batch size: $(algorithm.q_batch_size)")
    println(io, "\t\tAD backend: $(algorithm.ad_backend)")
    println(io, "\t\tSeed: $(algorithm.seed)")
    println(io, "\t\tParallel: $(algorithm.parallel)")
    if algorithm.parallel
        println(io, "\t\tThreads: $(Threads.nthreads())")
    end
end

struct PolicyGradientUpdate{P,O,VPR<:AbstractArray,VG<:AbstractArray} <: Algorithm
    pools::Vector{P}            # Vector of independent pools (one for each system)
    optimisers::O               # List of optimisers (one for each move)
    learn_ids::Vector{Int}      # List of learnable moves
    parameters_list::VPR        # List of current parameters values (one array for each move)
    gradients_data::VG          # Gradient information (one for each move)

    function PolicyGradientUpdate(chains::Vector{S}, pools::Vector{P}, optimisers::O, gradients_data::VG) where {S,P,O,VG}
        # Safety checks
        @assert length(chains) == length(pools)
        @assert all(k -> all(move -> move.parameters == getindex.(pools, k)[1].parameters, getindex.(pools, k)), eachindex(pools[1]))
        @assert all(k -> all(move -> move.weight == getindex.(pools, k)[1].weight, getindex.(pools, k)), eachindex(pools[1]))
        @assert length(optimisers) == length(pools[1])
        # Find learnable actions
        learn_ids = [k for k in eachindex(optimisers) if !isa(optimisers[k], Static)]
        #Make sure that all policies and parameters across chains refer to the same objects
        policy_list = [move.policy for move in pools[1]]
        parameters_list = [move.parameters for move in pools[1]]
        for pool in pools
            for k in eachindex(policy_list)
                pool[k].policy = policy_list[k]
                pool[k].parameters = parameters_list[k]
            end
        end
        return new{P,O,typeof(parameters_list),VG}(pools, optimisers, learn_ids, parameters_list,  gradients_data)
    end

end

function PolicyGradientUpdate(chains, pge::PolicyGradientEstimator)
    return PolicyGradientUpdate(chains, pge.pools, pge.optimisers, pge.gradients_data)
end

function make_step!(::Simulation, algorithm::PolicyGradientUpdate)
    for (k, lid) in enumerate(algorithm.learn_ids)
        gd = average(algorithm.gradients_data[k])
        learning_step!(algorithm.parameters_list[lid], gd, algorithm.optimisers[lid])
        algorithm.gradients_data[k] = initialise_gradient_data(algorithm.parameters_list[lid])
    end
    return nothing
end

function write_algorithm(io, algorithm::PolicyGradientUpdate, scheduler)
    println(io, "\tPolicyGradientUpdate")
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
    println(io, "\t\tLearnable moves: $(algorithm.learn_ids)")
    println(io, "\t\tOptimisers:")
    for (k, opt) in enumerate(algorithm.optimisers)
        println(io, "\t\t\tMove $k: " * replace(string(opt), r"\{[^\{\}]*\}" => ""))
    end
end


# if t == simulation.learning_scheduler[nl]
#                 for (k, lid) in enumerate(simulation.learn_ids)
#                     gd = average(simulation.gradients_data[k])
#                     learning_step!(simulation.parameters_list[lid], gd, simulation.optimisers[lid])
#                     simulation.gradients_data[k] = initialise_gradient_data(simulation.parameters_list[lid])
#                 end
#                 nl += 1
#             end

# @concrete mutable struct PolicyGuidedMonteCarloSimulation <: Simulation
#     chains                  # Vector of independent systems
#     pools                   # Vector of independent pools (one for each system)
#     optimisers              # List of optimisers (one for each move)
#     steps                   # Number of MC sweeps
#     sampletimes             # Time steps at which we store data 
#     sweepstep               # Number of mc steps per mc sweep
#     learn_ids               # List of learnable moves
#     q_batch_size            # Number of independent samples generated from proposal distributions
#     sweeps_per_gradient     # Number of mc sweeps per gradient sampling
#     sweeps_per_learning     # Number of mc sweeps per learning step
#     sampling_scheduler      # Time steps at which we sample gradients
#     learning_scheduler      # Time steps at which we update parameters
#     policy_list             # List of policies (one for each move)
#     parameters_list         # List of current parameters values (one array for each move)
#     objectives              # Cache for estimated objective functions Ĵ (one for each move)
#     gradients_data          # Gradient information (one for each move)
#     chains_shadow           # Copy of chains (for Enzyme)
#     ∇logqs_forward          # Preallocated forward gradients (one array for each move)
#     ∇logqs_backward         # Preallocated backward gradients (one array for each move)
#     ad_backend              # Backend for automatic differentiation (Enzyme or Zygote)
#     path                    # Simulation path
#     seed                    # Random number seed
#     store_trajectory        # Flag to store trajectories at each measurement
#     store_parameters        # Flag to store parameters at each measurement
#     parallel                # Flag to parallelise over different threads
#     verbose                 # Flag for verbose
# end

# function Simulation(
#     chains,
#     pools,
#     optimisers,
#     steps::Int;
#     sampletimes::Vector{Int}=scheduler(steps, 0, 1),
#     sweepstep::Int=1,
#     q_batch_size::Int=1,
#     sweeps_per_gradient::Int=1,
#     sweeps_per_learning::Int=1,
#     ad_backend::AD_Backend=Enzyme_Backend(),
#     path::AbstractString="data",
#     seed::Int=1,
#     store_trajectory::Bool=false,
#     store_parameters::Bool=false,
#     parallel::Bool=false,
#     verbose::Bool=false
# )
#     # Safety checks
#     @assert length(chains) == length(pools)
#     @assert all(k -> all(move -> move.parameters == getindex.(pools, k)[1].parameters, getindex.(pools, k)), eachindex(pools[1]))
#     @assert all(k -> all(move -> move.policy == getindex.(pools, k)[1].policy, getindex.(pools, k)), eachindex(pools[1]))
#     @assert all(k -> all(move -> move.weight == getindex.(pools, k)[1].weight, getindex.(pools, k)), eachindex(pools[1]))
#     # Find learnable actions
#     learn_ids = [k for k in eachindex(optimisers) if !isa(optimisers[k], Static)]
#     # Define schedulers
#     sampling_scheduler = scheduler(steps, sampletimes[1], sweeps_per_gradient)
#     learning_scheduler = scheduler(steps, sampletimes[1], sweeps_per_learning)
#     # Make sure that all policies and parameters across chains refer to the same objects
#     policy_list = [move.policy for move in pools[1]]
#     parameters_list = [move.parameters for move in pools[1]]
#     for pool in pools
#         for k in eachindex(policy_list)
#             pool[k].policy = policy_list[k]
#             pool[k].parameters = parameters_list[k]
#         end
#     end
#     # Create objectives and gradients caches
#     objectives = zeros(eltype(pools[1][1].parameters), length(learn_ids))
#     gradients_data = map(k -> initialise_gradient_data(parameters_list[k]), learn_ids)
#     # Create shadows for Enzyme
#     chains_shadow = deepcopy(chains)
#     ∇logqs_forward = map(zero, parameters_list)
#     ∇logqs_backward = map(zero, parameters_list)
#     # Return simulation
#     return PolicyGuidedMonteCarloSimulation(chains, pools, optimisers, steps, sampletimes, sweepstep,
#         learn_ids, q_batch_size, sweeps_per_gradient, sweeps_per_learning, sampling_scheduler, learning_scheduler,
#         policy_list, parameters_list, objectives, gradients_data, chains_shadow, ∇logqs_forward, ∇logqs_backward, ad_backend,
#         path, seed, store_trajectory, store_parameters, parallel, verbose)
# end

# function write_summary(simulation::PolicyGuidedMonteCarloSimulation, ::InitialiseSummary)
#     open(joinpath(simulation.path, "summary.log"), "w") do file
#         println(file, "POLICY-GUIDED MONTE CARLO SIMULATION")
#         println(file)
#         println(file, "System:")
#         write_system(file, simulation.chains[1])
#         println(file)
#         println(file, "Moves:")
#         for (k, move) in enumerate(simulation.pools[1])
#             println(file, "\tMove $k:")
#             println(file, "\t\tAction: " * replace(string(typeof(move.action)), r"\{.*" => ""))
#             println(file, "\t\tPolicy: " * replace(string(typeof(move.policy)), r"\{.*" => ""))
#             println(file, "\t\tParameters: " * write_parameters(move.policy, move.parameters))
#             println(file, "\t\tWeight: $(move.weight)")
#             println(file, "\t\tLearnable: $(!isa(simulation.optimisers[k], Static))")
#             if !isa(simulation.optimisers[k], Static)
#                 println(file, "\t\tOptimiser: " * replace(string(simulation.optimisers[k]), r"\{[^\{\}]*\}" => ""))
#             end
#         end
#         println(file)
#         println(file, "Simulation:")
#         println(file, "\tSeed: $(simulation.seed)")
#         println(file, "\tNumber of chains: $(length(simulation.chains))")
#         println(file, "\tMC sweeps: $(simulation.steps)")
#         println(file, "\tMC steps per MC sweep: $(simulation.sweepstep)")
#         println(file, "\tBurn-in sweeps: $(simulation.sampletimes[1])")
#         println(file, "\tNumber of measurements: $(length(simulation.sampletimes) - (simulation.sampletimes[end] > simulation.steps))")
#         println(file, "\tMC sweeps per samplig step: $(simulation.sweeps_per_gradient)")
#         println(file, "\tMC sweeps per learning step: $(simulation.sweeps_per_learning)")
#         println(file, "\tQ batch size: $(simulation.q_batch_size)")
#         println(file, "\tP batch size: $(Int(fld(simulation.sweeps_per_learning * length(simulation.chains), simulation.sweeps_per_gradient)))")
#         println(file, "\tAD backend: $(typeof(simulation.ad_backend))")
#         println(file, "\tStore trajectory: $(simulation.store_trajectory)")
#         println(file, "\tStore parameters: $(simulation.store_trajectory)")
#         println(file, "\tParallel: $(simulation.parallel)")
#         if simulation.parallel
#             println(file, "\tThreads: $(Threads.nthreads())")
#         end
#         println(file, "\tVerbose: $(simulation.verbose)")
#     end
# end

# function save_data(trj_files, prms_files, cb_files, callbacks, t::Int, simulation::PolicyGuidedMonteCarloSimulation)
#     simulation.store_trajectory && for c in eachindex(simulation.chains)
#         store_trajectory(trj_files[c], simulation.chains[c], t)
#     end
#     simulation.store_parameters && for k in eachindex(simulation.learn_ids)
#         println(prms_files[k], "$t $(collect(simulation.parameters_list[simulation.learn_ids[k]]))")
#     end
#     for k in eachindex(cb_files)
#         println(cb_files[k], "$t $(callbacks[k](simulation))")
#     end
#     return nothing
# end

# function run!(simulation::PolicyGuidedMonteCarloSimulation, callbacks...)
#     # INISIALISATION
#     simulation.verbose && println("INISIALISATION")
#     ## Define random number generator
#     seeds = [simulation.seed + c - 1 for c in eachindex(simulation.chains)]
#     rngs = [Xoshiro(s) for s in seeds]
#     ## Define transducers reducer and collecter
#     reducer = simulation.parallel ? Transducers.foldxt : Transducers.foldxl
#     collecter = simulation.parallel ? Transducers.tcollect : collect
#     ## Create simulation path
#     mkpath(simulation.path)
#     ## Initialise summary
#     write_summary(simulation, InitialiseSummary())
#     ## Burn initial configurations
#     simulation.verbose && println("Burn-in...")
#     burn_time = @elapsed collecter(
#         eachindex(simulation.chains) |> Map(c -> begin
#             for t in 1:simulation.sampletimes[1]
#                 mc_sweep!(simulation.chains[c], simulation.pools[c], rngs[c]; mc_steps=simulation.sweepstep)
#             end
#         end)
#     )
#     simulation.verbose && println("Burn-in completed in $burn_time s")
#     ## Update summary with burn-in time
#     write_summary(simulation, UpdateBurnTime(burn_time))
#     ## Create files for trajectories, parameters and callbacks
#     simulation.verbose && println("Opening files...")
#     trj_paths = joinpath.(simulation.path, "trajectories", ["$c" for c in eachindex(simulation.chains)])
#     mkpath.(trj_paths)
#     trj_files = simulation.store_trajectory ? open.(joinpath.(trj_paths, "trajectory.xyz"), "w") : nothing
#     simulation.verbose && simulation.store_trajectory && println("$(length(trj_files)) trajectory files created")
#     prms_paths = joinpath.(simulation.path, "parameters", ["$k" for k in simulation.learn_ids])
#     simulation.store_parameters && mkpath.(prms_paths)
#     prms_files = simulation.store_parameters ? open.(joinpath.(prms_paths, "parameters.dat"), "w") : nothing
#     simulation.verbose && simulation.store_parameters && println("$(length(prms_files)) parameters files created")
#     cb_paths = joinpath.(simulation.path, [replace(string(cb), "callback_" => "") * ".dat" for cb in callbacks])
#     cb_files = open.(cb_paths, "w")
#     simulation.verbose && println("$(length(cb_files)) callback files created")
#     try
#         ## Initial measurement
#         save_data(trj_files, prms_files, cb_files, callbacks, simulation.sampletimes[1], simulation)
#         ## Initialise schedulers 
#         n, ns, nl = 2, 2, 2
#         # MAIN LOOP
#         simulation.verbose && println("RUN...")
#         sim_time = @elapsed for t in simulation.sampletimes[1]+1:simulation.steps+simulation.sampletimes[1]
#             ## When scheduled, sample gradient for each learnable move and add it to gradient data
#             if t == simulation.sampling_scheduler[ns]
#                 for (k, lid) in enumerate(simulation.learn_ids)
#                     gd = reducer(+,
#                         eachindex(simulation.chains) |> Map(c -> begin
#                             1:simulation.q_batch_size |> Map(_ -> begin
#                                 sample_gradient_data(
#                                     simulation.pools[c][lid].action,
#                                     simulation.policy_list[lid],
#                                     simulation.parameters_list[lid],
#                                     simulation.chains[c],
#                                     rngs[c];
#                                     ∇logq_forward=simulation.∇logqs_forward[lid],
#                                     ∇logq_backward=simulation.∇logqs_backward[lid],
#                                     shadow=simulation.chains_shadow[c],
#                                     ad_backend=simulation.ad_backend
#                                     )
#                             end)
#                         end) |> Cat()
#                     )
#                     simulation.gradients_data[k] = simulation.gradients_data[k] + gd
#                     simulation.objectives[k] = simulation.gradients_data[k].j / simulation.gradients_data[k].n
#                 end
#                 ns += 1
#             end
#             ## When scheduled, average gradient and update parameters for each learnable move
#             if t == simulation.learning_scheduler[nl]
#                 for (k, lid) in enumerate(simulation.learn_ids)
#                     gd = average(simulation.gradients_data[k])
#                     learning_step!(simulation.parameters_list[lid], gd, simulation.optimisers[lid])
#                     simulation.gradients_data[k] = initialise_gradient_data(simulation.parameters_list[lid])
#                 end
#                 nl += 1
#             end
#             ## One mc sweep for each chain
#             collecter(
#                 eachindex(simulation.chains) |> Map(c -> begin
#                     mc_sweep!(simulation.chains[c], simulation.pools[c], rngs[c]; mc_steps=simulation.sweepstep)
#                 end)
#             )
#             ## Save data when scheduled
#             if t == simulation.sampletimes[n]
#                 simulation.verbose && println("t = $t")
#                 save_data(trj_files, prms_files, cb_files, callbacks, t, simulation)
#                 n += 1
#             end
#         end
#         ## Update summary
#         simulation.verbose && println("Simulation completed in $sim_time s")
#         write_summary(simulation, UpdateSimTime(sim_time))
#     finally
#         ## Make sure to close all files
#         simulation.store_trajectory && close.(trj_files)
#         simulation.store_parameters && close.(prms_files)
#         close.(cb_files)
#         # Save last snapshots
#         for c in eachindex(simulation.chains)
#             open(joinpath(trj_paths[c], "lastframe.xyz"), "w") do trj
#                 store_trajectory(trj, simulation.chains[c], simulation.steps)
#             end
#         end
#         ## Finalise
#         simulation.verbose && println("DONE")
#         write_summary(simulation, FinalReport())
#     end
# end

# nothing