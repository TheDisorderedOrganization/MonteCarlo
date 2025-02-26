abstract type Action end

abstract type Policy end

raise_error(s) = error("No $s is defined")
sample_action!(action::Action, policy::Policy, parameters, system, rng) = raise_error("sample_action!")
log_proposal_density(action, policy, parameters, system) = raise_error("log_proposal_density")
perform_action!(system, action::Action) = raise_error("perform_action!")
unnormalised_log_target_density(x, system) = raise_error("unnormalised_log_target_density")
delta_log_target_density(x₁, x₂, system) = unnormalised_log_target_density(x₂, system) - unnormalised_log_target_density(x₁, system)
invert_action!(action::Action, system) = raise_error("invert_action!")
perform_action_cached!(system, action::Action) = perform_action!(system, action)

mutable struct Move{A<:Action,P<:Policy,V<:AbstractArray,T<:AbstractFloat}
    action::A
    policy::P
    parameters::V
    weight::T
    total_calls::Int
    accepted_calls::Int
end

function Move(action, policy, parameters, weight)
    return Move(action, policy, parameters, weight, 0, 0)
end

function mc_step!(system, action::Action, policy::Policy, parameters::AbstractArray{T}, rng) where {T<:AbstractFloat}
    sample_action!(action, policy, parameters, system, rng)
    logq_forward = log_proposal_density(action, policy, parameters, system)
    x₁, x₂ = perform_action!(system, action)
    Δlogp = delta_log_target_density(x₁, x₂, system)
    invert_action!(action, system)
    logq_backward = log_proposal_density(action, policy, parameters, system)
    α = min(one(T), exp(Δlogp + logq_backward - logq_forward))
    if α > rand(rng)
        return 1
    else
        perform_action_cached!(system, action)
        return 0
    end
end

function mc_sweep!(system, pool, rng; mc_steps=1)
    weights = [move.weight for move in pool]
    for _ in 1:mc_steps
        id = rand(rng, Categorical(weights))
        move = pool[id]
        move.accepted_calls += mc_step!(system, move.action, move.policy, move.parameters, rng)
        move.total_calls += 1
    end
    return nothing
end

struct Metropolis{P,R<:AbstractRNG,C<:Function} <: Algorithm
    pools::Vector{P}            # Vector of independent pools (one for each system)
    sweepstep::Int              # Number of mc steps per mc sweep
    seed::Int                   # Random number seed
    rngs::Vector{R}             # Vector of random number generators
    parallel::Bool              # Flag to parallelise over different threads
    collecter::C                # Transducer to collect results from parallelised loops

    function Metropolis(
        chains::Vector{S},
        pools::Vector{P};
        sweepstep::Int=1,
        seed::Int=1,
        R::DataType=Xoshiro,
        parallel::Bool=false
    ) where {S,P}
        # Safety checks
        @assert length(chains) == length(pools)
        @assert all(k -> all(move -> move.parameters == getindex.(pools, k)[1].parameters, getindex.(pools, k)), eachindex(pools[1]))
        @assert all(k -> all(move -> move.weight == getindex.(pools, k)[1].weight, getindex.(pools, k)), eachindex(pools[1]))
        #Make sure that all policies and parameters across chains refer to the same objects
        policy_list = [move.policy for move in pools[1]]
        parameters_list = [move.parameters for move in pools[1]]
        for pool in pools
            for k in eachindex(policy_list)
                pool[k].policy = policy_list[k]
                pool[k].parameters = parameters_list[k]
            end
        end
        # Handle randomness
        seeds = [seed + c - 1 for c in eachindex(chains)]
        rngs = [R(s) for s in seeds]
        # Handle parallelism
        collecter = parallel ? Transducers.tcollect : collect
        return new{P,R,typeof(collecter)}(pools, sweepstep, seed, rngs, parallel, collecter)
    end

end

function Metropolis(chains; pool=missing, sweepstep=1, seed=1, R=Xoshiro, parallel=false, extras...)
    pools = [deepcopy(pool) for _ in chains]
    return Metropolis(chains, pools; sweepstep=sweepstep, seed=seed, R=R, parallel=parallel)
end

function make_step!(simulation::Simulation, algorithm::Metropolis)
    algorithm.collecter(
        eachindex(simulation.chains) |> Map(c -> begin
            mc_sweep!(simulation.chains[c], algorithm.pools[c], algorithm.rngs[c]; mc_steps=algorithm.sweepstep)
        end)
    )
    return nothing
end

function callback_acceptance(simulation)
    return mean([[move.accepted_calls / move.total_calls for move in pool] for pool in getproperty.(filter(x -> isa(x, Metropolis), simulation.algorithms), :pools)...])
end

function write_parameters(::Policy, parameters)
    return "$(collect(vec(parameters)))"
end

function write_algorithm(io, algorithm::Metropolis, scheduler)
    println(io, "\tMetropolis")
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
    println(io, "\t\tMC steps per simulation step: $(algorithm.sweepstep)")
    println(io, "\t\tSeed: $(algorithm.seed)")
    println(io, "\t\tParallel: $(algorithm.parallel)")
    if algorithm.parallel
        println(io, "\t\tThreads: $(Threads.nthreads())")
    end
    println(io, "\t\tMoves:")
    for (k, move) in enumerate(algorithm.pools[1])
        println(io, "\t\t\tMove $k:")
        println(io, "\t\t\t\tAction: " * replace(string(typeof(move.action)), r"\{.*" => ""))
        println(io, "\t\t\t\tPolicy: " * replace(string(typeof(move.policy)), r"\{.*" => ""))
        println(io, "\t\t\t\tParameters: " * write_parameters(move.policy, move.parameters))
        println(io, "\t\t\t\tWeight: $(move.weight)")
    end
end

struct StoreParameters{V<:AbstractArray} <: Algorithm
    paths::Vector{String}
    files::Vector{IOStream}
    parameters_list::V
    store_first::Bool
    store_last::Bool

    function StoreParameters(pool, path; ids=collect(eachindex(pool)), store_first::Bool=true, store_last::Bool=false)
        parameters_list = [move.parameters for move in pool[ids]]
        dirs = joinpath.(path, "parameters", ["$k" for k in ids])
        mkpath.(dirs)
        paths = joinpath.(dirs, "parameters.dat")
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{typeof(parameters_list)}(paths, files, parameters_list, store_first, store_last)
    end

end

function StoreParameters(chains; dependencies=missing, path=missing, ids=missing, store_first=true, store_last=false, extras...)
    @assert length(dependencies) == 1
    @assert isa(dependencies[1], Metropolis)
    pool = dependencies[1].pools[1]
    if ismissing(ids)
        ids = collect(eachindex(pool))
    end
    return StoreParameters(pool, path; ids=ids, store_first=store_first, store_last=store_last)
end

function initialise(algorithm::StoreParameters, simulation::Simulation)
    simulation.verbose && println("Opening parameter files...")
    algorithm.files .= open.(algorithm.paths, "w")
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreParameters)
    for k in eachindex(algorithm.files)
        println(algorithm.files[k], "$(simulation.t) $(collect(algorithm.parameters_list[k]))")
        flush(algorithm.files[k])
    end
end

function finalise(algorithm::StoreParameters, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    simulation.verbose && println("Closing parameter files...")
    close.(algorithm.files)
    return nothing
end

nothing