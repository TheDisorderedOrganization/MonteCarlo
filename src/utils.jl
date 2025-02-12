function build_schedule(steps::Int, burn::Int, Δt::Int)
    return collect(burn:Δt:steps) ∪ [steps]
end

function build_schedule(steps::Int, burn::Int, base::AbstractFloat)
    return unique(vcat([burn], [burn + Int(base^n) for n in 0:floor(Int, log(base, steps - burn))], [steps]))
end

function build_schedule(steps::Int, burn::Int, block::Vector{Int})
    nblock = (steps - burn) ÷ block[end]
    blocks = [block .+ burn .+ (m - 1) * block[end] for m in 1:nblock]
    return filter(x -> x ≤ steps, unique(vcat(blocks..., [steps])))
end

function write_system(io, system)
    println(io, "\t" * "$(typeof(system))")
    return nothing
end

function write_algorithm(io, algorithm::Algorithm)
    println(io, "\t" * replace(string(typeof(algorithm)), r"\{.*" => ""))
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ algorithm.scheduler[end], algorithm.scheduler)))")
end

function write_summary(simulation)
    open(joinpath(simulation.path, "summary.log"), "w") do file
        println(file, "SIMULATION SUMMARY")
        println(file)
        println(file, "Simulation:")
        println(file, "\tSteps: $(simulation.steps)")
        println(file, "\tNumber of chains: $(length(simulation.chains))")
        println(file, "\tNumber of algorithms: $(length(simulation.algorithms))")
        println(file, "\tVerbose: $(simulation.verbose)")
        println(file, "\tStarted on $(now())")
        println(file)
        println(file, "System:")
        write_system(file, simulation.chains[1])
        println(file)
        println(file, "Algorithms:")
        for algorithm in simulation.algorithms
            write_algorithm(file, algorithm)
        end
        println(file)
    end
end

function update_summary(simulation, sim_time)
    open(joinpath(simulation.path, "summary.log"), "a") do file
        println(file, "Report:")
        println(file, "\tSimulation time: $sim_time s")
    end
end

function finalise_summary(simulation)
    open(joinpath(simulation.path, "summary.log"), "a") do file
        total_size = 0
        for (root, dirs, files) in walkdir(simulation.path)
            for file in files
                total_size += filesize(joinpath(root, file))
            end
        end
        sim_size = total_size / 1024^2
        println(file, "\tSimulation size: $(sim_size) MB")
        println(file, "\tStatus: Completed on $(now())")
    end
end

struct StoreCallbacks{V,VS<:AbstractArray} <: Algorithm
    callbacks::V
    paths::Vector{String}
    files::Vector{IOStream}
    scheduler::VS

    function StoreCallbacks(chains, path, step, callbacks::V; scheduler::VS=1:steps) where {V, VS<:AbstractArray}
        mkpath(path)
        paths = joinpath.(path, [replace(string(cb), "callback_" => "") * ".dat" for cb in callbacks])
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{V,VS}(callbacks, paths, files, scheduler)
    end

end

function initialise(algorithm::StoreCallbacks, simulation::Simulation)
    simulation.verbose && println("Opening callback files...")
    algorithm.files .= open.(algorithm.paths, "w")
    make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreCallbacks)
    for (callback, file) in zip(algorithm.callbacks, algorithm.files)
        println(file, "$(simulation.t) $(callback(simulation))")
    end
end

function finalise(algorithm::StoreCallbacks, simulation::Simulation)
    simulation.verbose && println("Closing callback files...")
    close.(algorithm.files)
    return nothing
end

struct StoreTrajectories{VS<:AbstractArray} <: Algorithm
    paths::Vector{String}
    files::Vector{IOStream}
    scheduler::VS

    function StoreTrajectories(chains, path, steps; scheduler::VS=1:steps) where {VS<:AbstractArray}
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        paths = joinpath.(dirs, "trajectory.xyz")
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{VS}(paths, files, scheduler)
    end

end

function store_trajectory(trj, system, t)
    println(trj, "$t, $system")
    return nothing
end

function initialise(algorithm::StoreTrajectories, simulation::Simulation)
    simulation.verbose && println("Opening trajectory files...")
    algorithm.files .= open.(algorithm.paths, "w")
    make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreTrajectories)
    for c in eachindex(simulation.chains)
        store_trajectory(algorithm.files[c], simulation.chains[c], simulation.t)
    end
end

function finalise(algorithm::StoreTrajectories, simulation::Simulation)
    make_step!(simulation, algorithm)
    simulation.verbose && println("Closing trajectory files...")
    close.(algorithm.files)
    return nothing
end

struct StoreLastFrames{VS<:AbstractArray} <: Algorithm
    paths::Vector{String}
    scheduler::VS

    function StoreLastFrames(chains, path, steps; scheduler::VS=[steps]) where{VS<:AbstractArray}
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        paths = joinpath.(dirs, "lastframe.xyz")
        return new{VS}(paths, scheduler)
    end

end

function finalise(algorithm::StoreLastFrames, simulation::Simulation)
    for c in eachindex(simulation.chains)
        open(algorithm.paths[c], "w") do file
            store_trajectory(file, simulation.chains[c], simulation.t)
        end
    end
    return nothing
end

struct PrintTimeSteps{VS<:AbstractArray} <: Algorithm
    scheduler::VS

    function PrintTimeSteps(chains, path, steps; scheduler::VS=1:steps) where {VS<:AbstractArray}
        return new{VS}(scheduler)
    end
    
end

function make_step!(simulation::Simulation, ::PrintTimeSteps)
    println("t = $(simulation.t)")
end

nothing