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

function write_algorithm(io, algorithm::Algorithm, scheduler)
    println(io, "\t" * replace(string(typeof(algorithm)), r"\{.*" => ""))
    println(io, "\t\tCalls: $(length(filter(x -> 0 < x ≤ scheduler[end], scheduler)))")
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
        for (algorithm, scheduler) in zip(simulation.algorithms, simulation.schedulers)
            write_algorithm(file, algorithm, scheduler)
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

struct StoreCallbacks{V} <: Algorithm
    callbacks::V
    paths::Vector{String}
    files::Vector{IOStream}
    store_first::Bool
    store_last::Bool

    function StoreCallbacks(callbacks::V, path; store_first::Bool=true, store_last::Bool=false) where {V}
        mkpath(path)
        paths = joinpath.(path, [replace(string(cb), "callback_" => "") * ".dat" for cb in callbacks])
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{V}(callbacks, paths, files, store_first, store_last)
    end

end

function StoreCallbacks(chains; path=missing, callbacks=missing, store_first=true, store_last=false, extras...)
    if ismissing(callbacks)
        callbacks = []
    end
    return StoreCallbacks(callbacks, path; store_first=store_first, store_last=store_last)
end

function initialise(algorithm::StoreCallbacks, simulation::Simulation)
    simulation.verbose && println("Opening callback files...")
    algorithm.files .= open.(algorithm.paths, "w")
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreCallbacks)
    for (callback, file) in zip(algorithm.callbacks, algorithm.files)
        println(file, "$(simulation.t) $(callback(simulation))")
        flush(file)
    end
end

function finalise(algorithm::StoreCallbacks, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    simulation.verbose && println("Closing callback files...")
    close.(algorithm.files)
    return nothing
end

abstract type Format end

struct TXT <: Format
    extension::String
    function TXT()
        return new(".txt")
    end
end

struct DAT <: Format
    extension::String
    function DAT()
        return new(".dat")
    end
end

struct StoreTrajectories{F<:Format} <: Algorithm
    paths::Vector{String}
    files::Vector{IOStream}
    fmt::F
    store_first::Bool
    store_last::Bool

    function StoreTrajectories(chains, path, fmt; store_first::Bool=true, store_last::Bool=false)
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        ext = fmt.extension
        paths = joinpath.(dirs, "trajectory$ext")
        files = Vector{IOStream}(undef, length(paths))
        try
            files = open.(paths, "w")
        finally
            close.(files)
        end
        return new{typeof(fmt)}(paths, files, fmt, store_first, store_last)
    end

end

function StoreTrajectories(chains; path=missing, fmt=DAT(), store_first=true, store_last=false, extras...)
    return StoreTrajectories(chains, path, fmt, store_first=store_first, store_last=store_last)
end

function store_trajectory(io, system, t, fmt::Format)
    println(io, "$t, $system")
    return nothing
end

function initialise(algorithm::StoreTrajectories, simulation::Simulation)
    simulation.verbose && println("Opening trajectory files...")
    algorithm.files .= open.(algorithm.paths, "w")
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreTrajectories)
    for c in eachindex(simulation.chains)
        store_trajectory(algorithm.files[c], simulation.chains[c], simulation.t, algorithm.fmt)
        flush(algorithm.files[c])
    end
end

function finalise(algorithm::StoreTrajectories, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    simulation.verbose && println("Closing trajectory files...")
    close.(algorithm.files)
    return nothing
end

struct StoreLastFrames <: Algorithm
    paths::Vector{String}
    fmt::Format
    function StoreLastFrames(chains, path, fmt)
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        ext = fmt.extension
        paths = joinpath.(dirs, "lastframe$ext")
        return new(paths, fmt)
    end

end

function StoreLastFrames(chains; path=missing, fmt=DAT(), extras...)
    return StoreLastFrames(chains, path, fmt)
end

function finalise(algorithm::StoreLastFrames, simulation::Simulation)
    for c in eachindex(simulation.chains)
        open(algorithm.paths[c], "w") do file
            store_trajectory(file, simulation.chains[c], simulation.t, algorithm.fmt)
        end
    end
    return nothing
end

struct StoreBackups <: Algorithm
    dirs::Vector{String}
    fmt::Format
    store_first::Bool
    store_last::Bool
    function StoreBackups(chains, path, fmt; store_first::Bool=false, store_last::Bool=false)
        dirs = joinpath.(path, "trajectories", ["$c" for c in eachindex(chains)])
        mkpath.(dirs)
        return new(dirs, fmt, store_first, store_last)
    end

end

function StoreBackups(chains; path=missing, fmt=DAT(), store_first=false, store_last=false, extras...)
    return StoreBackups(chains, path, fmt, store_first=store_first, store_last=store_last)
end

function initialise(algorithm::StoreBackups, simulation::Simulation)
    algorithm.store_first && make_step!(simulation, algorithm)
    return nothing
end

function make_step!(simulation::Simulation, algorithm::StoreBackups)
    for c in eachindex(simulation.chains)
        open(joinpath(algorithm.dirs[c], "restart_t$(simulation.t)$(algorithm.fmt.extension)"), "w") do file
            store_trajectory(file, simulation.chains[c], simulation.t, algorithm.fmt)
        end
    end
end

function finalise(algorithm::StoreBackups, simulation::Simulation)
    algorithm.store_last && make_step!(simulation, algorithm)
    return nothing
end

struct PrintTimeSteps <: Algorithm end

function PrintTimeSteps(chains; extras...)
    return PrintTimeSteps()
end

function make_step!(simulation::Simulation, ::PrintTimeSteps)
    t = simulation.t
    percent = t / simulation.steps
    bar_length = 50
    filled_length = Int(round(percent * bar_length))
    bar = "\033[1;34m" * repeat("■", filled_length) * "\033[0m" * repeat("□", bar_length - filled_length)
    @printf("\rProgress: [%s] %.0f%% t = %d", bar, percent * 100, t)
end

nothing