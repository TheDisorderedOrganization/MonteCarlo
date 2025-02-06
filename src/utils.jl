###############################################################################

function scheduler(steps::Int, burn::Int, Δt::Int)
    return collect(burn : Δt : (steps + burn + Δt * (steps % Δt != 0)))
end

function scheduler(steps::Int, burn::Int, base::AbstractFloat)
    return unique(vcat([0], [Int(base^n) for n in 0:floor(Int, log(base, steps))], [steps])) .+ burn
end

function scheduler(steps::Int, burn::Int, block::Vector{Int})
    nblock = steps ÷ block[end]
    blocks = Vector{Vector{Int}}(undef, nblock)
    blocks[1] = copy(block)
    for m ∈ 2:nblock
        blocks[m] = blocks[m-1] .+ block[end]
    end
    blocks[end][end] = steps
    return unique(vcat(blocks...)) .+ burn
end

###############################################################################

function store_trajectory(trj, system, t)
    println(trj, system)
    return nothing
end

abstract type SummaryWriter end

struct InitialiseSummary <: SummaryWriter end

function write_system(io, system)
    println(io, system)
    return nothing
end

function write_parameters(::Policy, parameters)
    return "$(collect(vec(parameters)))"
end

function write_summary(simulation, ::InitialiseSummary)
    open(joinpath(simulation.path, "summary.log"), "w") do file
        println(file, "MONTE CARLO SIMULATION")
        println(file)
        println(file, "System:")
        write_system(file, simulation.chains[1])
        println(file)
        println(file, "Moves:")
        for (k, move) in enumerate(simulation.pools[1])
            println(file, "\tMove $k:")
            println(file, "\t\tAction: " * replace(string(typeof(move.action)), r"\{.*" => ""))
            println(file, "\t\tPolicy: " * replace(string(typeof(move.policy)), r"\{.*" => ""))
            println(file, "\t\tParameters: " * write_parameters(move.policy, move.parameters))
            println(file, "\t\tWeight: $(move.weight)")
        end
        println(file)
        println(file, "Simulation:")
        println(file, "\tSeed: $(simulation.seed)")
        println(file, "\tNumber of chains: $(length(simulation.chains))")
        println(file, "\tMC sweeps: $(simulation.steps)")
        println(file, "\tMC steps per MC sweep: $(simulation.sweepstep)")
        println(file, "\tBurn-in sweeps: $(simulation.sampletimes[1])")
        println(file, "\tNumber of measurements: $(length(simulation.sampletimes) - (simulation.sampletimes[end] > simulation.steps))")
        println(file, "\tStore trajectory: $(simulation.store_trajectory)")
        println(file, "\tParallel: $(simulation.parallel)")
        if simulation.parallel
            println(file, "\tThreads: $(Threads.nthreads())")
        end
        println(file, "\tVerbose: $(simulation.verbose)")
    end
end

struct UpdateBurnTime{T<:AbstractFloat} <: SummaryWriter
    time::T
end

function write_summary(simulation, writer::UpdateBurnTime)
    open(joinpath(simulation.path, "summary.log"), "a") do file
        println(file)
        println(file, "Report:")
        println(file, "\tBurn-in time: $(writer.time) s")
    end
end

struct UpdateSimTime{T<:AbstractFloat} <: SummaryWriter
    time::T
end

function write_summary(simulation, writer::UpdateSimTime)
    open(joinpath(simulation.path, "summary.log"), "a") do file
        println(file, "\tSimulation time: $(writer.time) s")
    end
end

struct FinalReport <: SummaryWriter end

function write_summary(simulation, ::FinalReport)
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

###############################################################################

nothing