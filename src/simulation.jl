mutable struct Simulation{S, A}
    chains::Vector{S}               # Vector of independent systems
    algorithms::A                   # List of algorithms
    steps::Int                      # Number of MC sweeps
    t::Int                          # Current time step
    counters::Vector{Int}           # Counters for the schedulers (one for each algorithm)
    path::String                    # Simulation path
    verbose::Bool                   # Flag for verbose

    function Simulation(
        chains::Vector{S},
        algorithms::A,
        steps::Int;
        path::String="data",
        verbose::Bool=false
    ) where {S,A}
        for algorithm in algorithms
            @assert all(t -> 0 ≤ t ≤ steps, algorithm.scheduler)
            @assert issorted(algorithm.scheduler)
        end
        t = 0
        counters = findfirst.(x -> x > 0, [algorithm.scheduler for algorithm in algorithms])
        mkpath(path)
        return new{S, A}(chains, algorithms, steps, t, counters, path, verbose)
    end

end

abstract type Algorithm end

initialise(::Algorithm, ::Simulation) = nothing

make_step!(::Simulation, ::Algorithm) = nothing

finalise(::Algorithm, ::Simulation) = nothing

function run!(simulation::Simulation)
    try
        simulation.verbose && println("INISIALISATION")
        for algorithm in simulation.algorithms
            initialise(algorithm, simulation)
        end
        write_summary(simulation)
        simulation.verbose && println("RUN...")
        sim_time = @elapsed for simulation.t in 1:simulation.steps
            for k in eachindex(simulation.algorithms)
                if simulation.t == simulation.algorithms[k].scheduler[simulation.counters[k]]
                    make_step!(simulation, simulation.algorithms[k])
                    simulation.counters[k] += 1
                end
            end
        end
        simulation.verbose && println("Simulation completed in $sim_time s")
        update_summary(simulation, sim_time)
    finally
        simulation.verbose && println("FINALISATION")
        for algorithm in simulation.algorithms
            finalise(algorithm, simulation)
        end
        finalise_summary(simulation)
        simulation.verbose && println("DONE")
    end
    return nothing
end

nothing
