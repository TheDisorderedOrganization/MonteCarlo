# Running a Monte Carlo Simulation

To perform a Monte Carlo (MC) simulation, it is necessary to define the system and the set of possible moves, followed by executing the simulation using the appropriate functions. Below, we present a basic Monte Carlo simulation utilizing the Particle system and move set defined in [particle_1d.jl](https://github.com/TheDisorderedOrganization/Arianna/example/particle_1d/particle_1d.jl) 

The Particle system is characterized by three key quantities: its position  $x$, the inverse temperature  $\beta$, and its energy $e$. The Monte Carlo move applied to the particle consists of a displacement by  $\delta$, where  $\delta$  is sampled from a user-defined probability distribution that we call policy.

The following Julia script initializes and runs the simulation:

```julia
include("example/particle_1D/particle_1d.jl")

potential(x) = x^2

x0 = 0
temperature = 0.5
β = 1/temperature
M = 10
chains = [System(x0, β) for _ in 1:M]
pools = [(Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),) for _ in 1:M]
steps = 10^5
burn = 1000
block = [0, 10]
sampletimes = build_schedule(steps, burn, block)
path = "data/MC/particle_1d/Harmonic/beta$β/M$M/seed$seed"

algorithm_list = (
    (algorithm=Metropolis, pools=pools, seed=seed, parallel=false),
    (algorithm=StoreCallbacks, callbacks=(callback_energy, callback_acceptance), scheduler=sampletimes),
    (algorithm=StoreTrajectories, scheduler=sampletimes),
) 

simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
run!(simulation)
```
This implementation employs the **Metropolis algorithm** for Monte Carlo sampling, utilizing **Gaussian-distributed** displacements as the proposed moves. The simulation records **energy** and **acceptance** statistics while storing **particle trajectories** for analysis. The resulting data is saved in the specified output directory for further evaluation.