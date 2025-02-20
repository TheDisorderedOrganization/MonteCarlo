# MonteCarlo <img src="tdo_logo.png" alt="tdo" width="50"/>

[![License](https://img.shields.io/badge/license-GPL%203.0-red.svg)](https://github.com/TheDisorderedOrganization/MCMC/blob/main/LICENSE)
[![CI](https://github.com/TheDisorderedOrganization/MonteCarlo/actions/workflows/ci.yml/badge.svg)](https://github.com/TheDisorderedOrganization/MonteCarlo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TheDisorderedOrganization/MonteCarlo/graph/badge.svg?token=URGL1HJOOI)](https://codecov.io/gh/TheDisorderedOrganization/MonteCarlo)

## Overview

MonteCarlo is a flexible and extensible framework for Monte Carlo simulations. Instead of acting as a black-box simulator, it provides a modular structure where users define their own system and Monte Carlo "moves". The package includes some simple predefined systems for example purposes, and more complex systems are defined in other repos like [ParticlesMC](https://github.com/TheDisorderedOrganization/ParticlesMC).

## Features

- **General-Purpose Monte Carlo Engine**: A lightweight framework that provides the core algorithms for Monte Carlo sampling, allowing users to define their own systems, moves, and proposal distributions.
- **Extensible Algorithms**: Built-in support for Metropolis-Hastings with the flexibility to implement advanced techniques like event-chain Monte Carlo.
- **Policy-Guided Monte Carlo**: Integrates adaptive sampling using policy gradient methods to optimise move parameters dynamically.
- **Predefined Systems**: Includes simple examples to help users get started quickly, with additional system implementations available through companion repositories like [ParticlesMC](https://github.com/TheDisorderedOrganization/ParticlesMC).

## Installation

To install the MonteCarlo module, you can clone the repository and use the Julia package manager to add the module path to your environment.

```sh
git clone https://github.com/TheDisorderedOrganization/MonteCarlo.git
cd MonteCarlo
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

## Usage

### Running a Monte Carlo Simulation

To perform a Monte Carlo (MC) simulation, it is necessary to define the system and the set of possible moves, followed by executing the simulation using the appropriate functions. Below, we present a basic Monte Carlo simulation utilizing the Particle system and move set defined in [particle_1D.jl](example/particle_1d/particle_1d.jl).

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
pool = (Move(Displacement(0.0), StandardGaussian(), ComponentArray(σ=0.1), 1.0),)
steps = 10^5
burn = 1000
block = [0, 10]
sampletimes = build_schedule(steps, burn, block)
path = "data/MC/particle_1d/Harmonic/beta$β/M$M/seed$seed"

algorithm_list = (
    (algorithm=Metropolis, pool=pool, seed=seed, parallel=false),
    (algorithm=StoreCallbacks, callbacks=(callback_energy, callback_acceptance), scheduler=sampletimes),
    (algorithm=StoreTrajectories, scheduler=sampletimes),
) 

simulation = Simulation(chains, algorithm_list, steps; path=path, verbose=true)
run!(simulation)
```
This implementation employs the **Metropolis algorithm** for Monte Carlo sampling, utilizing **Gaussian-distributed** displacements as the proposed moves. The simulation records **energy** and **acceptance** statistics while storing **particle trajectories** for analysis. The resulting data is saved in the specified output directory for further evaluation.

### Adding Your Own System

Now that you understand how to run a Monte Carlo (MC) simulation, you may want to extend the framework by defining your own system. The file [particle_1D.jl](example/particle_1d/particle_1d.jl) provides a minimal example of a system, which you can use as a reference when creating a new one.

To define a new system, you need to specify its state variables, Monte Carlo moves and how to perform them. These components determine how the system evolves during the simulation. A system consists of:

- System: Specify the state representation and the target probability density.	
    1. State representation: Defines the key quantities describing the system (e.g., position, energy, temperature). Your system has to be a struct where each element is a state variable. Example:
    ```julia
    mutable struct Particle{T<:AbstractFloat}
        x::T
        β::T
        e::T
    end
    ```
    2. Target density: This is the actual probablity distribution of the system that you want to sample. In this case it's the Boltzmann distribution at inverse temperature $\beta$
    ```julia
    function MonteCarlo.delta_log_target_density(e₁, e₂, system::Particle)
        return -system.β * (e₂ - e₁)
    end
    ```

- Monte Carlo action: Specifies how the system state changes during the simulation (e.g., random displacements).
    1. Define an action. In the example, the action is a displacement.
    ```julia
    mutable struct Displacement{T<:AbstractFloat} <: Action
        δ::T
    end
    ```
    2. Define how this action is sampled in the `MonteCarlo.sample_action!` function. In the example, the displacement length is sampled from a normal distribution.
    ```julia
    function MonteCarlo.sample_action!(action::Displacement,::StandardGaussian, parameters, system::Particle, rng)
        action.δ = rand(rng, Normal(zero(action.δ), parameters.σ))
        return nothing
    end
    ```
    3. Specify the probablity of proposing the action in `MonteCarlo.log_proposal_density`. Note that this function must give the exact probability of sampling the action with `MonteCarlo.sample_action!`. In this case, we just need the density of the normal distribution.
    ```julia
    function MonteCarlo.log_proposal_density(action::Displacement, ::StandardGaussian, parameters, system::Particle)
        return -(action.δ)^2 / (2parameters.σ^2) - log(2π * parameters.σ^2) / 2
    end
    ```
    3. Finally, provide how this action changes the state of the system in the `MonteCarlo.perform_action!` function. In the example, performing the displacement updates the position and the energy of the particle. 
    ```julia
    function MonteCarlo.perform_action!(system::Particle, action::Displacement)
        e₁ = system.e
        system.x += action.δ
        system.e = potential(system.x)
        return e₁, system.e
    end
    ```

By modifying and extending the existing particle_1D.jl example, you can create a variety of physical and mathematical models suitable for MC simulations.

## Contributing

We welcome contributions from the community. If you have a new system or feature to add, please fork the repository, make your changes, and submit a pull request.

## Citing MonteCarlo

If you use MonteCarlo in your research, please cite it! You can find the citation information in the [CITATION](CITATION.cff) file or directly through GitHub’s "Cite this repository" button.

## License

This project is licensed under the GNU General Public License v3.0.  License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainers.
