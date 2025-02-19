# MonteCarlo <img src="tdo_logo.png" alt="tdo" width="50"/>

[![License](https://img.shields.io/badge/license-GPL%203.0-red.svg)](https://github.com/TheDisorderedOrganization/MCMC/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/TheDisorderedOrganization/MonteCarlo/graph/badge.svg?token=URGL1HJOOI)](https://codecov.io/gh/TheDisorderedOrganization/MonteCarlo)

## Overview

The `MonteCarlo` module provides a flexible and extensible Monte Carlo simulation framework. This framework allows users to perform Monte Carlo simulations with ease and provides the necessary tools to define and manage different types of systems. The module includes some simple predefined systems for example purposes, and more complex systems are defined in other repos like `ParticlesMC`.

## Features

- **Monte Carlo Simulation Framework**: A robust and flexible framework for performing Monte Carlo simulations.
- **Predefined Systems**: Includes several predefined systems to get started quickly.
- **Extensibility**: Users can easily add and integrate their own systems into the framework.
- **Policy-Guided Monte Carlo Simulation**: Advanced simulation techniques using policy-guided Monte Carlo methods.


## Installation

To install the `MonteCarlo` module, you can clone the repository and use the Julia package manager to add the module path to your environment.

```sh
git clone https://github.com/TheDisorderedOrganization/MonteCarlo.git
cd MonteCarlo
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

## Usage

### Running a Monte Carlo Simulation

To perform a Monte Carlo (MC) simulation, it is necessary to define the system and the set of possible moves, followed by executing the simulation using the appropriate functions. Below, we present a basic Monte Carlo simulation utilizing the Particle system and move set defined in [particle_1D.jl](example/particle_1d/particle_1d.jl).

The Particle system is characterized by three key quantities: its position  x , the inverse temperature  \beta, and its energy  e . The Monte Carlo move applied to the particle consists of a displacement by  \delta x , where  \delta x  is sampled from a user-defined probability distribution (that we call policy).

The following Julia script initializes and runs the simulation:

```
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

### Adding Your Own System

Now that you understand how to run a Monte Carlo (MC) simulation, you may want to extend the framework by defining your own system. The file [particle_1D.jl](example/particle_1d/particle_1d.jl) provides a minimal example of a system, which you can use as a reference when creating a new one.

To define a new system, you need to specify its state variables, Monte Carlo moves and how to perform them. These components determine how the system evolves during the simulation. A system consists of:
	
- State representation: Defines the key quantities describing the system (e.g., position, energy, temperature). Your system has to be a struct where each element is a state variable. Example:
```
mutable struct Particle{T<:AbstractFloat}
    x::T
    β::T
    e::T
end
```

- Monte Carlo action: Specifies how the system state changes during the simulation (e.g., random displacements).
    1. Define an action. In the example, the action is a displacement.
    ```
    mutable struct Displacement{T<:AbstractFloat} <: Action
        δ::T
    end
    ```
    2. Define how this action is sampled in the `MonteCarlo.sample_action!` function. In the example, the displacement length is sampled from a normal distribution.
    ```
    function MonteCarlo.sample_action!(action::Displacement,::StandardGaussian, parameters, system::Particle, rng)
        action.δ = rand(rng, Normal(zero(action.δ), parameters.σ))
        return nothing
    end
    ```
    3. Finally, provide how this action changes the state of the system in the `MonteCarlo.perform_action!` function. In the example, performing the displacement updates the position and the energy of the particle. 
    ```
    function MonteCarlo.perform_action!(system::Particle, action::Displacement)
        e₁ = system.e
        system.x += action.δ
        system.e = potential(system.x)
        return e₁, system.e
    end
    ```

- The probability of accepting / rejecting a move in the ` MonteCarlo.delta_log_target_density`. In the example, we provide the classical Metropolis probability of accepting a move based on the energy difference between the old and new state.
```
function MonteCarlo.sample_action!(action::Displacement, ::StandardGaussian, parameters, system::Particle, rng)
    action.δ = rand(rng, Normal(zero(action.δ), parameters.σ))
    return nothing
end
```

By modifying and extending the existing particle_1D.jl example, you can create a variety of physical and mathematical models suitable for MC simulations.

## Contributing

We welcome contributions from the community. If you have a new system or feature to add, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0.  License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainers.
