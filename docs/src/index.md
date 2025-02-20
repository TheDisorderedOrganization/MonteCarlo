# MonteCarlo

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

## Manual

```@contents
Pages = [
    "man/simulation.md",
    "man/system.md",
]
Depth = 1
```

## API

```@contents
Pages = [
    "lib/api.md",
]
Depth = 1
```

