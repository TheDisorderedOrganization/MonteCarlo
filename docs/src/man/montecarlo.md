# Markov Chain Monte Carlo Methods

Markov Chain Monte Carlo (MCMC) methods are a class of algorithms for sampling from a probability distribution by constructing a Markov chain that has the desired distribution as its equilibrium distribution. These methods are particularly useful when direct sampling is difficult.

## The Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is one of the most widely used MCMC methods. It works by generating a sequence of sample values in such a way that, as more samples are generated, they more closely approximate the desired distribution.

The algorithm follows these steps:

1. Start with an initial state x₀

2. For each iteration t:
   - Generate a proposal state x' from a proposal distribution q(x'|xₜ)
   - Calculate the acceptance ratio α:
     ```math
     α = min(1, \frac{p(x')q(x_t|x')}{p(x_t)q(x'|x_t)})
     ```
     where p(x) is the target distribution we want to sample from
   
   - Accept or reject the proposal:
     - With probability α: Set xₜ₊₁ = x'
     - With probability 1-α: Set xₜ₊₁ = xₜ

The key properties that make this algorithm work are:

- **Detailed Balance**: The acceptance ratio ensures that the chain satisfies detailed balance, which guarantees that the target distribution p(x) is the stationary distribution
- **Ergodicity**: Under mild conditions, the chain will eventually explore all regions of the state space with non-zero probability

## Implementation in Arianna

In Arianna, the Metropolis-Hastings algorithm is implemented through the `Metropolis` struct, which requires:

1. A system state representation
2. A pool of Monte Carlo moves (actions)
3. Proposal distributions (policies) for generating new states
4. Functions to calculate:
   - The log target density (`unnormalised_log_target_density`)
   - The log proposal density (`log_proposal_density`)
   - How to perform actions (`perform_action!`)

See the [Adding Your Own System](@ref) section for details on implementing these components for your specific problem.
