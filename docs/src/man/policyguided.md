# Policy-guided Monte Carlo

Policy-guided Monte Carlo (PGMC) is an **adaptive Monte Carlo method** that dynamically adjusts the proposal distribution in the Metropolis-Hastings (MH) kernel to **maximise sampling efficiency**, using a formalism inspired by **reinforcement learning**.

As long as the proposal distribution $Q$ guarantees ergodicity, here is significant flexibility in the choice of its specific form. PGMC aims at finding an optimal proposal distribution that maximises some measure of  efficiency of the Markov chain. To do this, it needs a **reward function** $r\left(x,x'\right)$ that quantifies the performance of a single transition $x\to x'$. The reward function must satisfy the constraint $r\left(x,x'\right)=0$. This can be used to define the **objective function**

```math
J\left(Q\right)=\mathbb E_{\substack{x\sim P \\ x'\sim K}}\left[r\left(x,x'\right)\right].
```

The goal is to find a proposal distribution $Q^\star$ that maximises the objective function $J$. To practically tackle the problem, we restrict the search to a family of distributions $Q_{\theta}$​ parameterised by a real vector $\theta$. Starting from an initial guess, we then update $\theta$ iteratively according to the **stochastic gradient ascent** procedure

```math
\theta\leftarrow\theta +\eta\,\widehat{\nabla_\theta J},
```

where $\eta$ is the learning rate and $\widehat{\nabla_\theta J}$ is a stochastic estimate of the actual gradient of $J$ with respect to $\theta$.