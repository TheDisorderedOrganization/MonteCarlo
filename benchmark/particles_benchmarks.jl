using MCMC
using BenchmarkTools
using StaticArrays
using Distributions
using Random
using ComponentArrays

# MEDIUM SS 3D NO LL
begin
    seed = 42
    rng = Xoshiro(seed)
    N = 216
    xA = 0.5
    NA = round(Int, N * xA)
    NB = N - NA
    d = 3
    temperature = 0.2
    density = 0.5342
    box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
    position = [box .* @SVector rand(rng, d) for i in 1:N]
    species = shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB)))
    model = MCMC.BHHP()
    system = System(position, species, density, temperature, model)
    displacement_policy = SimpleGaussian()
    displacement_parameters = ComponentArray(σ=0.1)
    pool = (Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),)
    @btime mc_step!(system, pool[1].action, pool[1].policy, pool[1].parameters, rng) # 5.114 μs (1 allocation: 16 bytes)
    @btime mc_sweep!(system, pool, rng; mc_steps=N) # 1.737 ms (436 allocations: 13.59 KiB)
end

# MEDIUM SS 3D LL
begin
    seed = 42
    rng = Xoshiro(seed)
    N = 216
    xA = 0.5
    NA = round(Int, N * xA)
    NB = N - NA
    d = 3
    temperature = 0.2
    density = 0.5342
    box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
    position = [box .* @SVector rand(rng, d) for i in 1:N]
    species = shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB)))
    model = MCMC.BHHP()
    system = System(position, species, density, temperature, model; list_type=LinkedList)
    displacement_policy = SimpleGaussian()
    displacement_parameters = ComponentArray(σ=0.1)
    pool = (Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),)
    @btime mc_step!(system, pool[1].action, pool[1].policy, pool[1].parameters, rng)
    @btime mc_sweep!(system, pool, rng; mc_steps=N)
end

# SMALL 2D NO LL
begin
    seed = 42
    rng = Xoshiro(seed)
    N = 50
    xA, xB, xC = 0.46, 0.26, 0.28
    NA = round(Int, N * xA)
    NB = round(Int, N * xB)
    NC = N - NA - NB
    d = 2
    temperature = 0.8
    density = 1.1920748468939728
    box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
    position = [box .* @SVector rand(rng, d) for i in 1:N]
    species = shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB), 3ones(Int, NC)))
    model = JBB()
    system = System(position, species, density, temperature, model)
    displacement_policy = SimpleGaussian()
    displacement_parameters = ComponentArray(σ=0.05)
    pool = (Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),)
    @btime mc_step!(system, pool[1].action, pool[1].policy, pool[1].parameters, rng)
    @btime mc_sweep!(system, pool, rng; mc_steps=N)
end

# LARGE 2D NO LL
begin
    seed = 42
    rng = Xoshiro(seed)
    N = 1000
    xA, xB, xC = 0.46, 0.26, 0.28
    NA = round(Int, N * xA)
    NB = round(Int, N * xB)
    NC = N - NA - NB
    d = 2
    temperature = 0.8
    density = 1.1920748468939728
    box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
    position = [box .* @SVector rand(rng, d) for i in 1:N]
    species = shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB), 3ones(Int, NC)))
    model = JBB()
    system = System(position, species, density, temperature, model)
    displacement_policy = SimpleGaussian()
    displacement_parameters = ComponentArray(σ=0.05)
    pool = (Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),)
    @btime mc_step!(system, pool[1].action, pool[1].policy, pool[1].parameters, rng)
    @btime mc_sweep!(system, pool, rng; mc_steps=N)
end

# LARGE 2D LL
begin
    seed = 42
    rng = Xoshiro(seed)
    N = 1000
    xA, xB, xC = 0.46, 0.26, 0.28
    NA = round(Int, N * xA)
    NB = round(Int, N * xB)
    NC = N - NA - NB
    d = 2
    temperature = 0.8
    density = 1.1920748468939728
    box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
    position = [box .* @SVector rand(rng, d) for i in 1:N]
    species = shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB), 3ones(Int, NC)))
    model = JBB()
    system = System(position, species, density, temperature, model; list_type=LinkedList)
    displacement_policy = SimpleGaussian()
    displacement_parameters = ComponentArray(σ=0.05)
    pool = (Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),)
    @btime mc_step!(system, pool[1].action, pool[1].policy, pool[1].parameters, rng)
    @btime mc_sweep!(system, pool, rng; mc_steps=N)
end

# LARGER 2D LL
begin
    seed = 42
    rng = Xoshiro(seed)
    N = 3000
    xA, xB, xC = 0.46, 0.26, 0.28
    NA = round(Int, N * xA)
    NB = round(Int, N * xB)
    NC = N - NA - NB
    d = 2
    temperature = 0.8
    density = 1.1920748468939728
    box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
    position = [box .* @SVector rand(rng, d) for i in 1:N]
    species = shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB), 3ones(Int, NC)))
    model = JBB()
    system = System(position, species, density, temperature, model; list_type=LinkedList)
    displacement_policy = SimpleGaussian()
    displacement_parameters = ComponentArray(σ=0.05)
    pool = (Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),)
    @btime mc_step!(system, pool[1].action, pool[1].policy, pool[1].parameters, rng)
    @btime mc_sweep!(system, pool, rng; mc_steps=N)
end

# LARGER SS 3D LL
begin
    seed = 42
    rng = Xoshiro(seed)
    N = 3000
    xA = 0.5
    NA = round(Int, N * xA)
    NB = N - NA
    d = 3
    temperature = 1.0
    density = 0.5
    box = @SVector fill(typeof(temperature)((N / density)^(1 / d)), d)
    position = [box .* @SVector rand(rng, d) for i in 1:N]
    species = shuffle!(rng, vcat(ones(Int, NA), 2ones(Int, NB)))
    model = BHHP()
    system = System(position, species, density, temperature, model; list_type=LinkedList)
    displacement_policy = SimpleGaussian()
    displacement_parameters = ComponentArray(σ=0.05)
    pool = (Move(Displacement(0, zero(box)), displacement_policy, displacement_parameters, 1.0),)
    @btime mc_step!(system, pool[1].action, pool[1].policy, pool[1].parameters, rng)
    @btime mc_sweep!(system, pool, rng; mc_steps=N)
end