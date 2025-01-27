using MCMC
using Test
using StaticArrays
using Distributions

# Tests pour la fonction de calcul d'énergie locale
@testset "Potential energy test" begin
    # Configuration du système
    data = readlines("config_0.xyz")
    N = parse(Int, data[4])
    gbox = parse.(Float64, split(data[6], " "))
    L = gbox[2] - gbox[1]
    box = @SVector [L, L]
    frame = data[10:end]
    species = zeros(Int, N)
    position = Vector{SVector{2,Float64}}(undef, N)
    for i in eachindex(frame)
        species[i] = parse(Int, split(frame[i], " ")[1])
        position[i] = fold_back(SVector{2,Float64}(parse.(Float64, split(frame[i], " ")[2:3])), box)
    end
    temperature = 0.231
    density = N / L^2
    system = System(position, species, density, temperature, JBB(); list_type=LinkedList)
    energy = mean(system.local_energy) / 2
    @test isapprox(energy, -2.676832, atol=1e-6)
end