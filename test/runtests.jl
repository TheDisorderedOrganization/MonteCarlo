using SafeTestsets

@safetestset "Distribution Test" begin
    include("distribution_test.jl")
end

@safetestset "PGMC Test" begin
    include("pgmc_test.jl")
end