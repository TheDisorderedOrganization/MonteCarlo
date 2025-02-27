using Documenter
using Arianna

readme = read(joinpath(@__DIR__, "..", "README.md"), String)
readme = replace(readme, r"<.*?>" => "")
readme = readme[findfirst("Arianna", readme)[1]:end]
readme = "# Arianna\n *A system-agnostic approach to Monte Carlo simulations*" * readme
write(joinpath(@__DIR__, "src", "index.md"), readme)

makedocs(sitename="Arianna",
    format=Documenter.HTML(
        prettyurls=(get(ENV, "CI", nothing) == "true"),
        size_threshold_ignore=["api.md"],
        sidebar_sitename=false,
    ),
    #modules=[Arianna],
pages = [
    "Home" => "index.md",
    "Manual" => Any[
        "man/montecarlo.md",
        "man/system.md",
        "man/simulation.md",
    ],
    "API" => "api.md",
],
)

# deploydocs(
#     repo="https://github.com/TheDisorderedOrganization/Arianna",
#     push_preview=true,
# )