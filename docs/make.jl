using Documenter
using Arianna

readme = read(joinpath(@__DIR__, "..", "README.md"), String)
readme_filtered = replace(readme, r"<.*?>" => "")
write(joinpath(@__DIR__, "src", "index.md"), readme_filtered)

makedocs(sitename="Arianna",
    format=Documenter.HTML(
        prettyurls=(get(ENV, "CI", nothing) == "true"),
        size_threshold_ignore=["api.md"],
    ),
    #modules=[Arianna],
pages = [
    "Home" => "index.md",
    "Manual" => Any[
        "man/system.md",
        "man/simulation.md",
    ],
    "API" => "lib/api.md",
],
)

# deploydocs(
#     repo="https://github.com/TheDisorderedOrganization/Arianna",
#     push_preview=true,
# )