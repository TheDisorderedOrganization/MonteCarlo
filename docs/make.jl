using Documenter, MonteCarlo

makedocs(sitename="MonteCarlo",
pages = [
    "Home" => "index.md",
    "Manual" => Any[
        "man/system.md",
        "man/simulation.md",
    ],
    "API" => "lib/api.md",

],
)