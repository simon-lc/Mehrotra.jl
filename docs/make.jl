using Documenter
using Mehrotra

makedocs(
    sitename = "Mehrotra",
    format = Documenter.HTML(prettyurls = false),  # optional
    pages = [
        "Get Started" => "index.md"
        "Solver" => "solver.md"
        "Examples" => "examples.md"
        "API Documentation" => "api.md"
        "Contributing" => "contributing.md"
        "Citing" => "citing.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/simon-lc/Mehrotra.jl.git",
)
