using Documenter
using Mehrotra

makedocs(
    sitename = "Mehrotra",
    format = Documenter.HTML(prettyurls = false),  # optional
    pages = [
        "Introduction" => "index.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/simon-lc/Mehrotra.jl.git",
)
