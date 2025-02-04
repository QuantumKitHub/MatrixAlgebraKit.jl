# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
    Pkg.resolve()
    Pkg.instantiate()
end

using MatrixAlgebraKit
using Documenter

plugins = []

# using DocumenterCitations
# bibpath = joinpath(@__DIR__, "src", "assets", "mpskit.bib")
# bib = CitationBibliography(bibpath; style=:authoryear)
# push!(plugins, bib)

# using DocumenterInterLinks
# links = InterLinks("TensorKit" => "https://jutho.github.io/TensorKit.jl/stable/",
#                    "TensorOperations" => "https://jutho.github.io/TensorOperations.jl/stable/",
#                    "KrylovKit" => "https://jutho.github.io/KrylovKit.jl/stable/",
#                    "BlockTensorKit" => "https://lkdvos.github.io/BlockTensorKit.jl/dev/")
# push!(plugins, links)

DocMeta.setdocmeta!(MatrixAlgebraKit, :DocTestSetup, :(using MatrixAlgebraKit); recursive=true)

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/physics"]),
                           :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                                        "tags" => "ams",
                                        "packages" => ["base", "ams", "autoload", "physics"])))
makedocs(;
         sitename="MatrixAlgebraKit.jl",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", nothing) == "true",
                                mathengine,
                                size_threshold=512000),
         pages=["Home" => "index.md", "Interface" => "interface.md", "Library" => "library.md"],
         checkdocs=:exports,
         doctest=true,
         plugins)

deploydocs(; repo="github.com/QuantumKitHub/MatrixAlgebraKit.jl.git", push_preview=true)
