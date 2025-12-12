using TVGLMShrink
using Documenter

DocMeta.setdocmeta!(TVGLMShrink, :DocTestSetup, :(using TVGLMShrink); recursive=true)

makedocs(;
    modules=[TVGLMShrink],
    authors="Mattias Villani",
    sitename="TVGLMShrink.jl",
    format=Documenter.HTML(;
        canonical="https://mattiasvillani.github.io/TVGLMShrink.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattiasvillani/TVGLMShrink.jl",
    devbranch="main",
)
