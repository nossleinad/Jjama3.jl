using Jjama3
using Documenter

DocMeta.setdocmeta!(Jjama3, :DocTestSetup, :(using Jjama3); recursive=true)

makedocs(;
    modules=[Jjama3],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="Jjama3.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Jjama3.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Jjama3.jl",
    devbranch="main",
)
