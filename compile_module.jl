#%%imports
using Pkg
using TOML
using UUIDs

#module
include("./src/LVisP.jl")
using .LVisP

#%%definitions
"""
    - function to build the project into an installable module

    Parameters
    ----------
        - `project_toml`
            - `String`, optional
            - path to `Project.toml` file of the module
            - the default is `./Project.toml`

    Raises
    ------

    Returns
    -------

    Dependencies
    ------------
        - `Pkg`
        - `TOML`
        - `UUIDs`

    Comments
    --------

"""
function build_project_toml(
    project_toml::String="./Project.toml",
    )

    #build first template (only contains dependencies)
    Pkg.activate(".")   #activate project env
    Pkg.resolve()       #ensure dependencies are up to date
    Pkg.status()        #list dependencies

    #read file
    toml = TOML.parsefile(project_toml)
    
    #add additional metadata
    toml["name"]            = LVisP.__modulename__
    toml["version"]         = LVisP.__version__
    toml["description"]     = "repository of some useful code snippets in various programming languages."
    toml["author"]          = LVisP.__author__
    toml["author_email"]    = LVisP.__author_email__
    toml["maintainer"]      = LVisP.__maintainer__
    toml["maintainer_email"]= LVisP.__maintainer_email__
    toml["url"]             = LVisP.__url__
    toml["credits"]         = LVisP.__credits__
    toml["last_changed"]    = LVisP.__last_changed__
    toml["uuid"]            = string(UUIDs.uuid4())

    println(toml)

    #add `[compat]` section
    toml["compat"] = Dict(
        "julia" => string(VERSION)              #juia version
    )

    # Write back to `Project.toml`
    open(project_toml, "w") do io
        TOML.print(io, toml)
    end

end


#%%control

#%%main
build_project_toml(
    "./Project.toml"
)