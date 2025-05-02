
"""
    - building blocks that are used in multiple files of the project
    - use by simply including and importing

    ```julia
        include(joinpath(@__DIR__, "_projectbuildingblocks.jl"))
        using .ProjectBuildingBlocks: ProjectBuildingBlocks as pbb
    ```
"""

module ProjectBuildingBlocks

#%%imports

#%%constants

#%%definitions
"""
    - struct description

    Fields
    ------
        - `x1`
            - `AbstractArray{Number,3}`, optional
            - description
            - the default is `Array{Number}(undef,0,0,0)`
        - `x2`
            - `Number`, optional
            - description
            - the default is `0.0`
    Methods
    -------
        - 

    Comments
    --------
"""
struct StructTemplate{T <: Number}
    x1::AbstractArray{T,3}
    x2::T

    #inner constructor (to allow default values)
    function StructTemplate(
        x1::AbstractArray{T,3}=Array{Number}(undef,0,0,0);    #positional argument
        x2::T=0.0,                                        #named argument
        ) where {T <: Number}

        new{T}(x1, x2)
    end

end

"""
    - function description

    Parameters
    ----------
        - `parg1`
            - `Real`
            - description
        - `parg2`
            - `Symbol`, optional
            - description
            - the default is `:x2` 
        - `narg1`
            - `Symbol`
            - description
        - `verbose`
            - `Int`, optional
            - verbosity level
            - the default is `0`
    
    Raises
    ------

    Returns
    -------
        - `out`
            - `Any`
            - description

    Comments
    --------
"""
function function_template(
    parg1::T,                   #positional argument
    parg2::Symbol=:x2;          #positional argument
    narg1::String,              #named argument
    verbose::Int=0,             #named argument
    )::Any where {T <: Real}
    
    out = nothing

    return out 
end

end #module
