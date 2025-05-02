
#%%imports
using CSV
using DataFrames
using Glob
using Plots
using Random

#%%definitions
"""
    - function to load all HEAD files in `path` into one single `DataFrame`

    Parameters
    ----------
        - `path`
            - `String`
            - path to the directory that hosts the csv files to be loaded
            - `n_files` files matching `"*HEAD*.csv"` will be read
        - `n_files`
            - `Int`, optional
            - number of files to ultimately load
            - the default is `-1`
                - loads all files matching `"*HEAD*.csv"`

    Raises
    ------

    Returns
    -------
        - `df_head`
            - `DataFrame`
            - all HEAD files loaded in the form of a `DataFrame`

    Dependencies
    ------------
        - `CSV`
        - `DataFrames`

    Comments
    --------
"""
function load_head2df(
    fns;
    n_files::Union{Int,Nothing}=nothing,
    )::DataFrame
    
    
    fns_head = sort(fns)
    n_files = isnothing(n_files) ? length(fns_head) : n_files

    dfs = DataFrame[]
    for fn in fns_head
        df = DataFrame(CSV.File(fn))
        src = contains(fn, "DES") ? "des" : "elasticc"
        df[!,:context] .= src
        rename!(df, lowercase.(names(df)))
        if in("snsubtype", names(df))
            select!(df, [:snid, :snsubtype, :context])
            rename!(df, :snsubtype => :sntype)
        else
            select!(df, [:snid, :sntype, :context])
        end
    
        push!(dfs,df)
    end
    # df_head = DataFrame.(CSV.File.(fns_head[1:n_files]))
    # df_head = select.(df_head, [:SNID, :SNSUBTYPE])
    df_head = vcat(dfs...)

    return df_head
end
"""
    - function to load all PHOT files in `path` into one single `DataFrame`

    Parameters
    ----------
        - `path`
            - `String`
            - path to the directory that hosts the csv files to be loaded
            - `n_files` files matching `"*PHOT*.csv"` will be read
        - `n_files`
            - `Int`, optional
            - number of files to ultimately load
            - the default is `-1`
                - loads all files matching `"*PHOT*.csv"`

    Raises
    ------

    Returns
    -------
        - `df_phot`
            - `DataFrame`
            - all PHOT files loaded in the form of a `DataFrame`

    Dependencies
    ------------
        - `CSV`
        - `DataFrames`

    Comments
    --------
"""
function load_phot2df(
    fns;
    n_files::Union{Int,Nothing}=nothing,
    )::DataFrame
    
    fns_phot = sort(fns)
    n_files = isnothing(n_files) ? length(fns_phot) : n_files
    
    dfs = DataFrame[]
    for fn in fns_phot
        df = DataFrame(CSV.File(fn))
        rename!(df, lowercase.(names(df)))
        select!(df, :snid, :mjd, :fluxcal, :fluxcalerr, :flt)
        
        dropmissing!(df)
        
        push!(dfs,df)
        
    end
    df_phot = vcat(dfs...)


    return df_phot
end
encoding_sntype = Dict(
    101=>"Ia", 120=>"IIP", 121=>"IIn", 122=>"IIL1", 123=>"IIL2", 132=>"Ib", 133=>"Ic",
    112=>"Ib/c", 113=>"II",
)


#%load files
fns_he_gp  = Glob.glob("*HEAD*.csv", joinpath(@__DIR__,"./gp/"))
fns_ph_gp  = Glob.glob("*PHOT*.csv", joinpath(@__DIR__,"./gp/"))
fns_ph_raw = Glob.glob("*PHOT*.csv", joinpath(@__DIR__,"./raw/"))
df_he_gp = load_head2df(fns_he_gp)
df_ph_gp = load_phot2df(fns_ph_gp)
df_ph_raw = load_phot2df(fns_ph_raw)
println(names(df_he_gp))
println(names(df_ph_gp))

# subset!(df_he_gp,
#     :sntype => t -> t .== 122
# )

#%%

snids = Dict{String,Vector}(
    "snia_des" => [31969503, 3787399, 13686088,],
    "snib_des" => [9635893, 11370314, 21794560,],
    "snic_des" => [15108182, 60158, 29508760,],
    "sniin_des" => [8777682, 2723412, 11500415],
    "sniip_des" => [18213902, 7137946, 30365194,],
    "sniil1_des" => [22502500, 33473486, 28191381,],
    "sniil2_des" => [29216742, 21951066, 1853138],
    "snii_elasticc" => [132358631, 124266324, 72147108,],
    "snibc_elasticc" => [114645810, 122276966, 120712717,],
)
snids = vcat(collect(values(snids))...)

# snids = unique(df_he_gp[!,:snid])
# ridxs = randperm(length(snids))[1:300]
# snids = snids[ridxs]
# println(unique(df_ph_gp[!,:flt]))

for snid in snids


    df_he_sn_gp = subset(df_he_gp, 
        :snid => s -> s .== snid,
    )
    df_lc_sn_gp = subset(df_ph_gp, 
        :snid => s -> s .== snid,
    )
    df_lc_sn_raw = subset(df_ph_raw, 
        :snid => s -> s .== snid,
    )
    sntype  = encoding_sntype[df_he_sn_gp[1,:sntype]]
    context = df_he_sn_gp[1,:context]


    sort!(df_lc_sn_gp,  [:flt,:mjd])
    sort!(df_lc_sn_raw, [:flt,:mjd])

    if length(unique(df_lc_sn_gp[!,:flt])) .>= 4

        p = plot(
            df_lc_sn_gp[!,:mjd], df_lc_sn_gp[!,:fluxcal],
            group=df_lc_sn_gp[!,:flt],
            lc=[palette(:rainbow, 6)...]',
            title="$(snid) ($(sntype), $(context))"
        )
        scatter!(
            df_lc_sn_raw[!,:mjd], df_lc_sn_raw[!,:fluxcal],
            group=df_lc_sn_raw[!,:flt],    
            mc=[palette(:rainbow, 6)...]',
        )

        display(p)
    end
end