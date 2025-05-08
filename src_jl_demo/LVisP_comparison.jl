#%%imports
using CSV
using DataFrames
using Glob
using NaNStatistics
using Plots
using Random
using Revise

include(joinpath(@__DIR__, "../src/LVisP.jl"))
using .LVisP

#%%definitions
"""
    - function defining a gaussian normaldistribution
"""
function gaussian_pdf(x, mu, sigma)
    f = 1 / (sigma * sqrt(2π)) * exp(-(x - mu)^2 / (2sigma^2))
    return f
end
"""
    - function to define a very simplistic phenomenological LC simulation
"""
function lc_sim(
    t::Vector;
    t_peak::Number, f_peak::Number,
    lambda::Number=1.0,
    stretch0::Number, stretch1::Number, stretch2::Number,
    noiselevel::Number=0.0,
    )::Vector

    f = (gaussian_pdf.(t, t_peak - stretch0/2, stretch1) .+ gaussian_pdf.(t, t_peak + stretch0/2, stretch2))
    f = f_peak .* f ./ maximum(f) .+ noiselevel .* randn(length(t))
    f .*= lambda #wavelength dependent scaling
    return f
end
"""
    - function to evaluate a sin with period `p` and `offset`
"""
function sin_sim(
    t::Real;
    f_peak::Real,
    p::Real, offset::Real=0.,
    noiselevel::Number=0.0
    )::Real
    f = f_peak * sin(t * 2pi/p + offset) .+ noiselevel .* randn()
    return f
end
"""
    - function to generate `nobjects` different dataseries according to `opt` (one of `"lc"`, `"sin"`)
"""
function simulate(
    nobjects::Int=12;
    opt::String="lc"
    )
    x = Array.(eachcol(collect(-20:0.2:100) .* ones(nobjects)'))
    theta_options = 0.4:0.3:4
    ridxs = randperm(length(theta_options))  #for random sampling without replacement
    theta = theta_options[ridxs][1:nobjects]
    
    if opt == "lc"
        t_peak = collect(range(0,40, nobjects)) .* 1
        y = map(i -> lc_sim(x[i]; t_peak=t_peak[i], f_peak=20, lambda=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=1.0), 1:nobjects)
        y_nonoise = map(i -> lc_sim(x[i]; t_peak=t_peak[i], f_peak=20, lambda=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=0.0), 1:nobjects)
    else
        theta .*= 10
        y = map(i -> sin_sim.(x[i]; f_peak=1, p=theta[i], offset=0.0, noiselevel=0.15), 1:nobjects)
        y_nonoise = map(i -> sin_sim.(x[i]; f_peak=1, p=theta[i], offset=0.0, noiselevel=0.0), 1:nobjects)
    end

    df_raw = DataFrame(
        :period=>vcat(repeat(theta, 1, length(x[1]))'...),
        :time=>vcat(x...),
        :amplitude=>vcat(y...),
        :amplitude_e=>NaN,
        :processing=>"raw",
    )
    df_pro = DataFrame(
        :period=>vcat(repeat(theta, 1, length(x[1]))'...),
        :time=>vcat(x...),
        :amplitude=>vcat(y_nonoise...),
        :amplitude_e=>NaN,
        :processing=>"nonoise",

    )
    df = vcat(df_raw, df_pro)

    # p = scatter(
    #     df_raw[!,:time], df_raw[!,:amplitude],
    #     group=df_raw[!,:period],
    #     )
    # plot!(p,
    #     df_pro[!,:time], df_pro[!,:amplitude],
    #     group=df_pro[!,:period],

    # )
    # display(p)

    return df
end


#%%demos

#%%data loading
fnames = Glob.glob("*.csv", joinpath(@__DIR__,"../data/"))
append!(fnames, [joinpath(@__DIR__, "../data/lc_simulated.jl"), joinpath(@__DIR__, "../data/sin_simulated.jl")])    #append pseudo filenames for data generated in this script
println(fnames)
fname = fnames[31]

#deal with on-the-fly data generation (pseudo filenames)
if fname == joinpath(@__DIR__, "../data/lc_simulated.jl")
    df = simulate(10; opt="lc")
elseif fname == joinpath(@__DIR__, "../data/sin_simulated.jl")
    df = simulate(10; opt="sin")
else
    df = DataFrame(CSV.File(fname))
end

parts = split(fname, r"[\_\.]")
survey = parts[end-1]
otype = parts[end-2]

df_raw = subset(df, :processing => p -> p .== "raw")    #raw
df_pro = subset(df, :processing => p -> p .!= "raw")    #processed


theta_raw = sort(unique(df_raw[!,1]))
x_raw = map(x -> Vector(x[!,2]), collect(groupby(df_raw, 1; sort=true)))
y_raw = map(y -> Vector(y[!,3]), collect(groupby(df_raw, 1; sort=true)))
theta_pro = sort(unique(df_pro[!,1]))
x_pro = map(x -> Vector(x[!,2]), collect(groupby(df_pro, 1; sort=true)))
y_pro = map(y -> Vector(y[!,3]), collect(groupby(df_pro, 1; sort=true)))

x_ref = nanminimum.(x_raw)
y_ref = nanminimum.(y_raw)

x_raw = map((x, xr) -> x .- xr, x_raw, x_ref)
x_pro = map((x, xr) -> x .- xr, x_pro, x_ref)
# y_raw = map((y, yr) -> y .- yr, y_raw, y_ref)
# y_pro = map((y, yr) -> y .- yr, y_pro, y_ref)

# y_raw = map(y -> y .- nanminimum(y), y_raw)
# y_pro = map(y -> y .- nanminimum(y), y_pro)

# theta_raw   = theta_raw[1:1]
# theta_pro   = theta_pro[1:1]
# x_raw       = x_raw[    1:1]
# x_pro       = x_pro[    1:1]
# y_raw       = y_raw[    1:1]
# y_pro       = y_pro[    1:1]

# p = scatter(x_raw, y_raw)
# plot!(p, x_pro, y_pro)
# display(p)


#%%get stats
unique_thetas = unique(theta_raw)
nthetas = length(unique_thetas)

display(unique_thetas)
display(nthetas)


#%%get required parameters
thetalims = (minimum(theta_raw), maximum(theta_raw))
xticks = collect(range(floor(minimum(minimum.(x_raw))), ceil(maximum(maximum.(x_raw))), 5))
yticks = collect(range(floor(minimum(minimum.(y_pro))), ceil(maximum(maximum.(y_pro))), 3))

# xticks = (xticks, ["x1 ", "x2 ", "x3 ", "x4 ", "x5 "])
# yticks = (yticks, string.(log10.(yticks)))
# yticks = nothing
# yticks = [17, 21]

# display(thetalims)
# display(xticks)
# display(extrema(x_pro[1]))


#%%plotting
#LVisP
panelsize = pi/(2*nthetas)
LVPC = LVisP.LVisPCanvas(
    thetalims=thetalims, xticks=xticks,
    thetaguidelims=(-pi/2,pi/2), thetaplotlims=(-pi/2+panelsize/2,pi/2-panelsize/2), xlimdeadzone=0.3, panelsize=panelsize,
    thetalabel="Passband Wavelength [nm]", xlabel="\n\nMJD - min(MJD) [d]" * " "^2, ylabel="",
    th_arrowlength=pi/3,
    panelbounds=true, ygrid=true,
    fontsizes=(thetalabel=9, ylabel=9, xlabel=9, thetaticklabel=9, xticklabel=9, yticklabel=7),
    thetaarrowkwargs=(color=:black, alpha=0.3),
    thetaticklabelkwargs=(halign=:center,),
    thetalabelkwargs=(halign=:center,),
    xtickkwargs=(linecolor=:black, linealpha=0.3,),
    xticklabelkwargs=(rotation=0, halign=:right, valign=:bottom),
    xlabelkwargs=(rotation=-90, halign=:right, valign=:top),
    ygridkwargs=(linecolor=:black, linealpha=0.3, linestyle=:solid,),
    yticklabelkwargs=(rotation=0,),
    # ylabelkwargs=(rotation=0, valign=:center, halign=:left),
    panelboundskwargs=(linecolor=:black, linealpha=0.5, linestyle=:solid,),
)
    
colors = palette(:rainbow, nthetas)

p = LVisP.plot(
    LVPC,
    theta_raw, x_raw, y_raw;
    yticks=yticks,
    thetaticklabels=["$thtl nm\nFluxcal []" for thtl in unique_thetas],
    plot_kwargs=[
    Dict(
        :mc=>colors[i], :label=>"",
        :seriestype=>:scatter,
    ) for i in eachindex(unique_thetas)]
)
plot!(p;
    size=(1200,1200),
    leftmargin=0Plots.mm, rightmargin=15Plots.mm,
    topmargin=15Plots.mm, bottommargin=0Plots.mm,
)

LVisP.plot!(
    LVPC,
    theta_pro, x_pro, y_pro;
    yticks=yticks,
    thetaticklabels=nothing,
    data_only=true,
    plot_kwargs=[Dict(
        :lc=>colors[i], :label=>"$(unique_thetas[i]) nm",
        :seriestype=>:path,
        :title=>"$otype ($survey)",
    ) for i in eachindex(unique_thetas)]
)


#traditional
p_trad1 = scatter(x_raw, y_raw;
    label="", mc=collect(colors)',
    xlabel="MJD-min(MJD) [d]",
    ylabel="Fluxcal []",
)
plot!(p_trad1, x_pro, y_pro; label=string.(theta_pro') .* " nm", lc=collect(colors)')

p_trad2 = plot(
    [plot(
        [x_raw[i], x_pro[i]], [y_raw[i], y_pro[i]];
        mc=colors[i], lc=colors[i],
        seriestype=[:scatter :path],
        label=["" "$(unique_thetas[i]) nm"],
        xlabel="MJD-min(MJD) [d]",
        ylabel="Fluxcal []",
        ylims=extrema(yticks),
    ) for i in eachindex(unique_thetas)]...
)

#combine LVisP and trad
p_comp = plot(p, p_trad1, p_trad2;
    size=(1200,900),
    layout=@layout[a [b;c]],
    rightmargin=5Plots.mm, leftmargin=-10Plots.mm,
    bottommargin=5Plots.mm,
)

display(p_comp)
savefig(p_comp, replace(fname, "./data/"=>"./gfx/", ".csv"=>".png", ".jl"=>".png"))
