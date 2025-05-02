

#%%imports
using Plots
using Random
using Revise

include(joinpath(@__DIR__, "../src/LVisP.jl"))
using .LVisP


#%%definitions
"""
    - function defining a gauss distribution
"""
function gaussian_pdf(x, μ, σ)
    f = 1 / (σ * sqrt(2π)) * exp(-(x - μ)^2 / (2σ^2))
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

#%%demos
nobjects = 12
time = Array.(eachcol(collect(-20:1:100) .* ones(nobjects)'))
passband_options = 0.4:0.3:4
ridxs = randperm(length(passband_options))  #for random sampling without replacement
passbands = passband_options[ridxs][1:nobjects]
# passbands = collect(range(0.4, 4.0, nobjects))
# passbands = [1,2,3,4] .- 0.4
f_peak = 20
t_peak = collect(range(0,40, nobjects)) .* 1
stretch0 = 5
stretch1 = 15
stretch2 = 40
flux = map(i -> lc_sim(time[i]; t_peak=t_peak[i], f_peak=f_peak, lambda=passbands[i], stretch0=stretch0, stretch1=stretch1, stretch2=stretch2, noiselevel=1), 1:nobjects)
# flux = map(i -> f_peak .* sin.((time[i] .* (2pi./collect(range(20, 60, nobjects))[i]))), 1:nobjects)

begin #full circle
    LVPC = LVisP.LVisPCanvas(
        thetalims=(0.2, 4.2), xticks=[-30, 0, 50, 100],
        thetaguidelims=(0,2pi), thetaplotlims=(0,3pi/2), xlimdeadzone=0.3, panelsize=pi/12,
        thetalabel="Passband Wavelength", xlabel="Time - Peak", ylabel="Flux",
        th_arrowlength=pi/3,
        panelbounds=false, ygrid=true,
        fontsizes=(thetalabel=15,ylabel=14),
        thetaarrowkwargs=(color=:black, alpha=0.3),
        thetaticklabelkwargs=(halign=:center,),
        thetalabelkwargs=(halign=:center,),
        xtickkwargs=(linecolor=:black, linealpha=0.3,),
        xticklabelkwargs=(rotation=-90, halign=:right, valign=:bottom),
        xlabelkwargs=(halign=:center,),
        ygridkwargs=(linecolor=:black, linealpha=0.3, linestyle=:solid,),
        yticklabelkwargs=(rotation=0,),
        # ylabelkwargs=(rotation=0,),
        panelboundskwargs=(linecolor=:black, linealpha=0.5, linestyle=:solid,),
    )
    
    # colors = Colorings.colorcode(passbands; cmap=:plasma, clims=extrema(passband_options))
    colors = palette(:rainbow, nobjects)
    plot_kwargs=[Dict(:lc=>colors[i], :label=>"$(round(passbands[i], digits=2))") for i in eachindex(passbands)]
    
    p = LVisP.plot(
        LVPC,
        passbands, time, flux;
        yticks=[0, 30, 60],
        plot_kwargs=plot_kwargs
    )
    
    display(p)
    # savefig(p, "$(@__DIR__)../../../report/gfx/temp_lvisp.png")
end

begin #advanced layout
    # LVPC = LVisP.LVisPCanvas(
    #     thetalims=(0.4, 4.0), xticks=([-20, 0, 50, 100],["A","Peak","C","D"]),
    #     # thetalims=(0.2, 4.2), xlims=(-30,70), ylims=(-10,100),
    #     thetaguidelims=(-pi/2,pi/2), thetaplotlims=(-0.9pi/2,0.9pi/2), xlimdeadzone=0.3, panelsize=pi/12,
    #     thetalabel="Passband Wavelength", xlabel="Time - Peak", ylabel="Flux",
    #     th_arrowlength=pi/3,
    #     panelbounds=true, ygrid=true,
    #     # fontsizes=(thetalabel=15,ylabel=14),
    #     # thetaarrowkwargs=(color=:black, alpha=0.3),
    #     # thetaticklabelkwargs=(halign=:center,),
    #     # thetalabelkwargs=(halign=:center,),
    #     # xtickkwargs=(linecolor=:black, linealpha=0.3,),
    #     xticklabelkwargs=(rotation=0, halign=:right, valign=:bottom),
    #     # xlabelkwargs=(halign=:center,),
    #     # ygridkwargs=(linecolor=:black, linealpha=0.3, linestyle=:solid,),
    #     yticklabelkwargs=(rotation=0,),
    #     # ylabelkwargs=(rotation=0,),
    #     # panelboundskwargs=(linecolor=:black, linealpha=0.5, linestyle=:solid,),
    # )
    
    # colors = Colorings.colorcode(passbands; cmap=:plasma, clims=extrema(passband_options))
    # plot_kwargs=[Dict(:lc=>colors[i], :label=>"$(round(passbands[i], digits=2))") for i in eachindex(passbands)]

    # p = plot(;size=(600,1200))
    # LVisP.plot_LVisPCanvas!(p, LVPC)
    # p = LVisP.plot(
    #     LVPC,
    #     passbands, time, flux;
    #     thetaticklabels=["PB $i" for i in eachindex(passbands)],
    #     yticks=[-10,0,50,80],
    #     plot_kwargs=plot_kwargs
    # )
    
    # display(p)
end

#%%