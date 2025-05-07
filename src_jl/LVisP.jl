"""
    - module implementing an LVisPlot (Line Visualisation Plot)
        - visualization of 2.5 dimensions by projecting into 2d
    
    Constants
    ---------
    
    Structs
    -------
        - `LVisPCanvas`
        - `LVisPPanel`
    
    Functions
    ---------
        - `plot_LVisPCanvas!()`
        - `plot_LVisPPanel!()`

    Extended Functions
    ------------------
        - `Plots.plot()`
        - `Plots.plot!()`
    
    Dependencies
    ------------
        - `NaNStatistics`
        - `Plots`


    Comments
    --------
        - if you need further customization it is recommended to use `plot_LVisPCanvas!()` combined with `plot_LVisPPanel!()`
            - check the sourcecode of `plot!()` for some insight into how

    Examples
    --------
        - see [LVisP_demo.jl](../../src_demo/LVisP_demo.jl)
"""

module LVisP

#%%imports
using NaNStatistics
using Plots

#import for extending
import Plots: plot, plot!

#intradependencies

#%%exports
export LVisPCanvas
export LVisPPanel
export plot_LVisPCanvas!
export plot_LVisPPanel!
export plot, plot!

#%%definitions
############################################
#helper functions
"""
    - function to convert carthesian coordinates to polar

    Parameters
    ----------
        - `x`
            - `Real`
            - x-coordinate
        - `y`
            - `Real`
            - y-coordinate
        
    Raises
    ------
            
    Returns
    -------
        - `r`
            - `Real`
            - radius
        - `theta`
            - `Real`
            - azimuthal angle

    Comments
    --------
"""
function carth2polar(x::Real, y::Real)::Tuple{Real,Real}
    r = sqrt(x.^2 + y.^2)
    theta = atan(y./x)
    if theta < 0.0 theta += 2pi
    elseif signbit(theta) theta += pi
    else #don't change anything for `theta >= 0`
    end
    return r, theta
end

"""
    - function to convert polar coordinates to carthesian

    Parameters
    ----------
        - `r`
            - `Real`
            - radius
        - `theta`
            - `Real`
            - azimuthal angle
        
    Raises
    ------
            
    Returns
    -------
        - `x`
            - `Real`
            - x-coordinate
        - `y`
            - `Real`
            - y-coordinate
    Comments
    --------
"""
function polar2carth(r::Real, theta::Real)::Tuple{Real,Real}
    x = r * cos(theta)
    y = r * sin(theta)
    return x, y
end

"""
    - function implementing min-max-scaling

    Parameters
    ----------
        - `x`
            - `Vector`
            - vector to be scaled
        - `xmin`
            - `Real`
            - minimum of target range
        - `xmax`
            - `Real`
            - maximum of target range
        - `xmin_ref`
            - `Real`, optional
            - reference value to use as lower bound for projection
            - the default is `nothing`
                - set to `nanminimum(x)`
        - `xmax_ref`
            - `Real`, optional
            - reference value to use as upper bound for projection
            - the default is `nothing`
                - set to `nanmaximum(x)`

    Raises
    ------

    Returns
    -------
        - `x_scaled`
            - `Vector`
            - scaled version of `x`
            - has values spanning the interval [xmin,xmax]

    Comments
    --------
"""
function minmaxscale(x::Vector, xmin::Real, xmax::Real; xmin_ref::Union{Real,Nothing}=nothing, xmax_ref::Union{Real,Nothing}=nothing)
    xmin_ref = isnothing(xmin_ref) ? nanminimum(x) : xmin_ref
    xmax_ref = isnothing(xmax_ref) ? nanmaximum(x) : xmax_ref
    
    x_scaled = (x .- xmin_ref) ./ (xmax_ref - xmin_ref)
    x_scaled = x_scaled .* (xmax - xmin) .+ xmin
    return x_scaled
end

"""
    - helper function to orient labels in a way that they are readable
    - flips label if some condition is met (if it would be upside down)

    Parameters
    ----------
        - `theta`
            - `Real`
            - azimuthal angle in polar coordinates
            - provided in degrees

    Raises
    ------

    Returns
    -------
        - `theta_corr`
            - `Real`
            - corrected version of `theta`
                - rotated by `pi`
                - prevents upside-down labels

    Comments
    --------    
"""
function correct_labelrotation(theta::Real)
    if sin(theta * pi/180) < 0  #use `sin` to also account for negative values
        return theta + 180
    else
        return theta
    end
end

############################################
#main
"""
    - struct containing global specifications
        - applied to guides
        - applied to all panels that will eventually be added
    - parent to `LVisPPanel`

    Fields
    ------
        - `thetalims`
            - `Tuple{Real,Real}`
            - axis limits applied to `theta`
                - `thetalims[1]` corresponds to `theta=0` on the unit-circle
                - `thetalims[2]` corresponds to `theta=2pi` on the unit-circle
        - `xticks`
            - `Vector{Real}`, `Tuple{Vector{Real},Vector{Any}}`
            - ticks (circles) to draw for the x-axis
            - also defines axis limits applied to `x`
                - i.e., in radial direction
                - `minimum(xticks[1])` corresponds to the end of `xlimdeadzone`
                - `maximum(xticks[1])` corresponds to the value plotted at the outer bound of the LVisPlot
            - if `Vector{Real}`
                - will use `xticks` as labels as well
            - if `Tuple{Vector{Real},Vector{Any}}`
                - will use `xticks[2]` as ticklabels
        - `thetaguidelims`
            - `Tuple{Real,Real}`, optional
            - range to be spanned by the entire plot guides
                - only affects the background grid
            - in radians
            - the default is `(0,2pi)`
                - an entire circle will be plotted
        - `thetaplotlims`
            - `Tuple{Real,Real}`, optional
            - range to be spanned by the individual theta-panels
            - sets the reference point for `thetalims`
                - `thetalims[1]` will be plotted at `thetaplotlims[1]`
                - `thetalims[2]` will be plotted at `thetaplotlims[2]`
            - in radians
            - the default is `nothing`
                - will be set to `thetaguidelims`
        - `xlimdeadzone`
            - `Real`, optional
            - amount of space to leave empty in the center of the plot
            - provided as a fraction of the entire plot-radius
            - used to reduce projection effects at tiny radii
            - the default is `0.3`
                - 30% of the radial direction are left empty
        - `panelsize`
            - `Real`, optional
            - size of individual panels in radians
            - defines the size of each panel
            - the entire figure can allocate `(thetaguidelims[2]-thetaguidelims[1])/panelsize` evenly distributed, nonoverlapping panels
            - the default is `pi/8`
                - each panel spans a sector with angular size of `pi/8`
                - a figure with a full circle can allocate 16 evenly distributed, nonoverlapping panels
        - `thetalabel`
            - `String`, optional
            - label to show for the theta-arrow
            - the default is `""`
        - `xlabel`
            - `String`, optional
            - label of the x-axis
            - the default is `""`
        - `ylabel`
            - `String`, optional
            - label of the y-axis
            - the default is `""`
        - `th_arrowlength`
            - `Real`, optional
            - length of the arrow indicating the `theta`-coordinate
            - given in radians
            - the default is `pi/4`
        - `panelbounds`
            - `Bool`, optional
            - whether to show bounds of the individual panels when rendering
            - the default is `false`
        - `ygrid`
            - `Bool`, optional
            - whether to show ticks and gridlines for y-values
            - the default is `true`
        - `fontsizes`
            - `NamedTuple`, optional
            - font size to use in the plot
            - the default is `nothing`
                - will be set to `(thetaticklabel=12,thetalabel=15,xticklabel=10,xlabel=12,yticklabel=10,ylabel=12)`
        - `thetaarrowkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `plot!()` and `quiver!()` when drawing the arrow indicating the theta-axis
            - used for styling
            - the default is `(color=:black, alpha=0.3)`
        - `thetaticklabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the ticklabels of the theta-axis
            - used for styling
            - the default is `(halign=:center,)`
        - `thetalabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the axis label of the theta-axis
            - used for styling
            - the default is `(halign=:center,)`
        - `xtickkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `plot!()` when drawing xticks (circles)
            - used for styling
            - the default is `(linecolor=:black, linealpha=0.3,)`
        - `xticklabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the ticklabels of the x-axis
            - used for styling
            - the default is `(rotation=-90, halign=:right, valign=:bottom)`
        - `xlabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the axis label of the x-axis
            - used for styling
            - the default is `(halign=:center,)`
        - `ygridkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `plot!()` when drawing the y-grid of each panel
            - applied to all panels
            - used for styling
            - the default is `(linecolor=:black, linealpha=0.3, linestyle=:solid,)` 
        - `yticklabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the ticklabels of the y-axis
            - applied to all panels
            - used for styling
            - the default is `NamedTuple()`
        - `ylabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the axis label of the y-axis
            - applied to all panels
            - used for styling
            - the default is `NamedTuple()`
        - `panelboundskwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `plot!()` when drawing bounds of each panel
            - used for styling
            - the default is `(linecolor=:black, linealpha=0.4, linestyle=:solid,)`
        
    Infered Fields
    --------------
        - `xlims`
            - `Tuple{Real,Real}`
            - axis limits applied to `x`
                - i.e., in radial direction
                - `xlims[1]` corresponds to the value plotted at the end of `xlimdeadzone`
                - `xlims[2]` corresponds to the value plotted at the outer bound of the LVisPlot
        - `xlimrange`
            - `Union{Real,Nothing}`, optional
            - range of x-values
            - convenience field for later use
                - relative definitions of plot elements

    Methods
    -------
        - `add_xaxis!()`
        - `add_thetalabel!()`
        - `plot_LVisPCanvas!()`
        - `get_xy2plot()`
        - `get_yticks()`
        - `get_thetaslice()`
        - `get_ygrid()`
        - `plot_LVisPPanel!()`

    Comments
    --------
"""
struct LVisPCanvas
    
    thetalims::Tuple{Real,Real}
    xticks::Tuple{Vector{Real},Vector{<: Any}}
    
    thetaguidelims::Tuple{Real,Real}
    thetaplotlims::Tuple{Real,Real}
    xlimdeadzone::Real
    panelsize::Real

    thetalabel::String
    xlabel::String
    ylabel::String
    
    th_arrowlength::Real
    
    panelbounds::Bool
    ygrid::Bool

    fontsizes::NamedTuple

    thetaarrowkwargs::NamedTuple
    thetaticklabelkwargs::NamedTuple
    thetalabelkwargs::NamedTuple
    
    xtickkwargs::NamedTuple
    xticklabelkwargs::NamedTuple
    xlabelkwargs::NamedTuple
    
    ygridkwargs::NamedTuple
    yticklabelkwargs::NamedTuple
    ylabelkwargs::NamedTuple
    
    panelboundskwargs::NamedTuple

    #infered attributes
    xlims::Tuple{Real,Real}
    xlimrange::Union{Real,Nothing}

    function LVisPCanvas(;
        thetalims::Tuple{Real,Real}, xticks::Union{Vector{T},Tuple{Vector{T},Vector{U}}},
        thetaguidelims::Tuple{Real,Real}=(0,2pi), thetaplotlims::Union{Tuple{Real,Real},Nothing}=nothing, xlimdeadzone::Real=0.3, panelsize::Real=pi/8,
        thetalabel::String="", xlabel::String="", ylabel::String="",
        th_arrowlength::Real=pi/4,
        panelbounds::Bool=false, ygrid::Bool=true,
        fontsizes::Union{NamedTuple,Nothing}=nothing,
        thetaarrowkwargs::NamedTuple=(color=:black, alpha=0.3),
        thetaticklabelkwargs::NamedTuple=(halign=:center,),
        thetalabelkwargs::NamedTuple=(halign=:center,),
        xtickkwargs::NamedTuple=(linecolor=:black, linealpha=0.3,),
        xticklabelkwargs::NamedTuple=(rotation=-90, halign=:right, valign=:bottom),
        xlabelkwargs::NamedTuple=(halign=:center,),
        ygridkwargs::NamedTuple=(linecolor=:black, linealpha=0.3, linestyle=:solid,),
        yticklabelkwargs::NamedTuple=NamedTuple(),
        ylabelkwargs::NamedTuple=NamedTuple(),
        panelboundskwargs::NamedTuple=(linecolor=:black, linealpha=0.4, linestyle=:solid,),
        ) where {T <: Real, U <: Any}

        #default attributes
        xticks = isa(xticks, Vector) ? (xticks, xticks) : xticks
        thetaplotlims = isnothing(thetaplotlims) ? thetaguidelims : thetaplotlims
        if ~in(:rotation, keys(xlabelkwargs))
            xlabelkwargs = merge(xlabelkwargs,(rotation=correct_labelrotation(thetaguidelims[1]*180/pi),))
        end
        if ~in(:rotation, keys(thetalabelkwargs))
            thetalabelkwargs = merge(thetalabelkwargs,(rotation=:auto,))
        end
        fontsizes_default = (thetaticklabel=12,thetalabel=15,xticklabel=10,xlabel=12,yticklabel=10,ylabel=12)
        if isnothing(fontsizes)
            fontsizes = fontsizes_default
        else
            fontsizes = merge(fontsizes_default, fontsizes)
        end

        #cehcks
        @assert (0 <= xlimdeadzone)&(xlimdeadzone < 1) "`xlimdeadzone` has to be a `Real` between `0` and `1`"
        
        begin #infered attributes
            xlims = extrema(xticks[1])
            xlimrange = maximum(xticks[1]) - minimum(xticks[1])   # =xlimrange
        end


        #create new
        new(
            thetalims, xticks,
            thetaguidelims, thetaplotlims, xlimdeadzone, panelsize,
            thetalabel, xlabel, ylabel,
            th_arrowlength,
            panelbounds, ygrid,
            fontsizes,
            thetaarrowkwargs, thetaticklabelkwargs, thetalabelkwargs,
            xtickkwargs, xticklabelkwargs, xlabelkwargs,
            ygridkwargs, yticklabelkwargs, ylabelkwargs,
            panelboundskwargs,
            xlims, xlimrange,
        )
    end
end

"""
    - method to add the x-axis (in form of concentric circles)

    Parameters
    ----------
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant settings
        - `plt`
            - `Plots.Plot`
            - plot instance to plot into

    Raises
    ------

    Returns
    -------

    Comments
    --------
"""
function add_xaxis!(
    LVPC::LVisPCanvas,
    plt::Plots.Plot;
    )


    #xticks
    th_circ = range(LVPC.thetaguidelims[1], LVPC.thetaguidelims[2], 100)
    r_circ = (LVPC.xticks[1] .- minimum(LVPC.xticks[1]))
    r_circ = minmaxscale(r_circ, maximum(r_circ).* LVPC.xlimdeadzone, maximum(r_circ))
    circles_x = r_circ .* cos.(th_circ)'
    circles_y = r_circ .* sin.(th_circ)'

    circles_x = hcat(circles_x[1,1] .+ zeros(length(r_circ),1), circles_x, zeros(length(r_circ),1) .+ circles_x[1,end]) #add start and endpoint of innermost circle (to ensure circles connect at innermost circle)
    circles_y = hcat(circles_y[1,1] .+ zeros(length(r_circ),1), circles_y, zeros(length(r_circ),1) .+ circles_y[1,end]) #add start and endpoint of innermost circle (to ensure circles connect at innermost circle)
    circles_x[1:end-1,[1,end]] .= NaN   #set to NaN to force breaks
    circles_y[1:end-1,[1,end]] .= NaN   #set to NaN to force breaks
    
    #xticklabels
    xtickpos_x = circles_x[:,2] 
    xtickpos_y = circles_y[:,2]
    xticklabs = LVPC.xticks[2]

    #xlabel
    xlabpos_x = xtickpos_x[end] - 0.05*LVPC.xlimrange
    xlabpos_y = xtickpos_y[end] - 0.05*LVPC.xlimrange

    #plotting
    plot!(plt, circles_x', circles_y'; label="", LVPC.xtickkwargs...) #xticks (circles)
    annotate!(plt, xtickpos_x, xtickpos_y, text.(xticklabs, LVPC.fontsizes[:xticklabel]; LVPC.xticklabelkwargs...))                                                   #xtick labels
    annotate!(plt, xlabpos_x, xlabpos_y, text(LVPC.xlabel, LVPC.fontsizes[:xlabel]; LVPC.xlabelkwargs...)) #xlabel
end

"""
    - method to add an arrow and label indicating the theta-axis

    Parameters
    ----------
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant settings
        - `plt`
            - `Plots.Plot`
            - plot instance to plot into

    Raises
    ------

    Returns
    -------

    Comments
    --------
"""
function add_thetalabel!(
    LVPC::LVisPCanvas,
    plt::Plots.Plot;
    )
    #indicator
    th_arrow = range(LVPC.thetaguidelims[1], LVPC.thetaguidelims[1]+LVPC.th_arrowlength,100)
    x_arrow = 1.4.*LVPC.xlimrange .* cos.(th_arrow)
    y_arrow = 1.4.*LVPC.xlimrange .* sin.(th_arrow)
    dx_arrow = -1e-5.*LVPC.xlimrange .* sin.(th_arrow[end])
    dy_arrow =  1e-5.*LVPC.xlimrange  .* cos.(th_arrow[end])
    
    #label
    th_label_x = 1.45 * LVPC.xlimrange * cos(nanmean(th_arrow))
    th_label_y = 1.45 * LVPC.xlimrange * sin(nanmean(th_arrow))
    
    #get correct label rotation
    throt = LVPC.thetalabelkwargs.rotation == :auto ? correct_labelrotation(nanmean(th_arrow)/pi*180) - 90 : LVPC.thetalabelkwargs.rotation
    thetalabelkwargs = merge(LVPC.thetalabelkwargs, (rotation=throt,))

    #plotting
    plot!(plt, x_arrow, y_arrow; label="", LVPC.thetaarrowkwargs...)                                                 #arrowtail for theta-label
    quiver!(plt, [x_arrow[end]], [y_arrow[end]]; quiver=([dx_arrow], [dy_arrow]), arrow=true, LVPC.thetaarrowkwargs...)  #arrowhead for theta-label
    annotate!(th_label_x, th_label_y, text(LVPC.thetalabel, LVPC.fontsizes[:thetalabel]; rotation=throt, thetalabelkwargs...))                #theta label
end

"""
    - method to combine all different guides into the base canvas
    - renders the canvas

    Parameters
    ----------
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant settings
        - `plt`
            - `Plots.Plot`
            - plot instance to plot into

    Raises
    ------

    Returns
    -------

    Comments
    --------
"""
function plot_LVisPCanvas!(plt::Plots.Plot,
    LVPC::LVisPCanvas;
    )


    begin   #disable some default settings
        plot!(plt;
            grid=false, xaxis=false, yaxis=false,
            aspect_ratio=:equal
        )
    end

    #adding elements
    add_xaxis!(LVPC, plt)
    add_thetalabel!(LVPC, plt)

end

"""
    - struct containing global specifications
        - applied to guides
        - applied to all panels that will eventually be added
    - parent to `LVisPPanel`

    Fields
    ------
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant global settings        
        - `theta`
            - `Real`
            - theta coordinate
            - specifies where along the circle the panel will be positioned
            - i.e. third variable
        - `x`
            - `Vector`
            - x-values of the dataseries
        - `y`
            - `Vector`
            - y-values of the dataseries
        - `thetaticklabel`
            - `String`, optional
            - ticklabel to use instead of `theta`
            - the default is `nothing`
                - will use `theta` as the label for the panel
        - `yticks`
            - `Vector{Real}`, `Tuple{Vector{Real},Vector{Any}}`, optional
            - ticks to draw for the y-axis
            - also defines axis limits applied to `y`
                - i.e., bounds of the respective panel
                - `minimum(yticks[1])` corresponds to the start of the theta-slice
                - `maximum(yticks[1])` corresponds to the end of the theta-slice
            - if `Vector{Real}`
                - will use `yticks` as ticklabels as well
            - if `Tuple{Vector{Real},Vector{Any}}`
                - will use `yticks[2]` as ticklabels
            - the default is `nothing`
                - automatically generates 5 equidistantly spaced yticks
        - `thetaplotlims`
            - `Tuple{Real,Real}`, optional
            - range to be spanned by the individual theta-panels
            - sets the reference point for `thetalims` (i.e. puts the panels position into context)
                - `thetalims[1]` will be plotted at `thetaplotlims[1]`
                - `thetalims[2]` will be plotted at `thetaplotlims[2]`
            - in radians
            - overrides `LVPC.thetaplotlims`
            - the default is `nothing`
                - will fall back to `LVPC.thetaplotlims`
        - `panelsize`
            - `Real`, optional
            - size of individual panels in radians
            - defines the size of each panel
            - the entire figure can allocate `(thetaguidelims[2]-thetaguidelims[1])/panelsize` evenly distributed, nonoverlapping panels
            - overrides `LVPC.panelsize`
            - the default is `nothing`
                - falls back to `LVPC.panelsize`
        - `ylabel`
            - `String`, optional
            - label of the y-axis
            - overrides `LVPC.ylabel`
            - the default is `nothing`
                - falls back to `LVPC.ylabel`
        - `panelbounds`
            - `Bool`, optional
            - whether to show bounds of the individual panels when rendering
            - overrides `LVPC.panelbounds`
            - the default is `nothing`
                - falls back to `LVPC.panelbounds`
        - `ygrid`
            - `Bool`, optional
            - whether to show ticks and gridlines for y-values
            - overrides `LVPC.ygrid`
            - the default is `nothing`
                - falls back to `LVPC.ygrid`
        - `fontsizes`
            - `NamedTuple`, optional
            - font size to use in the panel
            - overrides `LVPC.fontsizes`
            - the default is `nothing`
                - will fall back to `LVPC.fontsizes`
        - `thetaticklabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the ticklabels of the theta-axis
            - used for styling
            - overrides `LVPC.thetagridkwargs`
            - the default is `nothing`
                - falls back to `LVPC.thetagridkwargs`      
        - `ygridkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `plot!()` when drawing the y-grid of each panel
            - used for styling
            - overrides `LVPC.ygridkwargs`
            - the default is `nothing`
                - falls back to `LVPC.ygridkwargs`         
        - `yticklabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the ticklabels of the y-axis
            - applied to all panels
            - used for styling
            - overrides `LVPC.yticklabelkwargs`
            - the default is `nothing`
                - falls back to `LVPC.yticklabelkwargs`
        - `ylabelkwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `text()` instance used for defining the axis label of the y-axis
            - applied to all panels
            - used for styling
            - overrides `LVPC.ylabelkwargs`
            - the default is `nothing`
                - falls back to `LVPC.ylabelkwargs`
        - `panelboundskwargs`
            - `NamedTuple`, optional
            - kwargs to pass to `plot!()` when drawing bounds of each panel
            - used for styling
            - overrides `LVPC.panelboundskwargs`
            - the default is `nothing`
                - falls back to `LVPC.panelboundskwargs`
        
    Infered Fields
    --------------
        - `ylims`
            - `Tuple{Real,Real}`
            - axis limits applied to `y`
                - `ylims[1]` corresponds to the lower bound of the panel, i.e., `theta_offset-panelsize/2`
                - `ylims[2]` corresponds to the upper bound of the panel, i.e., `theta_offset+panelsize/2`
                    - `theta_offset` is, hereby, `theta` converted to an azimuthal angle

    Methods
    -------
        - `get_xy2plot()`
        - `get_yticks()`
        - `get_thetaslice()`
        - `get_ygrid()`
        - `plot_LVisPPanel!()`

    Comments
    --------
"""
struct LVisPPanel
    LVPC::LVisPCanvas
    theta::Real
    x::Vector
    y::Vector

    thetaticklabel::String

    yticks::Tuple{Vector{Real},Vector{Any}}
    
    thetaplotlims::Tuple{Real,Real}
    panelsize::Real
     
    ylabel::String
    
    panelbounds::Bool
    ygrid::Bool
    
    fontsizes::NamedTuple

    thetaticklabelkwargs::NamedTuple
    
    ygridkwargs::NamedTuple
    yticklabelkwargs::NamedTuple
    ylabelkwargs::NamedTuple

    panelboundskwargs::NamedTuple

    #infered attributes
    ylims::Tuple{Real,Real}

    function LVisPPanel(LVPC::LVisPCanvas,
        theta::Real, x::Vector, y::Vector;
        thetaticklabel::Union{String,Nothing}=nothing,
        yticks::Union{Tuple{Vector{T},Vector{V}},Vector{U},Nothing}=nothing,
        thetaplotlims::Union{Tuple{Real,Real},Nothing}=nothing, panelsize::Union{Real,Nothing}=nothing,
        ylabel::Union{String,Nothing}=nothing,
        panelbounds::Union{Bool,Nothing}=nothing, ygrid::Union{Bool,Nothing}=nothing,
        fontsizes::Union{NamedTuple,Nothing}=nothing,
        thetaticklabelkwargs::Union{NamedTuple,Nothing}=nothing,
        ygridkwargs::Union{NamedTuple,Nothing}=nothing,
        yticklabelkwargs::Union{NamedTuple,Nothing}=nothing,
        ylabelkwargs::Union{NamedTuple,Nothing}=nothing,
        panelboundskwargs::Union{NamedTuple,Nothing}=nothing,
        ) where {T <: Real, U <: Real, V <: Any}

        #defaults (from parent `LVPC`)
        thetaticklabel      = isnothing(thetaticklabel)         ? "$(round(theta;digits=2))" : thetaticklabel
        thetaplotlims       = isnothing(thetaplotlims)          ? LVPC.thetaplotlims : thetaplotlims
        panelsize           = isnothing(panelsize)              ? LVPC.panelsize : panelsize
        panelbounds         = isnothing(panelbounds)            ? LVPC.panelbounds : panelbounds
        ygrid               = isnothing(ygrid)                  ? LVPC.ygrid : ygrid
        fontsizes           = isnothing(fontsizes)              ? LVPC.fontsizes : fontsizes
        ylabel              = isnothing(ylabel)                 ? LVPC.ylabel : ylabel
        thetaticklabelkwargs= isnothing(thetaticklabelkwargs)   ? LVPC.thetaticklabelkwargs : thetaticklabelkwargs
        ygridkwargs         = isnothing(ygridkwargs)            ? LVPC.ygridkwargs : ygridkwargs
        yticklabelkwargs    = isnothing(yticklabelkwargs)       ? LVPC.yticklabelkwargs : yticklabelkwargs
        ylabelkwargs        = isnothing(ylabelkwargs)           ? LVPC.ylabelkwargs : ylabelkwargs
        panelboundskwargs   = isnothing(panelboundskwargs)      ? LVPC.panelboundskwargs : panelboundskwargs

        #default parameters (within panel)
        yticks              = isnothing(yticks)                 ? collect(range(floor(nanminimum(y)), ceil(nanmaximum(y)), 5)) : yticks
        yticks              = isa(yticks, Vector)               ? (yticks, yticks) : yticks        
        fontsizes_default   = (thetaticklabel=12,thetalabel=15,xticklabel=10,xlabel=12,yticklabel=10,ylabel=12)
        if isnothing(fontsizes)
            fontsizes = fontsizes_default
        else
            fontsizes = merge(fontsizes_default, fontsizes)
        end

        begin #infered attributes
            ylims = extrema(yticks[1])
        end

        #remove datapoints that are out of bounds
        xlimbool = (LVPC.xlims[1] .<= x) .& (x .<= LVPC.xlims[2])
        ylimbool = (ylims[1]      .<= y) .& (y .<= ylims[2])
        x = x[xlimbool .& ylimbool]
        y = y[xlimbool .& ylimbool]
        
        #append ylims to ensure correct positioning of yticks when `ylims` are provided
        append!(x, [NaN, NaN])
        append!(y, ylims)

        #instantiate
        new(LVPC,
            theta, x, y,
            thetaticklabel,
            yticks,
            thetaplotlims, panelsize,
            ylabel,
            panelbounds, ygrid,
            fontsizes,
            thetaticklabelkwargs,
            ygridkwargs, yticklabelkwargs, ylabelkwargs,
            panelboundskwargs,
            ylims,
        )
    end
end

"""
    - method to project `LVPP.x` and `LVPP.y` onto the canvas
    - makes sure that
        - `LVPC.xlimdeadzone` is empty
        - `y2plot` scales radially to fill entire panel (constraint to ylims)
        - `y2plot` does not exceed the panel bounds

    Parameters
    ----------
        - `LVPP`
            - `LVisPPanel`
            - parent instance containing all relevant settings for the panel
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant global settings
                - for the canvas
                - for all panels
        - `yref`
            - `Real`
            - reference value to use as a zero-point for plotting the y-series
            - set to `nanminimum(LVPP.y)`

    Raises
    ------

    Returns
    -------
        - `x2plot`
            - `Vector`
            - `LVPP.x` after fitting to suit plotting canvas, panel, and specifications
                - i.e., `LVPC.xlimdeadzone`, `LVPG.xlimrange`
        - `y2plot`
            - `Vector`
            - `LVPP.y` after fitting to suit plotting canvas, panel, and specifications
                - i.e.,
                    - comply to `LVPP.ylims`
                    - do not plot out of bounds
                    - correctly project onto azimuthal coordinate

    Comments
    --------
"""
function get_xy2plot_(
    LVPP::LVisPPanel, LVPC::LVisPCanvas;
    yref::Real,
    )::Tuple{Vector,Vector}
    
    #adapting `x` to fit into plot (`xmin_ref` and `xmax_ref` to ensure correct scaling when series does not reach xlims)
    x2plot = minmaxscale(LVPP.x, LVPC.xlimdeadzone .* LVPC.xlimrange, LVPC.xlimrange; xmin_ref=LVPC.xlims[1], xmax_ref=LVPC.xlims[2])

    #adapting `y` to fit into plot
    yslice = x2plot .* tan(LVPP.panelsize)  #convert sector of angle `panelsize` to carthesian equivalent
    
    y2plot = (LVPP.y .- yref)                                   #remove offset
    y2plot = y2plot .* range(LVPC.xlimdeadzone,1,length(LVPP.y))    #rescale `y` with distance from origin

    y2plot = minmaxscale(y2plot, 0, nanmaximum(yslice);) .- yslice/2   #rescale `y` w.r.t. size of the sector

    return x2plot, y2plot
end
function get_xy2plot(
    LVPP::LVisPPanel, LVPC::LVisPCanvas;
    yref::Real,
    )::Tuple{Vector,Vector}
    
    #adapting `x` to fit into plot (`xmin_ref` and `xmax_ref` to ensure correct scaling when series does not reach xlims)
    x2plot = minmaxscale(LVPP.x, LVPC.xlimdeadzone .* LVPC.xlimrange, LVPC.xlimrange; xmin_ref=LVPC.xlims[1], xmax_ref=LVPC.xlims[2])

    #adapting `y` to fit into plot
    yslice = x2plot .* tan(LVPP.panelsize)  #convert sector of angle `panelsize` to carthesian equivalent
    
    scaler = minmaxscale(x2plot, LVPC.xlimdeadzone, 1; xmin_ref=LVPC.xlimdeadzone .* LVPC.xlimrange, xmax_ref=LVPC.xlimrange)                                 #map to range(xlimdeadzone,1) to ensure dependence on x-values
    y2plot = minmaxscale(LVPP.y, 0, 1;)  #map to range(0,1) for easy manipulation
    y2plot = y2plot .* scaler                                                           #rescale `y` with distance from origin
    
    ##map into panel of angular width yslice
    # y2plot = yslice .* y2plot .- yslice/2                                   #preserves relative dataseries-positions (i.e. for binning) #more projection effects
    y2plot = nanmaximum(yslice) .* y2plot .- yslice/2                       #less projection effects (preserves series-signatures) #leads to discrepancies between dataseries-positions     
    # y2plot = minmaxscale(y2plot, 0, nanmaximum(yslice)) .- yslice/2

    return x2plot, y2plot
end

"""
    - method to
        - obtain azimuthal angles to correctly position the y-yicks (i.e. y-grid)
        - generate correct labels for the y-ticks

    Parameters
    ----------
        - `LVPP`
            - `LVisPPanel`
            - parent instance containing all relevant settings for the panel
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant global settings
                - for the canvas
                - for all panels
        - `thmin`
            - `Real`
            - reference value for the position of first y-tick
            - lower bound of the theta-slice/panel
        - `thmax`
            - `Real`
            - reference value for the position of last y-tick
            - upper bound of the theta-slice/panel

    Raises
    ------

    Returns
    -------
        - `ytickpos`
            - `Vector`
            - positions of the y-ticks as azimuthal angle
            - will be converted to carthesian coordinates in main method for final plotting
        - `yticklabs`
            - `Vector`
            - ticklabels to be displayed for the y-ticks
            - rounded to 2 decimals for readability

    Comments
    --------
"""
function get_yticks(LVPP::LVisPPanel, LVPC::LVisPCanvas;
    thmin::Real, thmax::Real,
    )::Tuple{Vector,Vector}

    ytickpos = minmaxscale(LVPP.yticks[1], thmin, thmax)    #project into correct range (theta-slice)
    yticklabs = LVPP.yticks[2]
    return ytickpos, yticklabs
end

"""
    - method to obtain characteristic parameters of the theta-slice

    Parameters
    ----------
        - `LVPP`
            - `LVisPPanel`
            - parent instance containing all relevant settings for the panel
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant global settings
                - for the canvas
                - for all panels
        - `theta_offset`
            - `Real`
            - offset of the panel in theta-coordinate
            - ensures angular position of panel can be interpreted as a coordinate

    Raises
    ------

    Returns
    -------
        - `thslicebound_x`
            - `Matrix`
            - x-coordinates specifying lower and upper bounds of theta-slice
        - `thslicebound_y`
            - `Matrix`
            - y-coordinates specifying lower and upper bounds of theta-slice
        - `thslicemiddle_x`
            - `Vector`
            - x-coordinates specifying the center-line of the theta-slice
        - `thslicemiddle_y`
            - `Vector`
            - y-coordinates specifying the center-line of the theta-slice

    Comments
    --------
"""
function get_thetaslice(LVPP::LVisPPanel, LVPC::LVisPCanvas;
    theta_offset::Real
    )::Tuple{Matrix,Matrix,Vector,Vector}
    
    #upper and lower bound of theta slice
    thslice_lb = theta_offset - LVPP.panelsize/2
    thslice_ub = theta_offset + LVPP.panelsize/2

    #get bounds and middle in carthesian coordinates
    thslicebound_x = [LVPC.xlimdeadzone*LVPC.xlimrange, LVPC.xlimrange] .* cos.([thslice_lb, thslice_ub])'    #bounds of the theta slice
    thslicebound_y = [LVPC.xlimdeadzone*LVPC.xlimrange, LVPC.xlimrange] .* sin.([thslice_lb, thslice_ub])'    #bounds of the theta slice
    thslicemiddle_x = [LVPC.xlimdeadzone*LVPC.xlimrange, LVPC.xlimrange] .* cos(theta_offset) 
    thslicemiddle_y = [LVPC.xlimdeadzone*LVPC.xlimrange, LVPC.xlimrange] .* sin(theta_offset) 
    return thslicebound_x, thslicebound_y, thslicemiddle_x, thslicemiddle_y
end

"""
    - method to obtain carthesian coordinates specifying the y-grid
    - does so by converting `ytickpos` (azimuthal coordinate representation) into carthesian coordinates

    Parameters
    ----------
        - `LVPP`
            - `LVisPPanel`
            - parent instance containing all relevant settings for the panel
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant global settings
                - for the canvas
                - for all panels
        - `ytickpos`
            - `Vector`
            - positions of the y-ticks as azimuthal angle
            - will be converted to carthesian coordinates for final plotting

    Raises
    ------

    Returns
    -------
        - `ygrid_x`
            - `Matrix`
            - x-coordinates of the y-gridlines
        - `ygrid_y`
            - `Matrix`
            - y-coordinates of the y-gridlines

    Comments
    --------
"""
function get_ygrid(LVPP::LVisPPanel, LVPC::LVisPCanvas;
    ytickpos::Vector
    )::Tuple{Matrix, Matrix}
    ygrid_x = [LVPC.xlimdeadzone*LVPC.xlimrange, LVPC.xlimrange] .* cos.(ytickpos)' 
    ygrid_y = [LVPC.xlimdeadzone*LVPC.xlimrange, LVPC.xlimrange] .* sin.(ytickpos)' 
    return ygrid_x, ygrid_y
end

"""
    - method to add the panel to the plot
    - ideally a canvas (`render_LVisPCanvas()`) is already present in the plot

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - plot instance to plot into        
        - `LVPP`
            - `LVisPPanel`
            - parent instance containing all relevant settings for the panel
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant global settings
                - for the canvas
                - for all panels
        - `data_only`
            - `Bool`, optional
            - convenience flag to plot data only (no guides, labels, etc.)
            - useful for overplotting additional dataseries
            - the default is `false`
                - will plot everything (including guides, labels, etc.)
        - `plot_kwargs`
            - `Vararg`
            - kwargs to be passed to `plot!()`
            - only influences the plotted series

    Raises
    ------

    Returns
    -------

    Comments
    --------
"""
function plot_LVisPPanel!(plt::Plots.Plot,
    LVPP::LVisPPanel, LVPC::LVisPCanvas;
    data_only::Bool=false,
    plot_kwargs...,
    )

    #get global params
    yref = nanminimum(LVPP.y)

    #get series
    x2plot, y2plot = get_xy2plot(LVPP, LVPC; yref=yref)
    
    ##convert to polar coordinates for layouting
    rtheta2plot = carth2polar.(x2plot, y2plot)
    r2plot, theta2plot = getindex.(rtheta2plot,1), getindex.(rtheta2plot,2)

    ##custom offset to differentiate between different individual series
    theta_offset = minmaxscale([LVPP.theta], LVPP.thetaplotlims[1], LVPP.thetaplotlims[2]; xmin_ref=LVPC.thetalims[1], xmax_ref=LVPC.thetalims[2])[1]
    theta2plot = theta2plot .+ theta_offset

    ##convert back to carthesian coordinates for plotting
    xy_rotated = polar2carth.(r2plot, theta2plot)
    x_rotated, y_rotated = getindex.(xy_rotated,1), getindex.(xy_rotated,2)
    
    #get guides
    # ytickpos, yticklabs = get_yticks(LVPP; thmin=nanminimum(theta2plot), thmax=nanmaximum(theta2plot))
    ytickpos, yticklabs = get_yticks(LVPP, LVPC; thmin=theta_offset-LVPC.panelsize/2, thmax=theta_offset+LVPC.panelsize/2)
    thslicebound_x, thslicebound_y, thslicemiddle_x, thslicemiddle_y = get_thetaslice(LVPP, LVPC; theta_offset=theta_offset)
    ygrid_x, ygrid_y = get_ygrid(LVPP, LVPC; ytickpos=ytickpos)
    
    #plotting
    ##series
    plot!(plt, x_rotated, y_rotated; plot_kwargs...)
    
    #plain data plotting
    if data_only
        return
    else    #add guides
        ##get some corrections
        thetaticklabelkwargs = LVPP.thetaticklabelkwargs
        if ~in(:rotation, keys(thetaticklabelkwargs))
            thetaticklabelkwargs = merge(thetaticklabelkwargs, (rotation=correct_labelrotation(theta_offset * 180/pi) - 90,))
        end
        ylabelkwargs = LVPP.ylabelkwargs
        if ~in(:rotation, keys(ylabelkwargs))
            ylabelkwargs = merge(ylabelkwargs, (rotation=correct_labelrotation(theta_offset * 180/pi) - 90,))
        end
        yticklabelkwargs = LVPP.yticklabelkwargs
        if ~in(:rotation, keys(yticklabelkwargs))
            yticklabelkwargs = merge(yticklabelkwargs, (rotation=correct_labelrotation(theta_offset * 180/pi) - 90,))
        end    
    
    
        
        ##guides
        if LVPP.ygrid #grid for y-values
            plot!(plt, ygrid_x, ygrid_y; label="", LVPP.ygridkwargs...)
            annotate!(plt, 1.1.*ygrid_x[2,:], 1.1.*ygrid_y[2,:], map(ytp -> text("$(ytp)", LVPP.fontsizes[:yticklabel]; yticklabelkwargs...), yticklabs); label="")                                                     #y-ticks
        end 
        if LVPP.panelbounds #theta sector bounds
            plot!(plt, thslicebound_x[:,:], thslicebound_y[:,:]; label="", LVPP.panelboundskwargs...)
        end 
    
        ##labels
        annotate!(plt, 1.35*thslicemiddle_x[2], 1.25*thslicemiddle_y[2], text(LVPP.thetaticklabel, LVPP.fontsizes[:thetaticklabel]; thetaticklabelkwargs...); label="")                                             #theta sector
        annotate!(plt, 1.2*thslicemiddle_x[2], 1.18*thslicemiddle_y[2], text(LVPP.ylabel, LVPP.fontsizes[:ylabel]; ylabelkwargs...); label="")  #y-label
    end

    
    begin#temporary: mark nanmaximum
        
        # #find nanmaximum
        # maxidx = nanargmax(LVPP.y[1:end-2])
        # # println(x_rotated[maxidx])
        # # println(LVPP.y[maxidx])
        # maxx = x_rotated[maxidx]
        # maxy = y_rotated[maxidx]
        # #plot nanmaximum
        # scatter!(plt, [maxx], [maxy]; marker=:circle, markersize=5, color=:red, label="")
    end
end

"""
    - extensions to `Plots.plot!()` and `Plots.plot()`

    Parameters
    ----------
        - `plt`
            - `Plots.Plot`
            - panel to plot into    
        - `LVPC`
            - `LVisPCanvas`
            - parent instance containing all relevant global settings
                - for the canvas
                - for all panels
        - `theta`
            - `Vector{Real}`
            - values to be plotted around around the circle
        - `x`
            - `Vector{Vector{Real}}`
            - x-values of the dataseries
            - has to have same size as `theta`
            - each entry is one series to be plotted
                - entries can have different lengths
        - `y`
            - `Vector{Vector{Real}}`
            - y-values of the dataseries
            - has to have same size as `theta`
            - each entry is one series to be plotted
                - entries can have different lengths
        - `thetaticklabels`
            - `Vector{String}`, optional
            - same length as `theta`
            - ticklabels to use for the different theta sectors
            - the default is `nothing`
                - will use `theta` as the labels
        - `yticks`
            - `Vector{Real}`, `Tuple{Vector{Real},Vector{Any}}`, optional
            - ticks to draw for the y-axis
            - applied to all plotted theta-slices
            - also defines axis limits applied to `y`
                - i.e., bounds of the respective panel
                - `minimum(yticks[1])` corresponds to the start of the theta-slice
                - `maximum(yticks[1])` corresponds to the end of the theta-slice
            - if `Vector{Real}`
                - will use `yticks` as ticklabels as well
            - if `Tuple{Vector{Real},Vector{Any}}`
                - will use `yticks[2]` as ticklabels
            - the default is `nothing`
                - automatically generates 5 equidistantly spaced yticks for each panel
                - no common y-limits across panels
        - `data_only`
            - `Bool`, optional
            - convenience flag to plot data only (no guides, labels, etc.)
            - useful for overplotting additional dataseries
            - the default is `false`
                - will plot everything (including guides, labels, etc.)
        - `plot_kwargs`
            - `Vector{Dict{Symbol,Any}`, optional
            - kwargs to pass to `plot()`
            - has to have same size as `theta`
                - one dict for each series to be plotted
            - the default is `nothing`
                - will be set to `fill(Dict(), (length(theta)))`
                - no additional kwargs to pass

    Raises
    ------

    Returns
    -------
    - `plt`
        - `Plots.Plot`
        - created panel

    Comments
    --------
        
"""
function Plots.plot!(plt::Plots.Plot,
    LVPC::LVisPCanvas,
    theta::Vector{<: Real}, x::Vector{Vector{T}}, y::Vector{Vector{T}};
    thetaticklabels::Union{Vector{String},Nothing}=nothing,
    yticks::Union{Tuple{Vector{<: Real},Vector{V}},Vector{<: Real},Nothing}=nothing,
    data_only::Bool=false,
    plot_kwargs::Union{Vector{Dict{Symbol,Any}},Nothing}=nothing,
    ) where {T <: Real, U <: Real, V <: Any}

    #default parameters
    thetaticklabels = isnothing(thetaticklabels) ? fill(nothing, (length(theta))) : thetaticklabels
    plot_kwargs = isnothing(plot_kwargs) ? fill(Dict(), (length(theta))) : plot_kwargs

    #generate panels
    LVPP = []
    for (i, (thetai, xi, yi)) in enumerate(zip(theta, x, y))
        push!(LVPP,
            LVisPPanel(
                LVPC,
                thetai, xi, yi;
                thetaticklabel=thetaticklabels[i],
                yticks=yticks,
            )
        )
    end

    #render
    if ~data_only   #ignore guides if specified
        plot_LVisPCanvas!(plt, LVPC)
    end
    for i in eachindex(LVPP)
        plot_LVisPPanel!(plt, LVPP[i], LVPC; data_only=data_only, plot_kwargs[i]...)
    end
end
function Plots.plot(
    LVPC::LVisPCanvas,
    theta::Vector{<: Real}, x::Vector{Vector{T}}, y::Vector{Vector{T}};
    thetaticklabels::Union{Vector{String},Nothing}=nothing,
    yticks::Union{Tuple{Vector{<: Real},Vector{V}},Vector{<: Real},Nothing}=nothing,
    data_only::Bool=false,
    plot_kwargs::Union{Vector{Dict{Symbol,Any}},Nothing}=nothing,
    ) where {T <: Real, U <: Real, V <: Any}
    
    plt = plot(;
        size=(1200,1200),
        leftmargin=20Plots.mm, rightmargin=0Plots.mm,
        topmargin=0Plots.mm, bottommargin=15Plots.mm,
    )
    plot!(plt, LVPC, theta, x, y;
        thetaticklabels=thetaticklabels, yticks=yticks,
        data_only=data_only,
        plot_kwargs=plot_kwargs
    )
    return plt
end

end #module