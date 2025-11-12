#%%imports
import logging
import numpy as np
from ..utils import polar2carth

logger = logging.getLogger(__name__)

#%%definitions
#canvas
def add_xaxis(LSC, ax):
    #get quantities
    circles_x, circles_y, \
    xtickpos_x, xtickpos_y, xticklabs, \
    xlabpos_x, xlabpos_y, = LSC.compute_xaxis()

    #plotting
    ax.plot(circles_x.T, circles_y.T, **LSC.xtickkwargs)
    for i in range(len(xticklabs)):
        ax.annotate(xticklabs[i], xy=(xtickpos_x[i], xtickpos_y[i]), annotation_clip=False, **LSC.xticklabelkwargs)
    ax.annotate(LSC.xlabel, xy=(xlabpos_x, xlabpos_y), annotation_clip=False, **LSC.xlabelkwargs)

    return

def add_thetaaxis(LSC, ax):
    thetaticklabelpos_x, thetaticklabelpos_y, \
    thetatickpos_xi, thetatickpos_yi, thetatickpos_xo, thetatickpos_yo, \
    th_label_x, th_label_y, \
    x_arrow, y_arrow, = LSC.compute_thetaaxis()

    ax.plot(np.array([thetatickpos_xi, thetatickpos_xo]), np.array([thetatickpos_yi, thetatickpos_yo]), **LSC.thetatickkwargs)
    for i in range(len(LSC.thetaticks[0])):    #ticklabels
        ax.annotate(f"{LSC.thetaticks[1][i]}", xy=(thetaticklabelpos_x[i], thetaticklabelpos_y[i]), annotation_clip=False, **LSC.thetaticklabelkwargs)
    line, = ax.plot(x_arrow[:-1], y_arrow[:-1], **LSC.thetatickkwargs)
    ax.annotate("",
        xy=(x_arrow[-1],y_arrow[-1]),
        xytext=(x_arrow[-2],y_arrow[-2]),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=line.get_linewidth(),
            color=line.get_color(),
            fill=True,
        ),
        annotation_clip=False, 
    )        
    ax.annotate(LSC.thetalabel, xy=(th_label_x,th_label_y), annotation_clip=False, **LSC.thetalabelkwargs)
    return

def add_ylabel(LSC, ax):
    ylabpos_x, ylabpos_y = LSC.compute_ylabel()
    ax.annotate(LSC.ylabel, xy=(ylabpos_x, ylabpos_y), annotation_clip=False, **LSC.ylabelkwargs)
    return

#panels
def add_yaxis(LSP, ax):

    #get panel boundaries
    theta_offset, theta_lb, theta_ub = LSP.get_thetabounds()
    r_lb, r_ub = LSP.get_rbounds()
    r_bounds = np.array([r_lb, r_ub])

    #get yticks
    ytickpos_th, yticklabs = LSP.get_yticks(theta_lb, theta_ub)

    #convert to carthesian for plotting
    x_lb, y_lb  = polar2carth(r_bounds, theta_lb)
    x_ub, y_ub  = polar2carth(r_bounds, theta_ub)
    x_bounds = np.array([x_lb,x_ub])
    y_bounds = np.array([y_lb,y_ub])

    pad = LSP.yticklabelkwargs.pop("pad")   #padding for yticklabels
    r_, th_ = np.meshgrid(r_bounds, ytickpos_th)
    ytickpos_x, ytickpos_y              = polar2carth(r_, th_)
    yticklabelpos_x, yticklabelpos_y    = polar2carth((1+pad)*r_ub, ytickpos_th)

    # ytickpos_x, ytickpos_y = ytickpos_x[::-1], ytickpos_y[::-1]
    # yticklabelpos_x, yticklabelpos_y = yticklabelpos_x[::-1], yticklabelpos_y[::-1]

    if LSP.show_yticks:
        ax.plot(ytickpos_x.T, ytickpos_y.T, **LSP.ytickkwargs)
        for i in range(len(ytickpos_th)):
            ax.annotate(yticklabs[i], xy=(yticklabelpos_x[i],yticklabelpos_y[i]), annotation_clip=False, **LSP.yticklabelkwargs)
    if LSP.show_panelbounds: ax.plot(x_bounds.T, y_bounds.T, **LSP.panelboundskwargs)

    return

#plotting
def scatter(ax, x, y, *args, **kwargs):
    ax.scatter(x, y, **kwargs)
    return

def plot(ax, x, y, *args, **kwargs):
    ax.plot(x, y, **kwargs)
    return

#combined
def show(LSC, ax):

    #disable some default settings
    ax.set_aspect("equal")
    ax.set_axis_off()

    #add canvas elements
    add_xaxis(LSC, ax)
    add_thetaaxis(LSC, ax)
    add_ylabel(LSC, ax)

    #update switch denoting that panel has been drawn
    LSC.canvas_drawn = True

    for LSP in LSC.Panels:
        if not LSP.panel_drawn:
            add_yaxis(LSP, ax)
            LSP.panel_drawn = True

        #plot all dataseries
        for ds in LSP.dataseries:

            if ds["seriestype"] == "scatter": func = scatter
            elif ds["seriestype"] == "plot":  func = plot
            else:
                logger.warning(f"seriestype fof {ds['seriestype']} is not supported. try one of `['scatter','plot']`")
                continue

            func(ax, ds["x"], ds["y"], **ds["kwargs"])
    return ax
