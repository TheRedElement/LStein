
#%%imports
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
from typing import Any, List, Tuple

import utils as lvisu

#%%classes
class LVisPCanvas:

    def __init__(self,
        thetaticks:Tuple[List[float],List[Any]], xticks:Tuple[List[float],List[Any]], yticks:Tuple[List[float],List[Any]],
        thetaguidelims:Tuple[float,float]=None, thetaplotlims:Tuple[float,float]=None, xlimdeadzone:float=0.3, panelsize:float=np.pi/8,
        thetalabel:str=None, xlabel:str=None, ylabel:str=None,
        thetaarrowlength:float=np.pi/4,
        panelbounds:bool=False, ygrid:bool=True,
        # fontsizes::Union{NamedTuple,Nothing}=nothing,
        thetatickkwargs:dict=None,
        thetaticklabelkwargs:dict=None,
        thetalabelkwargs:dict=None,
        xtickkwargs:dict=None,
        xticklabelkwargs:dict=None,
        xlabelkwargs:dict=None,
        # ygridkwargs::NamedTuple=(linecolor=:black, linealpha=0.3, linestyle=:solid,),
        # yticklabelkwargs::NamedTuple=NamedTuple(),
        # ylabelkwargs::NamedTuple=NamedTuple(),
        # panelboundskwargs::NamedTuple=(linecolor=:black, linealpha=0.4, linestyle=:solid,),
        ):
        
        self.thetaticks     = (thetaticks, thetaticks) if isinstance(thetaticks, (list, np.ndarray)) else thetaticks
        self.xticks         = (xticks, xticks) if isinstance(xticks, (list, np.ndarray)) else xticks
        self.yticks         = (yticks, yticks) if isinstance(yticks, (list, np.ndarray)) else yticks
        
        self.thetaguidelims = (0,2,np.pi) if thetaguidelims is None else thetaguidelims
        self.thetaplotlims  = thetaguidelims if thetaplotlims is None else thetaplotlims
        self.xlimdeadzone   = xlimdeadzone
        self.panelsize      = panelsize

        self.thetalabel     = "" if thetalabel is None else thetalabel
        self.xlabel         = "" if xlabel is None else xlabel
        self.ylabel         = "" if ylabel is None else ylabel

        self.thetaarrowlength = thetaarrowlength

        self.panelbounds    = panelbounds
        self.ygrid          = ygrid

        self.thetatickkwargs       = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if thetatickkwargs is None else thetatickkwargs
        self.thetaticklabelkwargs   = dict(ha="center", va="center", pad=0.2) if thetaticklabelkwargs is None else thetaticklabelkwargs
        self.thetalabelkwargs       = dict(textcoords="offset fontsize", xytext=(-2,-2), rotation="auto") if thetalabelkwargs is None else thetalabelkwargs
        
        self.xtickkwargs            = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if xtickkwargs is None else xtickkwargs
        self.xticklabelkwargs       = dict(textcoords="offset fontsize", xytext=(-1,-1)) if xticklabelkwargs is None else xticklabelkwargs
        self.xlabelkwargs           = dict(textcoords="offset fontsize", xytext=(-2,-2)) if xlabelkwargs is None else xlabelkwargs

        #infered attributes
        self.xlimrange = np.max(self.xticks[0]) - np.min(self.xticks[0])

        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + ", ".join([f"{attr}={val}" for attr, val in self.__dict__.items()]) + ")"

    def add_xaxis(self,
        ax:plt.Axes
        ):

        #xticks
        th_circ = np.linspace(self.thetaguidelims[0], self.thetaguidelims[1], 100)
        r_circ = self.xticks[0] - np.min(self.xticks[0])
        r_circ = lvisu.minmaxscale(r_circ, np.max(r_circ) * self.xlimdeadzone, np.max(r_circ))
        circles_x = r_circ.reshape(-1,1) @ np.cos(th_circ).reshape(1,-1)
        circles_y = r_circ.reshape(-1,1) @ np.sin(th_circ).reshape(1,-1)

        circles_x = np.concat([circles_x[0,0]+np.zeros((len(r_circ),1)), circles_x, circles_x[0,-1]+np.zeros((len(r_circ),1))], axis=1) #add start and endpoint of innermost circle (to ensure circles connect at innermost circle)
        circles_y = np.concat([circles_y[0,0]+np.zeros((len(r_circ),1)), circles_y, circles_y[0,-1]+np.zeros((len(r_circ),1))], axis=1) #add start and endpoint of innermost circle (to ensure circles connect at innermost circle)
        circles_x[0:-1,[0,-1]] = np.nan #set to NaN to force breaks
        circles_y[0:-1,[0,-1]] = np.nan #set to NaN to force breaks

        #xticklabels
        xtickpos_x  = circles_x[:,1]
        xtickpos_y  = circles_y[:,1]
        xticklabs   = self.xticks[1]

        #xlabel
        xlabpos_x = xtickpos_x[-1]
        xlabpos_y = xtickpos_y[-1]

        ax.plot(circles_x.T, circles_y.T, **self.xtickkwargs)
        for i in range(len(xticklabs)):
            ax.annotate(xticklabs[i], xy=(xtickpos_x[i], xtickpos_y[i]), **self.xticklabelkwargs)
        ax.annotate(self.xlabel, xy=(xlabpos_x, xlabpos_y), **self.xlabelkwargs)

        return
    
    def add_thetaaxis(self,
        ax:plt.Axes,               
        ):

        #ticks
        th_pad = 1-self.thetaticklabelkwargs.pop("pad")     #get padding (scales position)
        thetatickpos_ri = th_pad * self.xlimdeadzone*self.xlimrange     #inner edge of theta ticks
        thetatickpos_ro = self.xlimdeadzone*self.xlimrange              #outer edge of theta ticks
        thetatickpos_th = lvisu.minmaxscale(self.thetaticks[0], self.thetaplotlims[0], self.thetaplotlims[1])
        thetaticklabelpos_x, thetaticklabelpos_y    = lvisu.polar2carth(thetatickpos_ri, thetatickpos_th)
        thetatickpos_xi, thetatickpos_yi            = lvisu.polar2carth(thetatickpos_ro*(th_pad+0.15), thetatickpos_th)
        thetatickpos_xo, thetatickpos_yo            = lvisu.polar2carth(thetatickpos_ro, thetatickpos_th)

        #indicator
        th_arrow    = np.linspace(self.thetaguidelims[0], self.thetaguidelims[0]+self.thetaarrowlength, 101)
        x_arrow, y_arrow = lvisu.polar2carth(1.4*self.xlimrange, th_arrow)

        #label
        th_label_x, th_label_y = lvisu.polar2carth(1.45 * self.xlimrange, np.mean(th_arrow))

        ##get correct rotation
        th_rot = lvisu.correct_labelrotation(np.mean(th_arrow)/np.pi*180)-90 if self.thetalabelkwargs["rotation"] == "auto" else self.thetalabelkwargs["rotation"]
        thetalabelkwargs = self.thetalabelkwargs.copy()
        thetalabelkwargs["rotation"] = th_rot

        #plotting
        ax.plot(np.array([thetatickpos_xi, thetatickpos_xo]), np.array([thetatickpos_yi, thetatickpos_yo]), **self.thetatickkwargs)
        for i in range(len(self.thetaticks[0])):    #ticklabels
            ax.annotate(f"{self.thetaticks[1][i]}", xy=(thetaticklabelpos_x[i], thetaticklabelpos_y[i]), **self.thetaticklabelkwargs)
        line, = ax.plot(x_arrow[:-1], y_arrow[:-1], **self.thetatickkwargs)
        ax.annotate("",
            xy=(x_arrow[-1],y_arrow[-1]),
            xytext=(x_arrow[-2],y_arrow[-2]),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=line.get_linewidth(),
                color=line.get_color(),
                fill=True,
            )
        )        
        ax.annotate(self.thetalabel, xy=(th_label_x,th_label_y), **thetalabelkwargs)
        return

    def plot_LVisPCanvas(self,
        ax:plt.Axes,                 
        ):

        #disable some default settings
        ax.set_aspect("equal")
        ax.set_axis_off()

        self.add_xaxis(ax)
        self.add_thetaaxis(ax)

        return

#%%
panelsize = np.pi/5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()

LVPC = LVisPCanvas(
    [-3,0,7], [-20,0,100], [0, 1, 10],
    thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
    xlimdeadzone=0.3,
    panelsize=panelsize,
    thetalabel=r"$\theta$-label", xlabel=r"$x$-label", ylabel=r"$y$-label",
    thetaarrowlength=np.pi/2,
    thetatickkwargs=None, thetaticklabelkwargs=None, thetalabelkwargs=None,
    xtickkwargs=None, xticklabelkwargs=None, xlabelkwargs=None,
)
LVPC.plot_LVisPCanvas(ax)
# LVPC.add_xaxis(ax)
# LVPC.add_thetaaxis(ax)

fig.tight_layout()

    