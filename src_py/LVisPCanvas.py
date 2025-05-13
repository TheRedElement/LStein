
#%%imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Any, Dict, List, Literal, Tuple, Union

import utils as lvisu
from LVisPPanel import LVisPPanel

#%%classes
class LVisPCanvas:

    def __init__(self,
        ax:plt.Axes,
        thetaticks:Tuple[List[float],List[Any]], xticks:Tuple[List[float],List[Any]], yticks:Tuple[List[float],List[Any]],
        thetaguidelims:Tuple[float,float]=None, thetaplotlims:Tuple[float,float]=None, xlimdeadzone:float=0.3, 
        thetalabel:str=None, xlabel:str=None, ylabel:str=None,
        thetaarrowlength:float=np.pi/4,
        thetatickkwargs:dict=None,
        thetaticklabelkwargs:dict=None,
        thetalabelkwargs:dict=None,
        xtickkwargs:dict=None,
        xticklabelkwargs:dict=None,
        xlabelkwargs:dict=None,
        ylabelkwargs:dict=None,
        ):
        
        self.ax             = ax

        self.thetaticks     = (thetaticks, np.round(thetaticks,0).astype(int)) if isinstance(thetaticks, (list, np.ndarray)) else thetaticks
        self.xticks         = (xticks, np.round(xticks,0).astype(int)) if isinstance(xticks, (list, np.ndarray)) else xticks
        self.yticks         = (yticks, np.round(yticks,0).astype(int)) if isinstance(yticks, (list, np.ndarray)) else yticks
        
        self.thetaguidelims = (0,2,np.pi) if thetaguidelims is None else thetaguidelims
        self.thetaplotlims  = thetaguidelims if thetaplotlims is None else thetaplotlims
        self.xlimdeadzone   = xlimdeadzone
        self.panelsize      = panelsize

        self.thetalabel     = "" if thetalabel is None else thetalabel
        self.xlabel         = "" if xlabel is None else xlabel
        self.ylabel         = "" if ylabel is None else ylabel

        self.thetaarrowlength = thetaarrowlength


        self.thetatickkwargs        = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if thetatickkwargs is None else thetatickkwargs
        self.thetaticklabelkwargs   = dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center", pad=0.2) if thetaticklabelkwargs is None else thetaticklabelkwargs
        self.thetalabelkwargs       = dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center") if thetalabelkwargs is None else thetalabelkwargs
        
        self.xtickkwargs            = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if xtickkwargs is None else xtickkwargs
        self.xticklabelkwargs       = dict(c=plt.rcParams["axes.labelcolor"], textcoords="offset fontsize", xytext=(-1,-1)) if xticklabelkwargs is None else xticklabelkwargs
        self.xlabelkwargs           = dict(c=plt.rcParams["axes.labelcolor"], textcoords="offset fontsize", xytext=(-2,-2)) if xlabelkwargs is None else xlabelkwargs
        
        self.ylabelkwargs           = dict(c=plt.rcParams["axes.labelcolor"]) if ylabelkwargs is None else ylabelkwargs

        #infered attributes
        self.xlims = (np.min(self.xticks[0]), np.max(self.xticks[1]))
        self.xlimrange = np.max(self.xticks[0]) - np.min(self.xticks[0])
        self.Panels = []
        self.canvas_drawn = False

        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + ", ".join([f"{attr}={val}" for attr, val in self.__dict__.items()]) + ")"

    #canvas methods
    def add_xaxis(self,
        ax:plt.Axes=None,                 
        ):

        #default parameters
        if ax is None: ax = self.ax

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
            ax.annotate(xticklabs[i], xy=(xtickpos_x[i], xtickpos_y[i]), annotation_clip=False, **self.xticklabelkwargs)
        ax.annotate(self.xlabel, xy=(xlabpos_x, xlabpos_y), annotation_clip=False, **self.xlabelkwargs)

        return
    
    def add_thetaaxis(self,
        ax:plt.Axes=None,                 
        ):

        #default parameters
        if ax is None: ax = self.ax

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
        x_arrow, y_arrow = lvisu.polar2carth(1.0*thetatickpos_ro, th_arrow)

        #label
        # th_label_x, th_label_y = lvisu.polar2carth(1.45 * self.xlimrange, np.mean(th_arrow))
        th_label_x, th_label_y = (0,0)

        ##get correct rotation
        # th_rot = lvisu.correct_labelrotation(np.mean(th_arrow)/np.pi*180)-90 if self.thetalabelkwargs["rotation"] == "auto" else self.thetalabelkwargs["rotation"]
        # thetalabelkwargs = self.thetalabelkwargs.copy()
        # thetalabelkwargs["rotation"] = th_rot

        #plotting
        ax.plot(np.array([thetatickpos_xi, thetatickpos_xo]), np.array([thetatickpos_yi, thetatickpos_yo]), **self.thetatickkwargs)
        for i in range(len(self.thetaticks[0])):    #ticklabels
            ax.annotate(f"{self.thetaticks[1][i]}", xy=(thetaticklabelpos_x[i], thetaticklabelpos_y[i]), annotation_clip=False, **self.thetaticklabelkwargs)
        line, = ax.plot(x_arrow[:-1], y_arrow[:-1], **self.thetatickkwargs)
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
        ax.annotate(self.thetalabel, xy=(th_label_x,th_label_y), annotation_clip=False, **self.thetalabelkwargs)
        return

    def draw_LVisPCanvas(self,
        ax:plt.Axes=None,                 
        ):

        #default parameters
        if ax is None: ax = self.ax

        #disable some default settings
        ax.set_aspect("equal")
        ax.set_axis_off()

        self.add_xaxis(ax)
        self.add_thetaaxis(ax)

        #update switch denoting that panel has been drawn
        self.canvas_drawn = True

        return

    #panel methods
    def add_panel(self,
        theta:float,
        yticks:Tuple[List[float],List[Any]]=None,
        panelsize:float=np.pi/8,
        show_panelbounds:bool=False, show_yticks:bool=True,
        y_projection_method:Literal["y","theta"]="y",
        ytickkwargs:dict=None,
        yticklabelkwargs:dict=None,
        panelboundskwargs:dict=None,
        ):

        #default parameters
        if isinstance(yticks, (list, np.ndarray)):
            yticks = (yticks, yticks)
        elif yticks is None and self.yticks is not None:
            yticks = self.yticks
        else:
            yticks = yticks


        LVPP = LVisPPanel(self,
            theta=theta,
            yticks=yticks,
            panelsize=panelsize,
            show_panelbounds=show_panelbounds, show_yticks=show_yticks,
            y_projection_method=y_projection_method,
            ytickkwargs=ytickkwargs,
            yticklabelkwargs=yticklabelkwargs,
            panelboundskwargs=panelboundskwargs,
        )

        self.Panels.append(LVPP)

        return LVPP
    
    #get methods
    def get_thetas(self) -> List[float]:
        thetas = [P.theta for P in self.Panels]

        return thetas
    
    def get_panel(self,
        theta:float,
        ) -> LVisPPanel:

        panel = [P for P in self.Panels if P.theta == theta][0]

        return panel

    #convenience methods
    def plot(self,
        theta:np.ndarray, X:List[np.ndarray], Y:List[np.ndarray],
        panel_kwargs:List[Dict]=None,
        plot_kwargs:List[Dict]=None,
        ):

        #default parameters
        panel_kwargs = [dict() for _ in theta.__iter__()] if panel_kwargs is None else panel_kwargs
        plot_kwargs  = [dict() for _ in theta.__iter__()] if plot_kwargs is None else plot_kwargs

        #get existing panels
        thetas = self.get_thetas()

        #generate colors
        colors = lvisu.get_colors(theta)
        for i in range(len(plot_kwargs)):
            if "c" not in plot_kwargs[i].keys(): plot_kwargs[i]["c"] = mcolors.to_hex(colors[i])
        
        for i in range(len(theta)):
            #avoid drawing the panel twice
            if theta[i] not in thetas:
                LVPP = self.add_panel(
                    theta=theta[i],
                    **panel_kwargs[i]
                )
            else:
                LVPP = self.get_panel(theta[i])
            
            #draw the series
            LVPP.plot(X[i], Y[i], **plot_kwargs[i])
            
        return
    
    def scatter(self,
        theta:np.ndarray, X:List[np.ndarray], Y:List[np.ndarray],
        panel_kwargs:List[Dict]=None,
        scatter_kwargs:List[Dict]=None,
        ):
    
        #default parameters
        panel_kwargs    = [dict() for _ in theta.__iter__()] if panel_kwargs is None else panel_kwargs
        scatter_kwargs  = [dict() for _ in theta.__iter__()] if scatter_kwargs is None else scatter_kwargs

        #get existing panels
        thetas = self.get_thetas()

        #generate colors
        colors = lvisu.get_colors(theta)
        for i in range(len(scatter_kwargs)):
            if "c" not in scatter_kwargs[i].keys(): scatter_kwargs[i]["c"] = mcolors.to_hex(colors[i])
        
        for i in range(len(theta)):
            #avoid drawing the panel twice
            if theta[i] not in thetas:
                LVPP = self.add_panel(
                    theta=theta[i],
                    **panel_kwargs[i]
                )
            else:
                LVPP = self.get_panel(theta[i])
            
            #draw the series
            LVPP.scatter(X[i], Y[i], **scatter_kwargs[i])

        return    

#%%pseudo data
def gaussian_pdf(x, mu, sigma):
    """
        - function defining a gaussian normal distribution
    """    
    y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))
    return y
def lc_sim(
    t:np.ndarray,
    t_peak:float, f_peak:float,
    stretch0:float, stretch1:float, stretch2:float,
    lbda:float=1.0,
    noiselevel:float=0.0,
    ) -> np.ndarray:
    """
        - function to define a very simplistic phenomenological LC simulation
    """
    f = (gaussian_pdf(t, t_peak - stretch0/2, stretch1) + gaussian_pdf(t, t_peak + stretch0/2, stretch2))
    f = f_peak * f / np.max(f) + noiselevel * np.random.randn(*t.shape)
    f *= lbda
    return f
def sin_sim(
    t:np.array,
    f_peak:float,
    p:float, offset:float=0.,
    noiselevel:float=0.0
    ) -> float:
    """
        - function to evaluate a sin with period `p` and `offset`
    """
    f = f_peak * np.sin(t * 2*np.pi/p + offset) + noiselevel * np.random.randn(*t.shape)
    return f
def simulate(
    nobjects:int=6,
    opt:Literal["lc","sin"]="lc"
    ):
    res = 500
    x = np.sort(np.random.choice(np.linspace(-50,100,res), size=(nobjects,res)), axis=1)
    theta_options = np.arange(0.4, 4, 0.5)
    theta = np.random.choice(theta_options, size=nobjects, replace=False)
    
    if opt == "lc":
        t_peak = np.linspace(0,40,nobjects) * 0
        y           = np.array([*map(lambda i: lc_sim(x[i], t_peak=t_peak[i], f_peak=20, lbda=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=1.0), range(nobjects))])
        y_nonoise   = np.array([*map(lambda i: lc_sim(x[i], t_peak=t_peak[i], f_peak=20, lbda=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=0.0), range(nobjects))])
    else:
        theta *= 10
        y           = np.array([*map(lambda i: sin_sim(x[i], f_peak=1, p=theta[i], offset=0.0, noiselevel=0.1), range(nobjects))])
        y_nonoise   = np.array([*map(lambda i: sin_sim(x[i], f_peak=1, p=theta[i], offset=0.0, noiselevel=0.0), range(nobjects))])

    return theta, x, y, y_nonoise

theta, X, Y, Y_nonoise = simulate(4, opt="lc")
theta, X, Y, Y_nonoise = simulate(5, opt="sin")

fig = plt.figure()
for i in range(len(theta)):
    plt.scatter(X[i], Y[i])
    plt.plot(X[i], Y_nonoise[i])

#%%
thetaticks = np.linspace(np.floor(np.min(theta)), np.ceil(np.max(theta)), 4)
yticks = np.round(np.linspace(np.floor(np.min(np.concat(Y))), np.ceil(np.max(np.concat(Y))), 4), decimals=0)
# yticks = np.sort(np.append(yticks, [-10, 80]))
panelsize = np.pi/8

#%%standard usage
fig = plt.figure(figsize=(5,9))
ax = fig.add_subplot(111)
LVPC = LVisPCanvas(ax,
    thetaticks, [-20,0,100,120], yticks,
    thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
    xlimdeadzone=0.3,
    thetalabel=r"$\theta$-label", xlabel=r"$x$-label", ylabel=r"$y$-label",
    thetaarrowlength=np.pi/2,
    thetatickkwargs=None, thetaticklabelkwargs=None, thetalabelkwargs=None,
    xtickkwargs=None, xticklabelkwargs=None, xlabelkwargs=None,
)
colors = lvisu.get_colors(theta)

for i in range(len(X)):
    LVPP = LVPC.add_panel(
        theta=theta[i],
        # yticks=None,
        yticks=yticks,
        # yticks=(yticks, ["A", "B", "C", "D"]), 
        panelsize=panelsize,
        show_panelbounds=True, show_yticks=True,
        y_projection_method="y",
        # y_projection_method="theta",
        ytickkwargs=None, yticklabelkwargs=None,
        panelboundskwargs=None,
    )

    LVPP.scatter(X[i], Y[i], c=Y[i], s=5,  alpha=np.linspace(0, 1, Y[i].shape[0]))
    LVPP.plot(X[i], Y_nonoise[i], c="w", lw=3)
    LVPP.plot(X[i], Y_nonoise[i], color=colors[i])

plt.show()

#%%convenience usage
fig = plt.figure(figsize=(5,9))
ax = fig.add_subplot(111)
LVPC = LVisPCanvas(ax,
    thetaticks, [-20,0,100,120], yticks,
    thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
    xlimdeadzone=0.3,
    thetalabel=r"$\theta$-label", xlabel=r"$x$-label", ylabel=r"$y$-label",
    thetaarrowlength=np.pi/2,
    thetatickkwargs=None, thetaticklabelkwargs=None, thetalabelkwargs=None,
    xtickkwargs=None, xticklabelkwargs=None, xlabelkwargs=None,
)
LVPC.scatter(theta, X, Y)
LVPC.plot(theta, X, Y_nonoise, plot_kwargs=[dict(lw=3, c="w") for _ in theta])
LVPC.plot(theta, X, Y_nonoise)

# fig.tight_layout()
plt.show()
