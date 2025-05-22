
#%%imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Any, Dict, List, Literal, Tuple, Union

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)

import utils as lvisu
from LVisPPanel import LVisPPanel

#%%classes
class LVisPCanvas:
    """
        - class containing the canvas to draw `LVisPPanel`s into
        - analogous to `matplotlib.figure.Figure`
        - parent to `LVisPPanel`

        Attributes
        ----------
            - `ax`
                `plt.Axes`
                - axes to add the `LVisPlot` to
            - `thetaticks`
                - `Tuple[List[float],List[Any]]`, `List[float]`
                - ticks to draw for the theta-axis (angular positioning)
                - also defines axis limits applied to `theta`
                    - i.e., in azimuthal direction
                    - `np.min(thetaticks[0])` corresponds to the lowest value of `theta` that will be plotted
                    - `np.max(thetaticks[0])` corresponds to the highest value of `theta` that will be plotted
                - if `List[float]`
                    - will use `thetaticks` as labels as well
                - if `Tuple[List[float],List[Any]]`
                    - will use `thetaticks[1]` as ticklabels
            - `xticks`
                - `Tuple[List[float],List[Any]]`, `List[float]`
                - ticks (circles) to draw for the x-axis
                - also defines axis limits applied to `x`
                    - i.e., in radial direction
                    - `np.min(xticks[0])` corresponds to the end of `xlimdeadzone`
                    - `np.max(xticks[0])` corresponds to the value plotted at the outer bound of the LVisPlot
                - if `List[float]`
                    - will use `xticks` as labels as well
                - if `Tuple[List[float],List[Any]]`
                    - will use `xticks[1]` as ticklabels
            - `yticks`
                - `Tuple[List[float],List[Any]]`, `List[float]`
                - ticks to draw for the y-axis
                - also defines axis limits applied to `y`
                    - i.e., bounds of the respective panel
                    - `np.min(yticks[0])` corresponds to the start of the panel
                    - `np.max(yticks[0])` corresponds to the end of the panel
                - if `List[float]`
                    - will use `yticks` as ticklabels as well
                - if `Tuple[List[float],List[Any]]`
                    - will use `yticks[1]` as ticklabels
            - `thetaguidelims`
                - `Tuple[float,float]`, optional
                - range to be spanned by the entire plot guides
                    - only affects the background grid
                - in radians
                - the default is `None`
                    - will be set to `(0,2*np.pi)`
                    - an entire circle will be plotted
            - `thetaplotlims`
                - `Tuple[float,float]`, optional
                - range to be populated by with theta-panels
                - sets the reference point for `thetaticks`
                    - `np.min(thetaticks[0])` will be plotted at `thetaplotlims[0]`
                    - `np.max(thetaticks[0])` will be plotted at `thetaplotlims[1]`
                - in radians
                - the default is `None`
                    - will be set to `thetaguidelims`
            - `xlimdeadzone`
                - `float`, optional
                - amount of space to leave empty in the center of the plot
                - provided as a fraction of the entire plot-radius
                - used to
                    - reduce projection effects at small radii
                    - have space for labelling
                - the default is `0.3`
                    - 30% of the radial direction is left empty
            - `thetalabel`
                - `str`, optional
                - label of the theta-axis
                - the default is `None`
                    - will be set to `""`
            - `xlabel`
                - `str`, optional
                - label of the x-axis
                - the default is `None`
                    - will be set to `""`
            - `ylabel`
                - `str`, optional
                - label of the y-axis
                - the default is `None`
                    - will be set to `""`
            - `th_arrowpos_th`
                - `float`, optional
                - position of the arrow indicating the theta-axis
                - given in units of `theta`
                - the default is `None`
                    - will be set to `np.mean(thetaticks[0])`
            - `ylabpos_th`
                - `float`, optional
                - position of `ylabel`
                - given in units of `theta`
                - the default is `None`
                    - will be set to `thetaticks[0][0]`
                    - at the first tick of the theta axis
            - `thetatickkwargs`
                - `dict`, optional
                - kwargs to pass to `ax.plot()` when drawing the theta ticks
                - used for styling
                - the default is `None`
                    - will be set to `dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"])`
            - `thetaticklabelkwargs`
                - `dict`, optional
                - kwargs to pass to `ax.annotate()` calls used for defining the ticklabels of the theta-axis
                - used for styling
                - `pad` determines the padding w.r.t. the ticks        
                - the default is `None`
                    - will be set to `dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center", pad=0.2)`
            - `thetalabelkwargs`
                - `dict`, optional
                - kwargs to pass to `ax.annotate()` call used for defining the axis label of the theta-axis
                - used for styling
                - the default is `None`
                    - will be set to `dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center")`
            - `xtickkwargs`
                - `dict`, optional
                - kwargs to pass to `ax.plot()` when drawing xticks (circles)
                - used for styling
                - the default is `None`
                    - will be set to `dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"])`
            - `xticklabelkwargs`
                - `dict`, optional
                - kwargs to pass to `ax.annotate()` calls used for defining the ticklabels of the x-axis
                - used for styling
                - the default is `None`
                    - will be set to `dict(c=plt.rcParams["axes.labelcolor"], textcoords="offset fontsize", xytext=(-1,-1))`
            - `xlabelkwargs`
                - `dict`, optional
                - kwargs to pass to `ax.annotate()` call used for defining the axis label of the x-axis
                - used for styling
                - the default is `None`
                    - will be set to `dict(c=plt.rcParams["axes.labelcolor"], textcoords="offset fontsize", xytext=(-2,-2))`
            - `ylabelkwargs`
                - `dict`, optional
                - kwargs to pass to `ax.annotate()` call used for defining the axis label of the y-axis
                - used for styling
                - `pad` determines the padding w.r.t. the ticks     
                - the default is `None`
                    - will be set to `dict(c=plt.rcParams["axes.labelcolor"], pad=0.15)`
            
        Infered Attributes
        ------------------
            - `thetalims`
                - `Tuple[float,float]`
                - axis limits applied to `theta`
                    - i.e., in azimuthal direction
                    - `thetalims[0]` corresponds to the lowest value of `theta` that will be plotted
                    - `thetalims[1]` corresponds to the highest value of `theta` that will be plotted
            - `xlims`
                - `Tuple[float,float]`
                - axis limits applied to `x`
                    - i.e., in radial direction
                    - `xlims[0]` corresponds to the value plotted at the end of `xlimdeadzone`
                    - `xlims[1]` corresponds to the value plotted at the outer bound of the LVisPlot
            - `xlimrange`
                - `real`
                - range of x-values
                - convenience field for relative definitions of plot elements
            - `Panels`
                - `List[LVisPPanel]`
                - collection of panels associated with `LVisPCanvas` instance
            - `canvas_drawn`
                - `bool`
                - flag denoting if the canvas has been drawn alrady
                - to prevent drawing the canvas several times when plotting

        Methods
        -------
            - `add_xaxis()`
            - `add_thetaaxis()`
            - `add_ylabel()`
            - `draw_LVisPCanvas()`
            - `add_panel()`
            - `get_thetas()`
            - `get_panel()`
            - `plot()`
            - `scatter()`

        Comments
        --------
    """
    def __init__(self,
        ax:plt.Axes,
        thetaticks:Union[Tuple[List[float],List[Any]],List[float]], xticks:Union[Tuple[List[float],List[Any]],List[float]], yticks:Union[Tuple[List[float],List[Any]],List[float]],
        thetaguidelims:Tuple[float,float]=None, thetaplotlims:Tuple[float,float]=None, xlimdeadzone:float=0.3,
        thetalabel:str=None, xlabel:str=None, ylabel:str=None,
        thetaarrowpos_th:float=None, ylabpos_th:float=None,
        thetatickkwargs:dict=None, thetaticklabelkwargs:dict=None, thetalabelkwargs:dict=None,
        xtickkwargs:dict=None, xticklabelkwargs:dict=None, xlabelkwargs:dict=None,
        ylabelkwargs:dict=None,
        ):
        
        self.ax             = ax

        self.thetaticks     = (thetaticks, thetaticks) if isinstance(thetaticks, (list, np.ndarray)) else thetaticks
        self.xticks         = (xticks, xticks) if isinstance(xticks, (list, np.ndarray)) else xticks
        self.yticks         = (yticks, yticks) if isinstance(yticks, (list, np.ndarray)) else yticks
        
        self.thetaguidelims = (0,2*np.pi) if thetaguidelims is None else thetaguidelims
        self.thetaplotlims  = thetaguidelims if thetaplotlims is None else thetaplotlims
        self.xlimdeadzone   = xlimdeadzone

        self.thetalabel     = "" if thetalabel is None else thetalabel
        self.xlabel         = "" if xlabel is None else xlabel
        self.ylabel         = "" if ylabel is None else ylabel

        self.thetaarrowpos_th   = np.mean(self.thetaticks[0]) if thetaarrowpos_th is None else thetaarrowpos_th
        self.ylabpos_th         = self.thetaticks[0][0] if ylabpos_th is None else ylabpos_th

        self.thetatickkwargs        = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if thetatickkwargs is None else thetatickkwargs
        self.thetaticklabelkwargs   = dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center", pad=0.2) if thetaticklabelkwargs is None else thetaticklabelkwargs
        if "pad" not in self.thetaticklabelkwargs.keys(): self.thetalabelkwargs["pad"] = 0.2
        self.thetalabelkwargs       = dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center") if thetalabelkwargs is None else thetalabelkwargs
        
        self.xtickkwargs            = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if xtickkwargs is None else xtickkwargs
        self.xticklabelkwargs       = dict(c=plt.rcParams["axes.labelcolor"], textcoords="offset fontsize", xytext=(-1,-1)) if xticklabelkwargs is None else xticklabelkwargs
        self.xlabelkwargs           = dict(c=plt.rcParams["axes.labelcolor"], textcoords="offset fontsize", xytext=(-2,-2)) if xlabelkwargs is None else xlabelkwargs
        
        self.ylabelkwargs           = dict(c=plt.rcParams["axes.labelcolor"], pad=0.15) if ylabelkwargs is None else ylabelkwargs
        if "pad" not in self.ylabelkwargs.keys(): self.ylabelkwargs["pad"] = 0.15

        #infered attributes
        self.thetalims = (np.min(self.thetaticks[0]), np.max(self.thetaticks[1]))
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
        """
            - method to add the x-axis to the Canvas

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`, optiona
                    - axis to draw into
                    - the default is `None`
                        - will draw into parent axis of `LVisPCanvas`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #default parameters
        if ax is None: ax = self.ax

        #xticks
        th_circ = np.linspace(self.thetaguidelims[0], self.thetaguidelims[1], 100)
        r_circ = self.xticks[0] - np.min(self.xticks[0])
        r_circ = lvisu.minmaxscale(r_circ, self.xlimrange * self.xlimdeadzone, self.xlimrange)
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
        """
            - method to add the theta-axis (azimuthal) to the Canvas

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`, optiona
                    - axis to draw into
                    - the default is `None`
                        - will draw into parent axis of `LVisPCanvas`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
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
        thetaarrowpos_th = lvisu.minmaxscale(np.linspace(self.thetalims[0], self.thetaarrowpos_th, 101),
            self.thetaplotlims[0], self.thetaplotlims[1],
            xmin_ref=self.thetaticks[0][0], xmax_ref=self.thetaticks[0][-1],
        )

        x_arrow, y_arrow = lvisu.polar2carth(1.0*thetatickpos_ro, thetaarrowpos_th)

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

    def add_ylabel(self,
        ax:plt.Axes=None,                 
        ):
        """
            - method to add the y-label to the Canvas

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`, optiona
                    - axis to draw into
                    - the default is `None`
                        - will draw into parent axis of `LVisPCanvas`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        #default parameters
        if ax is None: ax = self.ax

        ylabpos = lvisu.minmaxscale(self.ylabpos_th,
            self.thetaplotlims[0], self.thetaplotlims[1],
            xmin_ref=self.thetaticks[0][0], xmax_ref=self.thetaticks[0][-1],
        )

        pad = self.ylabelkwargs.pop("pad")
        ylabpos_x, ylabpos_y = lvisu.polar2carth((1+pad) * self.xlimrange, ylabpos)

        ax.annotate(self.ylabel, xy=(ylabpos_x, ylabpos_y), annotation_clip=False, **self.ylabelkwargs)

        return

    def draw_LVisPCanvas(self,
        ax:plt.Axes=None,                 
        ):
        """
            - method to combine Canvas elements and draw the canvas

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`, optiona
                    - axis to draw into
                    - the default is `None`
                        - will draw into parent axis of `LVisPCanvas`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        #default parameters
        if ax is None: ax = self.ax

        #disable some default settings
        ax.set_aspect("equal")
        ax.set_axis_off()

        #add canvas elements
        self.add_xaxis(ax)
        self.add_thetaaxis(ax)
        self.add_ylabel(ax)

        #update switch denoting that panel has been drawn
        self.canvas_drawn = True

        return

    #panel methods
    def add_panel(self,
        theta:float,
        yticks:Union[Tuple[List[float],List[Any]],List[float]]=None,
        panelsize:float=np.pi/8,
        show_panelbounds:bool=False, show_yticks:bool=True,
        y_projection_method:Literal["y","theta"]="theta",
        ytickkwargs:dict=None,
        yticklabelkwargs:dict=None,
        panelboundskwargs:dict=None,
        ) -> LVisPPanel:
        """
            - method to add a `LVisPPanel` to the Canvas
            - similar to matplotlibs `fig.add_subplot()`

            Parameters
            ----------
                - `theta`
                    - `float`
                    - theta value the panel is associated with
                    - equivalent to 2.5th dimension of the dataset
                    - determines where on the Canvas the panel will be located
                        - created panel will be centered around `theta`
                - `yticks`
                    `Tuple[List[float],List[Any]]`, `List[float]`, optional
                    - ticks to draw for the y-axis
                    - also defines axis limits applied to `y`
                        - i.e., bounds of the respective panel
                        - `np.min(yticks[0])` corresponds to the start of the panel
                        - `np.max(yticks[0])` corresponds to the end of the panel
                    - if `List[float]`
                        - will use `yticks` as ticklabels as well
                    - if `Tuple[List[float],List[Any]]`
                        - will use `yticks[1]` as ticklabels
                    - overrides `self.yticks`
                    - the default is `None`
                        - will fall back to `self.yticks`
                - `panelsize`
                    - `float`, optional
                    - (angular)space the created panel will occupy
                    - in radians
                    - the entire Canvas can allocate `(thetaguidelims[1]-thetaguidelims[0])/panelsize` evenly distributed, nonoverlapping panels
                    - the default is `np.pi/8`
                - `show_panelbounds`
                    - `bool`, optional
                    - whether to show bounds of the individual panels when rendering
                    - the default is `False`
                - `show_yticks`
                    - `bool`, optional
                    - whether to show ticks and gridlines for y-values
                    - the default is `True`
                - `y_projection_mode`
                    - `Literal["theta","y"]`, optioal
                    - method to use for the projection
                    - the default is `theta`
                        - uses `LVisPPanel.project_xy_theta()`
                - `ytickkwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.plot()` when drawing yticks (lines in radial direction)
                    - used for styling
                    - the default is `None`
                        - will be set to `dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"])`
                - `yticklabelkwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.annotate()` calls used for defining the ticklabels of the y-axis
                    - used for styling
                    - `pad` determines the padding w.r.t. the ticks        
                    - the default is `None`
                        - will be set to `dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center", pad=0.1)`
                - `panelboundskwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.plot()` when drawing bounds of each panel
                    - used for styling
                    - the default is `None`
                        - will be set to `dict(c=plt.rcParams["axes.edgecolor"])`
                    
            Raises
            ------

            Returns
            -------
                - `LVPP`
                    - `LVisPPanel`
                    - created panel

            Comments
            --------
        """
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
        """
            - method to get `theta` of all currently added Panels

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `thetas`
                    - `List[float]`
                    - theta values associated with each Panel in `self.Panels`

            Comments
            --------
        """        
        thetas = [P.theta for P in self.Panels]

        return thetas
    
    def get_panel(self,
        theta:float,
        ) -> LVisPPanel:
        """
            - method to get Panel associated with `theta`
            - only returns the first match

            Parameters
            ----------
                - `theta`
                    - `float`
                    - theta-value to extract panel of

            Raises
            ------

            Returns
            -------
                - `panel`
                    - `LVisPPanel`, `None`
                    - Panel associated with `theta`
                    - `None` if no Panel associated with `theta`

            Comments
            --------
        """  

        for P in self.Panels:
            if P.theta == theta:
                return P
        return None


    #convenience methods
    def plot(self,
        theta:np.ndarray, X:List[np.ndarray], Y:List[np.ndarray],
        panel_kwargs:List[Dict]=None,
        plot_kwargs:List[Dict]=None,
        ):
        """
            - convenience function to plot a set of series
            - similar yo `plt.plot()`

            Parameters
            ----------
                - `theta`
                    - `np.ndarray`
                    - theta values associated for to each series in `zip(X,Y)`
                    - 2.5th dimension
                - `X`
                    - `List[np.ndarray]`
                    - set of x-values of each series
                    - has to have same length as `theta`
                    - can contain arrays of different lengths
                        - have to have same length as corresponding entries in `Y`
                    - each series will be plotted in it's own Panel associated with `theta`
                - `Y`
                    - `List[np.ndarray]`
                    - set of y-values of each series
                    - has to have same length as `theta`
                    - can contain arrays of different lengths
                        - have to have same length as corresponding entries in `y`
                    - each series will be plotted in it's own Panel associated with `theta`
                - `panel_kwargs`
                    - `List[Dict]`, optional
                    - kwargs to pass to `self.add_panel()`
                    - has to have same length as `theta`
                    - the default is `None`
                        - will be set to `dict()` for all panels
                - `plot_kwargs`
                    - `List[Dict]`
                    - kwargs to pass to `LVisPPanel.plot()`
                    - has to have same length as `theta`
                    - the default is `None`
                        - will be set to `dict()` for all panels
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
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
        sctr_kwargs:List[Dict]=None,
        ):
        """
            - convenience function to plot a set of series
            - similar yo `plt.plot()`

            Parameters
            ----------
                - `theta`
                    - `np.ndarray`
                    - theta values associated for to each series in `zip(X,Y)`
                    - 2.5th dimension
                - `X`
                    - `List[np.ndarray]`
                    - set of x-values of each series
                    - has to have same length as `theta`
                    - can contain arrays of different lengths
                        - have to have same length as corresponding entries in `Y`
                    - each series will be plotted in it's own Panel associated with `theta`
                - `Y`
                    - `List[np.ndarray]`
                    - set of y-values of each series
                    - has to have same length as `theta`
                    - can contain arrays of different lengths
                        - have to have same length as corresponding entries in `y`
                    - each series will be plotted in it's own Panel associated with `theta`
                - `panel_kwargs`
                    - `List[Dict]`, optional
                    - kwargs to pass to `self.add_panel()`
                    - has to have same length as `theta`
                    - the default is `None`
                        - will be set to `dict()` for all panels
                - `sctr_kwargs`
                    - `List[Dict]`
                    - kwargs to pass to `LVisPPanel.scatter()`
                    - has to have same length as `theta`
                    - the default is `None`
                        - will be set to `dict()` for all panels
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        #default parameters
        panel_kwargs    = [dict() for _ in theta.__iter__()] if panel_kwargs is None else panel_kwargs
        sctr_kwargs     = [dict() for _ in theta.__iter__()] if sctr_kwargs is None else sctr_kwargs

        #get existing panels
        thetas = self.get_thetas()

        #generate colors
        colors = lvisu.get_colors(theta)
        for i in range(len(sctr_kwargs)):
            if "c" not in sctr_kwargs[i].keys(): sctr_kwargs[i]["c"] = mcolors.to_hex(colors[i])
        
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
            LVPP.scatter(X[i], Y[i], **sctr_kwargs[i])

        return    

#%%pseudo data

# theta, X, Y, Y_nonoise = simulate(4, opt="lc")
# # theta, X, Y, Y_nonoise = simulate(5, opt="sin")

# fig = plt.figure()
# for i in range(len(theta)):
#     plt.scatter(X[i], Y[i])
#     plt.plot(X[i], Y_nonoise[i])

# #%%
# thetaticks = np.round(np.linspace(np.floor(np.min(theta)), np.ceil(np.max(theta)), 4),0).astype(int)
# yticks = np.round(np.linspace(np.floor(np.min(np.concat(Y))), np.ceil(np.max(np.concat(Y))), 4), decimals=0).astype(int)
# # yticks = np.sort(np.append(yticks, [-10, 80]))
# panelsize = np.pi/8
# print(thetaticks)

# #%%standard usage
# fig = plt.figure(figsize=(5,9))
# ax = fig.add_subplot(111)
# LVPC = LVisPCanvas(ax,
#     thetaticks, [-20,0,100,120], yticks,
#     thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
#     xlimdeadzone=0.3,
#     thetalabel=r"$\theta$-label", xlabel=r"$x$-label", ylabel=r"$y$-label",
#     thetaarrowpos_th=None, ylabpos_th=np.min(theta),
#     thetatickkwargs=None, thetaticklabelkwargs=None, thetalabelkwargs=None,
#     xtickkwargs=None, xticklabelkwargs=None, xlabelkwargs=None,
# )
# colors = lvisu.get_colors(theta)

# for i in range(len(X)):
#     LVPP = LVPC.add_panel(
#         theta=theta[i],
#         # yticks=None,
#         yticks=yticks,
#         # yticks=(yticks, ["A", "B", "C", "D"]), 
#         panelsize=panelsize,
#         show_panelbounds=True, show_yticks=True,
#         y_projection_method="y",
#         # y_projection_method="theta",
#         ytickkwargs=None, yticklabelkwargs=None,
#         panelboundskwargs=None,
#     )

#     LVPP.scatter(X[i], Y[i], c=Y[i], s=5,  alpha=np.linspace(0, 1, Y[i].shape[0]))
#     LVPP.plot(X[i], Y_nonoise[i], c="w", lw=3)
#     LVPP.plot(X[i], Y_nonoise[i], color=colors[i])

# plt.show()

# #%%convenience usage
# print(np.sort(theta))
# fig = plt.figure(figsize=(5,9))
# ax = fig.add_subplot(111)
# LVPC = LVisPCanvas(ax,
#     thetaticks, [-20,0,100,120], yticks,
#     thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
#     xlimdeadzone=0.3,
#     thetalabel=r"$\theta$-label", xlabel=r"$x$-label", ylabel=r"$y$-label",
#     thetaarrowpos_th=2, ylabpos_th=None,
#     thetatickkwargs=dict(c="k"), thetaticklabelkwargs=None, thetalabelkwargs=None,
#     xtickkwargs=None, xticklabelkwargs=None, xlabelkwargs=None,
# )
# LVPC.scatter(theta, X, Y)
# LVPC.plot(theta, X, Y_nonoise, plot_kwargs=[dict(lw=3, c="w") for _ in theta])
# LVPC.plot(theta, X, Y_nonoise)

# # fig.tight_layout()
# plt.show()
