#%%imports
import logging
import matplotlib.pyplot as plt
import numpy as np
from ..utils import polar2carth

from ..base.LSteinCanvas import LSteinCanvas
from ..base.LSteinPanel import LSteinPanel

logger = logging.getLogger(__name__)

#%%definitions
class LSteinMPL:
    """
        - matplotlib backend to show an `LSteinCanvas` with all its `LSteinPanel` elements

        Attributes
        ----------
            - `LSC`
                - `LSteinCanvas`
                - canvas to display

        Methods
        -------
            - `add_xaxis()`
            - `add_thetaaxis()`
            - `add_yaxis()`
            - `add_ylabel()`
            - `add_yaxis()`
            - `scatter_()`
            - `plot_()`
            - `show()`

        Dependencies
        ------------
            - `logging`
            - `matplotlib`
            - `numpy`

        Comments
        --------
            - `ax` as method argument to ensure signature equivalence of different backends
    """
    
    def __init__(self,
        LSC:LSteinCanvas,
        ):
        self.LSC = LSC
        return

    #canvas
    def add_xaxis(self,
        ax:plt.Axes
        ):
        """
            - method to add the x-axis to `ax`

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - axis to draw into

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        #get quantities
        circles_x, circles_y, \
        xtickpos_x, xtickpos_y, xticklabs, \
        xlabpos_x, xlabpos_y, = self.LSC.compute_xaxis()

        #plotting
        ax.plot(circles_x.T, circles_y.T, **self.LSC.xtickkwargs)
        for i in range(len(xticklabs)):
            ax.annotate(xticklabs[i], xy=(xtickpos_x[i], xtickpos_y[i]), annotation_clip=False, **self.LSC.xticklabelkwargs)
        ax.annotate(self.LSC.xlabel, xy=(xlabpos_x, xlabpos_y), annotation_clip=False, **self.LSC.xlabelkwargs)

        return

    def add_thetaaxis(self,
        ax:plt.Axes
        ):
        """
            - method to add the theta-axis to `ax`

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - axis to draw into

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """    
        #get quantities
        thetatickpos_xi, thetatickpos_yi, thetatickpos_xo, thetatickpos_yo, \
        thetaticklabelpos_x, thetaticklabelpos_y, thetaticklabs, \
        th_label_x, th_label_y, \
        x_arrow, y_arrow, = self.LSC.compute_thetaaxis()

        #plotting
        ax.plot(np.array([thetatickpos_xi, thetatickpos_xo]), np.array([thetatickpos_yi, thetatickpos_yo]), **self.LSC.thetatickkwargs)
        for i in range(len(self.LSC.thetaticks[0])):    #ticklabels
            ax.annotate(f"{thetaticklabs[i]}", xy=(thetaticklabelpos_x[i], thetaticklabelpos_y[i]), annotation_clip=False, **self.LSC.thetaticklabelkwargs)
        line, = ax.plot(x_arrow[:-1], y_arrow[:-1], **self.LSC.thetatickkwargs)
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
        ax.annotate(self.LSC.thetalabel, xy=(th_label_x,th_label_y), annotation_clip=False, **self.LSC.thetalabelkwargs)
        return

    def add_ylabel(self,
        ax:plt.Axes
        ):
        """
            - method to add the y-label to `ax`

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - axis to draw into

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """        
        #get quantities
        ylabpos_x, ylabpos_y = self.LSC.compute_ylabel()
        #plotting
        ax.annotate(self.LSC.ylabel, xy=(ylabpos_x, ylabpos_y), annotation_clip=False, **self.LSC.ylabelkwargs)
        return

    #panels
    def add_yaxis(self,
        LSP:LSteinPanel, ax:plt.Axes
        ):
        """
            - method to add the y-axis of `LSP` to `ax`

            Parameters
            ----------
                - `LSP`
                    - `LSteinPanel`
                    - y-axis of this canvas will be drawn
                - `ax`
                    - `plt.Axes`
                    - axis to draw into

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """    
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

        #plotting
        if LSP.show_yticks:
            ax.plot(ytickpos_x.T, ytickpos_y.T, **LSP.ytickkwargs)
            for i in range(len(ytickpos_th)):
                ax.annotate(yticklabs[i], xy=(yticklabelpos_x[i],yticklabelpos_y[i]), annotation_clip=False, **LSP.yticklabelkwargs)
        if LSP.show_panelbounds: ax.plot(x_bounds.T, y_bounds.T, **LSP.panelboundskwargs)

        return

    #plotting methods
    def scatter_(self,
        ax:plt.Axes, x:np.ndarray, y:np.ndarray,
        *args, **kwargs
        ):
        """
            - method to add a scatterplot

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - axis to draw into
                - `x`
                    - `np.ndarray`
                    - x-values of the series
                    - has to have same length as `y`
                - `y`
                    - `np.ndarray`
                    - y-values of the series
                    - has to have same length as `x`
                -`kwargs`
                    - kwargs to pass to `ax.scatter()`
                        
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - only to be called from within `LSteinMPL.show()`
        """
        ax.scatter(x, y, **kwargs)
        return

    def plot_(self,
        ax:plt.Axes, x:np.ndarray, y:np.ndarray,
        *args, **kwargs
        ):
        """
            - method to add a lineplot

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - axis to draw into
                - `x`
                    - `np.ndarray`
                    - x-values of the series
                    - has to have same length as `y`
                - `y`
                    - `np.ndarray`
                    - y-values of the series
                    - has to have same length as `x`
                - `kwargs`
                    - kwargs to pass to `ax.plot()`
                        
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - only to be called from within `LSteinMPL.show()`
        """    
        ax.plot(x, y, **kwargs)
        return

    #combined
    def show(self, ax:plt.Axes):
        """
            - method to display `self.LSC` within a matplotlib figure
            - similar to `plt.show()`
            - will
                - draw the canvas
                - add each panel
                - plot series for each panel

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`
                    - axis to draw into
            
            Raises
            ------

            Returns
            -------
                - `ax`
                    - `plt.Axes`
                    - `ax` with the respective elements added

            Comments
            --------
        """

        #disable some default settings
        ax.set_aspect("equal")
        ax.set_axis_off()

        #add canvas elements
        self.add_xaxis(ax)
        self.add_thetaaxis(ax)
        self.add_ylabel(ax)

        #update switch denoting that canvas has been drawn
        self.LSC.canvas_drawn = True

        #draw panels
        for LSP in self.LSC.Panels:
            #draw panel if not drawn already
            if not LSP.panel_drawn:
                self.add_yaxis(LSP, ax)
                LSP.panel_drawn = True

            #plot all dataseries
            for ds in LSP.dataseries:

                if ds["seriestype"] == "scatter": func = self.scatter_
                elif ds["seriestype"] == "plot":  func = self.plot_
                else:
                    logger.warning(f"seriestype fof {ds['seriestype']} is not supported. try one of `['scatter','plot']`")
                    continue

                func(ax, ds["x"], ds["y"], **ds["kwargs"])
        return ax
