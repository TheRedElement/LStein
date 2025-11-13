#%%imports
import logging
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ..utils import polar2carth

from ..base.LSteinCanvas import LSteinCanvas
from ..base.LSteinPanel import LSteinPanel

logger = logging.getLogger(__name__)

#%%definitions
class LSteinPlotly:
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

        #convert kwargs from matplotlib to plotly
        self.translate_kwargs()

        return
    
    def translate_kwargs(self):
        """
            - method to translate default kwargs specified in `LSteinCanvas` and `LSteinPanel` to plotly
                - original kwargs are specified in matplotlib

            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        ls2plotly = {"-":"solid", "--":"dash", ":":"dot", "-.":"dashdot"}
        c2plotly = {"k":"#000000", "black":"#000000", "w":"#ffffff", "tab:grey":"#7f7f7f"}
        va2plotly = {"center":"middle"}

        #thetatickkwargs
        c, ls, lw = (self.LSC.thetatickkwargs.pop(k) for k in ["c", "ls", "lw"])
        if ls in ls2plotly.keys(): ls = ls2plotly[ls]
        if c in c2plotly.keys(): c = c2plotly[c]
        if "line" not in self.LSC.thetatickkwargs.keys(): self.LSC.thetatickkwargs["line"] = dict(color=c, dash=ls, width=lw,)
        #thetaticklabelkwargs
        c, ha, va = (self.LSC.thetaticklabelkwargs.pop(k) for k in ["c", "ha", "va"])
        if c in c2plotly.keys(): c = c2plotly[c]
        if va in va2plotly.keys(): va = va2plotly[va]
        if "font" not in self.LSC.thetaticklabelkwargs.keys(): self.LSC.thetaticklabelkwargs["font"] = dict(color=c,)
        if "xanchor" not in self.LSC.thetaticklabelkwargs.keys(): self.LSC.thetaticklabelkwargs["xanchor"] = ha
        if "yanchor" not in self.LSC.thetaticklabelkwargs.keys(): self.LSC.thetaticklabelkwargs["yanchor"] = va
        #thetalabelkwargs
        c, ha, va = (self.LSC.thetalabelkwargs.pop(k) for k in ["c", "ha", "va"])
        if c in c2plotly.keys(): c = c2plotly[c]
        if va in va2plotly.keys(): va = va2plotly[va]
        if "font" not in self.LSC.thetalabelkwargs.keys(): self.LSC.thetalabelkwargs["font"] = dict(color=c,)
        if "xanchor" not in self.LSC.thetalabelkwargs.keys(): self.LSC.thetalabelkwargs["xanchor"] = ha
        if "yanchor" not in self.LSC.thetalabelkwargs.keys(): self.LSC.thetalabelkwargs["yanchor"] = va
        #xtickkwargs
        c, ls, lw = (self.LSC.xtickkwargs.pop(k) for k in ["c", "ls", "lw"])
        if ls in ls2plotly.keys(): ls = ls2plotly[ls]
        if "line" not in self.LSC.xtickkwargs.keys(): self.LSC.xtickkwargs["line"] = dict(color=c, dash=ls, width=lw,)
        #xticklabelkwargs
        c, xytext, textcoords = (self.LSC.xticklabelkwargs.pop(k) for k in ["c", "xytext", "textcoords"])
        if c in c2plotly.keys(): c = c2plotly[c]
        if "font" not in self.LSC.xticklabelkwargs.keys(): self.LSC.xticklabelkwargs["font"] = dict(color=c,)
        if "xshift" not in self.LSC.xticklabelkwargs.keys(): self.LSC.xticklabelkwargs["xshift"] = xytext[0]    #will be interpreted as pixels!
        if "yshift" not in self.LSC.xticklabelkwargs.keys(): self.LSC.xticklabelkwargs["yshift"] = xytext[1]    #will be interpreted as pixels!
        #xlabelkwargs
        c, xytext, textcoords = (self.LSC.xlabelkwargs.pop(k) for k in ["c", "xytext", "textcoords"])
        if c in c2plotly.keys(): c = c2plotly[c]
        if "font" not in self.LSC.xlabelkwargs.keys(): self.LSC.xlabelkwargs["font"] = dict(color=c,)
        if "xshift" not in self.LSC.xlabelkwargs.keys(): self.LSC.xlabelkwargs["xshift"] = xytext[0]    #will be interpreted as pixels!
        if "yshift" not in self.LSC.xlabelkwargs.keys(): self.LSC.xlabelkwargs["yshift"] = xytext[1]    #will be interpreted as pixels!
        #ylabelkwargs
        c, = (self.LSC.ylabelkwargs.pop(k) for k in ["c"])
        if c in c2plotly.keys(): c = c2plotly[c]
        if "font" not in self.LSC.ylabelkwargs.keys(): self.LSC.ylabelkwargs["font"] = dict(color=c,)

        for LSP in self.LSC.Panels:
            #ytickkwargs
            c, ls, lw = (LSP.ytickkwargs.pop(k) for k in ["c", "ls", "lw"])
            if ls in ls2plotly.keys(): ls = ls2plotly[ls]
            if c in c2plotly.keys(): c = c2plotly[c]
            if "line" not in LSP.ytickkwargs.keys(): LSP.ytickkwargs["line"] = dict(color=c, dash=ls, width=lw,)
            #yticklabelkwargs
            c, ha, va = (LSP.yticklabelkwargs.pop(k) for k in ["c", "ha", "va"])
            if c in c2plotly.keys(): c = c2plotly[c]
            if va in va2plotly.keys(): va = va2plotly[va]
            if "font" not in LSP.yticklabelkwargs.keys(): LSP.yticklabelkwargs["font"] = dict(color=c,)
            if "xanchor" not in LSP.yticklabelkwargs.keys(): LSP.yticklabelkwargs["xanchor"] = ha
            if "yanchor" not in LSP.yticklabelkwargs.keys(): LSP.yticklabelkwargs["yanchor"] = va
            #panelboundskwargs
            c, = (LSP.panelboundskwargs.pop(k) for k in ["c"])
            if c in c2plotly.keys(): c = c2plotly[c]
            if "line" not in LSP.panelboundskwargs.keys(): LSP.panelboundskwargs["line"] = dict(color=c, dash="solid")
        return

    #canvas
    def add_xaxis(self,
        fig:go.Figure,
        row:int, col:int,
        ):
        """
            - method to add the x-axis to `fig`

            Parameters
            ----------
                - `fig`
                    - `Figure`
                    - plotly figure to draw into
                - `row`
                    - `int`
                    - row of the panel to plot into
                - `col`
                    - `int`
                    - column of the panel to plot into

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
        ##ticks
        for i in range(len(xticklabs)):
            ##circles
            fig.add_trace(
                go.Scatter(x=circles_x[i], y=circles_y[i],
                    showlegend=False,
                    **self.LSC.xtickkwargs,    
                ),
                row, col,
            )
            ##ticklabels
            fig.add_annotation(x=xtickpos_x[i], y=xtickpos_y[i], text=f"{xticklabs[i]}",
                row=row, col=col,
                showarrow=False,
                **self.LSC.xticklabelkwargs
            )
        ##axis label
        fig.add_annotation(x=xlabpos_x, y=xlabpos_y, text=self.LSC.xlabel,
            row=row, col=col,
            showarrow=False,
            **self.LSC.xlabelkwargs
        )

        return

    def add_thetaaxis(self,
        fig:go.Figure,
        row:int, col:int,
        ):
        """
            - method to add the theta-axis to `fig`

            Parameters
            ----------
                - `fig`
                    - `Figure`
                    - plotly figure to draw into
                - `row`
                    - `int`
                    - row of the panel to plot into
                - `col`
                    - `int`
                    - column of the panel to plot into

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
        for i in range(len(self.LSC.thetaticks[0])):
            ##ticks
            fig.add_trace(
                go.Scatter(x=[thetatickpos_xi[i],thetatickpos_xo[i]], y=[thetatickpos_yi[i],thetatickpos_yo[i]],
                    showlegend=False,
                    mode="lines",
                    **self.LSC.thetatickkwargs,
                ),
                row, col,
            )
            ##ticklabels
            fig.add_annotation(x=thetaticklabelpos_x[i], y=thetaticklabelpos_y[i], text=f"{thetaticklabs[i]}",
                showarrow=False,
                **self.LSC.thetaticklabelkwargs
            )

        ##axis label
        fig.add_annotation(x=th_label_x, y=th_label_y, text=self.LSC.thetalabel,
            showarrow=False,
            **self.LSC.thetalabelkwargs,
        )

        ##indicator
        dx = x_arrow[-1]-x_arrow[-2]
        dy = y_arrow[-1]-y_arrow[-2]
        fig.add_annotation(x=x_arrow[-1], y=y_arrow[-1],
            text="►",
            textangle=np.arctan2(dx, dy)/np.pi*180 - 90,
            font=dict(color=self.LSC.thetatickkwargs["line"]["color"]),
            xanchor="center",
            yanchor="middle",
            showarrow=False,
        )
        return

    def add_ylabel(self,
        fig:go.Figure,
        row:int, col:int,
        ):
        """
            - method to add the y-label to `fig`

            Parameters
            ----------
                - `fig`
                    - `Figure`
                    - plotly figure to draw into
                - `row`
                    - `int`
                    - row of the panel to plot into
                - `col`
                    - `int`
                    - column of the panel to plot into

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
        ##axis label
        fig.add_annotation(x=ylabpos_x, y=ylabpos_y, text=self.LSC.ylabel,
            row=row, col=col,
            showarrow=False,
            **self.LSC.ylabelkwargs
        )
        # ax.annotate(self.LSC.ylabel, xy=(ylabpos_x, ylabpos_y), annotation_clip=False, **self.LSC.ylabelkwargs)
        return

    #panels
    def add_yaxis(self,
        LSP:LSteinPanel,
        fig:go.Figure,
        row:int, col:int,        
        ):
        """
            - method to add the y-axis of `LSP` to `fig`

            Parameters
            ----------
                - `fig`
                    - `Figure`
                    - plotly figure to draw into
                - `row`
                    - `int`
                    - row of the panel to plot into
                - `col`
                    - `int`
                    - column of the panel to plot into
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
            for i in range(len(ytickpos_th)):
                ##ticks
                fig.add_trace(
                    go.Scatter(x=ytickpos_x[i], y=ytickpos_y[i],
                        showlegend=False,
                        mode="lines",
                        **LSP.ytickkwargs,
                    ),
                    row, col,
                )
                ##ticklabels
                fig.add_annotation(x=yticklabelpos_x[i], y=yticklabelpos_y[i], text=f"{yticklabs[i]}",
                    showarrow=False,
                    **LSP.yticklabelkwargs
                )
        ##panel boundaries
        if LSP.show_panelbounds:
            for xb, yb in zip(x_bounds, y_bounds):
                fig.add_trace(
                    go.Scatter(x=xb, y=yb,
                        showlegend=False,
                        mode="lines",
                        **LSP.panelboundskwargs,
                    ),
                    row, col,
                )

        return

    #plotting methods
    def scatter_(self,
        fig:go.Figure,
        row:int, col:int,                 
        x:np.ndarray, y:np.ndarray,
        *args, **kwargs
        ):
        """
            - method to add a scatterplot

            Parameters
            ----------
                - `fig`
                    - `Figure`
                    - plotly figure to draw into
                - `row`
                    - `int`
                    - row of the panel to plot into
                - `col`
                    - `int`
                    - column of the panel to plot into
                - `x`
                    - `np.ndarray`
                    - x-values of the series
                    - has to have same length as `y`
                - `y`
                    - `np.ndarray`
                    - y-values of the series
                    - has to have same length as `x`
                -`kwargs`
                    - kwargs to pass to `go.Scatter()`
                        
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - only to be called from within `LSteinPlotly.show()`
        """
        fig.add_trace(
            go.Scatter(x=x, y=y,
                mode="markers",
                **kwargs,    
            ),
            row, col,
        )
        return

    def plot_(self,
        fig:go.Figure,
        row:int, col:int,                  
        x:np.ndarray, y:np.ndarray,
        *args, **kwargs
        ):
        """
            - method to add a lineplot

            Parameters
            ----------
                - `fig`
                    - `Figure`
                    - plotly figure to draw into
                - `row`
                    - `int`
                    - row of the panel to plot into
                - `col`
                    - `int`
                    - column of the panel to plot into
                - `x`
                    - `np.ndarray`
                    - x-values of the series
                    - has to have same length as `y`
                - `y`
                    - `np.ndarray`
                    - y-values of the series
                    - has to have same length as `x`
                -`kwargs`
                    - kwargs to pass to `go.Scatter()`
                        
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - only to be called from within `LSteinPlotly.show()`
        """    
        fig.add_trace(
            go.Scatter(x=x, y=y,
                mode="lines",
                **kwargs,    
            ),
            row, col,
        )
        return

    #combined
    def show(self,
        fig:go.Figure,
        row:int, col:int,
        ):
        """
            - method to display `self.LSC` within a plotly figure
            - will
                - draw the canvas
                - add each panel
                - plot series for each panel

            Parameters
            ----------
                - `fig`
                    - `Figure`
                    - plotly figure to draw into
                - `row`
                    - `int`
                    - row of the panel to plot into
                - `col`
                    - `int`
                    - column of the panel to plot into
            
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `go.Figure`
                    - `fig` with the respective elements added

            Comments
            --------
        """


        #disable some default settings
        fig.update_yaxes(row=row, col=col, scaleratio=1, scaleanchor="x", visible=False)
        fig.update_xaxes(row=row, col=col, scaleratio=1, scaleanchor="y", visible=False)

        #add canvas elements
        self.add_xaxis(fig, row, col)
        self.add_thetaaxis(fig, row, col)
        self.add_ylabel(fig, row, col)

        #update switch denoting that canvas has been drawn
        self.LSC.canvas_drawn = True

        #draw panels
        for LSP in self.LSC.Panels:
            #draw panel if not drawn already
            if not LSP.panel_drawn:
                self.add_yaxis(LSP, fig, row, col)
                LSP.panel_drawn = True

            #plot all dataseries
            for ds in LSP.dataseries:

                if ds["seriestype"] == "scatter": func = self.scatter_
                elif ds["seriestype"] == "plot":  func = self.plot_
                else:
                    logger.warning(f"seriestype of {ds['seriestype']} is not supported. try one of `['scatter','plot']`")
                    continue

                func(fig, row, col, ds["x"], ds["y"], **ds["kwargs"])
        return
