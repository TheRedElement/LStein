#%%imports
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Literal, Tuple

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from LVisPCanvas import LVisPCanvas   #no import because leads to curcular import
import utils as lvisu


#%%classes
class LVisPPanel:

    def __init__(self,
        LVPC,#:LVisPCanvas,
        theta:float,
        yticks:Tuple[List[float],List[Any]]=None,
        panelsize:float=np.pi/8,
        show_panelbounds:bool=False, show_yticks:bool=True,
        y_projection_method:Literal["y","theta"]="y",
        ytickkwargs:dict=None, yticklabelkwargs:dict=None,
        panelboundskwargs:dict=None,
        ):

        self.LVPC               = LVPC
        
        self.theta              = theta
        
        self.yticks             = yticks
        
        self.panelsize          = panelsize

        self.show_panelbounds   = show_panelbounds
        self.show_yticks        = show_yticks

        self.y_projection_method= y_projection_method

        self.ytickkwargs        = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if ytickkwargs is None else ytickkwargs
        self.yticklabelkwargs   = dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center", pad=0.1) if yticklabelkwargs is None else yticklabelkwargs
        
        self.panelboundskwargs  = dict(c=plt.rcParams["axes.edgecolor"]) if panelboundskwargs is None else panelboundskwargs

        #infered attributes
        self.ylims = (np.min(self.yticks[0]), np.max(self.yticks[0]))
        self.ylimrange = np.max(self.yticks[0]) - np.min(self.yticks[0])        
        self.panel_drawn = False

        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + ", ".join([f"{attr}={val}" for attr, val in self.__dict__.items()]) + ")"

    #panel methods
    def get_thetabounds(self) -> Tuple[float,float,float]:
        theta_offset = lvisu.minmaxscale(self.theta, #panel position
            self.LVPC.thetaplotlims[0], self.LVPC.thetaplotlims[1],
            xmin_ref=self.LVPC.thetaticks[0][0], xmax_ref=self.LVPC.thetaticks[0][-1]
        )
        theta_lb = theta_offset - self.panelsize/2   #lower bound of panel
        theta_ub = theta_offset + self.panelsize/2   #upper bound of panel
        return theta_offset, theta_lb, theta_ub

    def get_rbounds(self) -> Tuple[float,float]:
        r_lb = self.LVPC.xlimdeadzone*self.LVPC.xlimrange
        r_ub = self.LVPC.xlimrange
        return r_lb, r_ub

    def get_yticks(self,
        th_lb:float, th_ub:float
        ) -> Tuple[List[float],List[Any]]:
        ytickpos_th = lvisu.minmaxscale(self.yticks[0], th_lb, th_ub)
        yticklabs = self.yticks[1]

        return ytickpos_th, yticklabs

    def draw_LVisPPanel(self,
        ax:plt.Axes=None,
        ):
        """
            - method to draw the panel
                - just outlines, no data-series
        """

        #default parameters
        if ax is None: ax = self.LVPC.ax

        #get panel boundaries
        theta_offset, theta_lb, theta_ub = self.get_thetabounds()
        r_lb, r_ub = self.get_rbounds()
        r_bounds = np.array([r_lb, r_ub])

        #get yticks
        ytickpos_th, yticklabs = self.get_yticks(theta_lb, theta_ub)

        #convert to carthesian for plotting
        x_lb, y_lb  = lvisu.polar2carth(r_bounds, theta_lb)
        x_ub, y_ub  = lvisu.polar2carth(r_bounds, theta_ub)
        x_bounds = np.array([x_lb,x_ub])
        y_bounds = np.array([y_lb,y_ub])

        pad = self.yticklabelkwargs.pop("pad")   #padding for yticklabels
        r_, th_ = np.meshgrid(r_bounds, ytickpos_th)
        ytickpos_x, ytickpos_y              = lvisu.polar2carth(r_, th_)
        yticklabelpos_x, yticklabelpos_y    = lvisu.polar2carth((1+pad)*r_ub, ytickpos_th)

        if self.show_yticks:
            ax.plot(ytickpos_x.T, ytickpos_y.T, **self.ytickkwargs)
            for i in range(len(ytickpos_th)):
                ax.annotate(yticklabs[i], xy=(yticklabelpos_x[i],yticklabelpos_y[i]), annotation_clip=False, **self.yticklabelkwargs)
        if self.show_panelbounds: ax.plot(x_bounds.T, y_bounds.T, **self.panelboundskwargs)

        #update switch denoting that panel has been drawn
        self.panel_drawn = True

        return
    

    #dataseries methods
    def apply_axis_limits(self,
        x:np.ndarray, y:np.ndarray,
        **kwargs,
        ) -> Tuple[np.ndarray,np.ndarray, Dict]:

        x_bool = (self.LVPC.xlims[0]<=x)&(x<=self.LVPC.xlims[1])
        y_bool = (self.ylims[0]<=y)&(y<=self.ylims[1])
        limitbool = (x_bool&y_bool)

        x_cut = x[limitbool]
        y_cut = y[limitbool]

        #modifications of array kwargs
        if "c" in kwargs.keys():
            if isinstance(kwargs["c"], (np.ndarray, list)): kwargs["c"] = kwargs["c"][limitbool]
        if "s" in kwargs.keys(): 
            if isinstance(kwargs["s"], (np.ndarray, list)) : kwargs["s"] = kwargs["s"][limitbool]
        if "alpha" in kwargs.keys(): 
            if isinstance(kwargs["alpha"], (np.ndarray, list)) : kwargs["alpha"] = kwargs["alpha"][limitbool]
            
        
        return x_cut, y_cut, kwargs
    

    def project_xy_theta(self,
        x:np.ndarray, y:np.ndarray,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method implementing a way to project `x` and `y` into the panel
            - operates in `theta`-space when projecting the series

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the series to be projected into the panel
                - `y`
                    - `np.ndarray`
                    - y-values of the series to be projected into the panel


            Raises
            ------

            Returns
            -------
                - `x_proj`
                    - `np.ndarray`
                    - `x` after projection
                - `y_proj`
                    - `np.ndarray`
                    - `y` after projection

            Comments
            --------
                - more distorsion in x-direction
                - more accurate representation of y-direction
        """

        #global variables
        theta_offset, theta_lb, theta_ub = self.get_thetabounds()

        #convert ylims to theta-values
        r_min, th_min = lvisu.carth2polar(self.LVPC.xlims[1], self.ylims[0])
        r_max, th_max = lvisu.carth2polar(self.LVPC.xlims[1], self.ylims[1])

        #project x to obey axis-limits
        x_proj = lvisu.minmaxscale(x,
            self.LVPC.xlimdeadzone*self.LVPC.xlimrange, self.LVPC.xlimrange,
            xmin_ref=self.LVPC.xlims[0], xmax_ref=self.LVPC.xlims[1],
        )

        #project y to fit into panel
        y_scaler = lvisu.minmaxscale(x_proj,
            self.LVPC.xlimdeadzone, 1,
            xmin_ref=self.LVPC.xlimdeadzone*self.LVPC.xlimrange, xmax_ref=self.LVPC.xlimrange
        )
        y_proj = y_scaler * y
        
        #convert to polar coords for transformations
        r, theta = lvisu.carth2polar(x_proj, y_proj)
        
        ##rescale theta (i.e., make sure y obeys axis limits)
        theta = lvisu.minmaxscale(theta,
            theta_lb, theta_ub,
            # theta_lb, theta_ub,
            xmin_ref=th_min, xmax_ref=th_max,
        )

        #convert back to carthesian coords for plotting
        x_proj, y_proj = lvisu.polar2carth(r, theta)

        return x_proj, y_proj
    
    def project_xy_y(self,
        x:np.ndarray, y:np.ndarray,
        ) -> Tuple[np.ndarray,np.ndarray, Dict]:
        """
            - method implementing a way to project `x` and `y` into the panel
            - operates in `y`-space when projecting the series

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the series to be projected into the panel
                - `y`
                    - `np.ndarray`
                    - y-values of the series to be projected into the panel


            Raises
            ------

            Returns
            -------
                - `x_proj`
                    - `np.ndarray`
                    - `x` after projection
                - `y_proj`
                    - `np.ndarray`
                    - `y` after projection

            Comments
            --------
                - less distorsion in x-direction
                - less accurate values in y-direction
        """

        #global variables
        theta_offset, theta_lb, theta_ub = self.get_thetabounds()

        #project x to obey axis-limits
        x_proj = lvisu.minmaxscale(x,
            self.LVPC.xlimdeadzone*self.LVPC.xlimrange, self.LVPC.xlimrange,
            xmin_ref=self.LVPC.xlims[0], xmax_ref=self.LVPC.xlims[1],
        )

        #project y to fit into panel
        y_scaler = lvisu.minmaxscale(x_proj,
            self.LVPC.xlimdeadzone, 1,
            xmin_ref=self.LVPC.xlimdeadzone*self.LVPC.xlimrange, xmax_ref=self.LVPC.xlimrange
        )
        y_proj = lvisu.minmaxscale(y, 0, 1, xmin_ref=self.ylims[0], xmax_ref=self.ylims[1])
        y_proj = y_scaler * y_proj
        
        ##rescale to fill panel (considering bounds)
        y_slice = x_proj * np.tan(self.panelsize)
        y_proj = np.max(y_slice) * y_proj - y_slice/2
        
        #convert to polar coords for transformations
        r, theta = lvisu.carth2polar(x_proj, y_proj)
        theta += theta_offset + np.pi

        #convert back to carthesian coords for plotting
        x_proj, y_proj = lvisu.polar2carth(r, theta)

        return x_proj, y_proj
    
    def project_xy(self,
        x:np.ndarray, y:np.ndarray,
        y_projection_mode:Literal["theta", "y"]="y"
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to project `x` and `y` into the panel using `y_projection_method`

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the series to be projected into the panel
                - `y`
                    - `np.ndarray`
                    - y-values of the series to be projected into the panel
                - `y_projection_mode`
                    - `Literal["theta","y"]`, optioal
                    - method to use for the projection
                    - the default is `y`
                        - uses `self.project_xy_y()`

            Raises
            ------

            Returns
            -------
                - `x_proj`
                    - `np.ndarray`
                    - `x` after projection
                - `y_proj`
                    - `np.ndarray`
                    - `y` after projection

            Comments
            --------
                - more distorsion in x-direction
                - more accurate representation of y-direction
        """

        if y_projection_mode == "theta":
            x_proj, y_proj = self.project_xy_theta(x, y)
        elif y_projection_mode == "y":
            x_proj, y_proj = self.project_xy_y(x, y)

        return x_proj, y_proj
    

    #plotting methods
    def plot(self,
        x:np.ndarray, y:np.ndarray,
        ax:plt.Axes=None,
        **kwargs,
        ):

        #default parameters
        if ax is None: ax = self.LVPC.ax

        #draw canvas, panel if not done already
        if not self.LVPC.canvas_drawn: self.LVPC.draw_LVisPCanvas()
        if not self.panel_drawn: self.draw_LVisPPanel()

        #apply axis limits
        x_cut, y_cut, kwargs = self.apply_axis_limits(x, y, **kwargs)

        #project x and y
        x_proj, y_proj = self.project_xy(x_cut, y_cut, self.y_projection_method)

        #plotting
        line = ax.plot(x_proj, y_proj, **kwargs)

        # #TODO: temp
        # maxidx = np.argmax(y_cut)
        # ax.scatter(x_proj[maxidx], y_proj[maxidx])

        return line

    def scatter(self,
        x:np.ndarray, y:np.ndarray,
        ax:plt.Axes=None,
        **kwargs,
        ):

        #default parameters
        if ax is None: ax = self.LVPC.ax

        #draw canvas, panel if not done already
        if not self.LVPC.canvas_drawn: self.LVPC.draw_LVisPCanvas()
        if not self.panel_drawn: self.draw_LVisPPanel()

        #apply axis limits
        x_cut, y_cut, kwargs = self.apply_axis_limits(x, y, **kwargs)

        #project x and y
        x_proj, y_proj = self.project_xy(x_cut, y_cut, self.y_projection_method)

        #plotting
        sctr = ax.scatter(x_proj, y_proj, **kwargs)

        return sctr

