#%%imports
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
from typing import Any, Dict, List, Literal, Tuple

# from .LVisPCanvas import LVisPCanvas   #no import because leads to curcular import
from .utils import minmaxscale, polar2carth, carth2polar


#%%classes
class LVisPPanel:
    """
        - class defining a panel sitting within a `LVisPCanvas`

        Attributes
        ----------
            - `LVPC`
                - `LVisPCanvas`
                - parent canvas the panel is associated with
            - `theta`
                - `float`
                - theta value the panel is associated with
                - equivalent to 2.5th dimension of the dataset
                - similar to `pos` in `fig.add_subolot(pos)`
                - determines where on `LVPC` the panel will be located
                    - created panel will be centered around `theta` with a width of `panelsize`
            - `yticks`
                `Tuple[List[float],List[Any]]`
                - ticks to draw for the y-axis
                - also defines axis limits applied to `y`
                    - i.e., bounds of the respective panel
                    - `np.min(yticks[0])` corresponds to the start of the panel
                    - `np.max(yticks[0])` corresponds to the end of the panel
                    - `yticks[0]` will be used as tickpositions
                    - `yticks[1]` will be used as ticklabels
            - `panelsize`
                - `float`, optional
                - (angular) space the created panel will occupy
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
            - `y_projection_method`
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

        Infered Attributes
        ------------------
            - `ylims`
                - `Tuple[float,float]`
                - axis limits applied to `y`
            - `ylimrange`
                - `real`
                - range of y-values
                - convenience field for relative definitions of plot elements
            - `panel_drawn`
                - `bool`
                - flag denoting if the panel has been drawn alrady
                - to prevent drawing the panel several times when plotting
        
        Methods
        -------
            - `get_thetabounds()`
            - `get_rbounds()`
            - `set_yticks()`
            - `draw()`
            - `apply_axis_limits()`
            - `project_xy_theta()`
            - `project_xy_y()`
            - `project_xy()`
            - `plot()`
            - `scatter()`

        Dependencies
        ------------
            - `matplotlib`
            - `numpy`
            - `typing`

        Comments
        --------
        
    """

    def __init__(self,
        LVPC,#:LVisPCanvas,
        theta:float,
        yticks:Tuple[List[float],List[Any]],
        panelsize:float=np.pi/8,
        show_panelbounds:bool=False, show_yticks:bool=True,
        y_projection_method:Literal["theta","y"]="theta",
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
        if "c" not in self.ytickkwargs.keys():  self.ytickkwargs["c"]  = plt.rcParams["grid.color"]
        if "ls" not in self.ytickkwargs.keys(): self.ytickkwargs["ls"] = plt.rcParams["grid.linestyle"]
        if "lw" not in self.ytickkwargs.keys(): self.ytickkwargs["lw"] = plt.rcParams["grid.linewidth"]
        self.yticklabelkwargs   = dict(c=plt.rcParams["axes.labelcolor"], ha="center", va="center", pad=0.1) if yticklabelkwargs is None else yticklabelkwargs
        if "c" not in self.yticklabelkwargs.keys():     self.yticklabelkwargs["c"]   = plt.rcParams["axes.labelcolor"]
        if "ha" not in self.yticklabelkwargs.keys():    self.yticklabelkwargs["ha"]  = "center"
        if "va" not in self.yticklabelkwargs.keys():    self.yticklabelkwargs["va"]  = "center"
        if "pad" not in self.yticklabelkwargs.keys():   self.yticklabelkwargs["pad"] = 0.1
        
        self.panelboundskwargs  = dict(c=plt.rcParams["axes.edgecolor"]) if panelboundskwargs is None else panelboundskwargs
        if "c" not in self.panelboundskwargs.keys():self.panelboundskwargs["c"] = plt.rcParams["axes.edgecolor"]

        #infered attributes
        self.ylims = (np.min(self.yticks[0]), np.max(self.yticks[0]))
        self.ylimrange = np.max(self.yticks[0]) - np.min(self.yticks[0])        
        self.panel_drawn = False

        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + ", ".join([f"{attr}={val}" for attr, val in self.__dict__.items()]) + ")"

    #panel methods
    def get_thetabounds(self) -> Tuple[float,float,float]:
        """
            - method to compute bounds of the panel as an angle measured from the x-axis counterclockwise (in radians)

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `theta_lb`
                    - `float`
                    - lower bound of the panel as an angle in radians
                    - corresponds to `self.ylims[0]`
                - `theta_ub`
                    - `float`
                    - upper bound of the panel as an angle in radians
                    - corresponds to `self.ylims[1]`
                - `theta_offset`
                    - `float`
                    - offset of the panel w.r.t. the x-axis
                    - defines the angular position of the central ray of the panel
                        - as an angle measured from the x-axis counterclockwise
                        - in radians

            Comments
            --------
        """
        theta_offset = minmaxscale(self.theta, #panel position
            self.LVPC.thetaplotlims[0], self.LVPC.thetaplotlims[1],
            xmin_ref=self.LVPC.thetaticks[0][0], xmax_ref=self.LVPC.thetaticks[0][-1]
        )
        theta_lb = theta_offset - self.panelsize/2   #lower bound of panel
        theta_ub = theta_offset + self.panelsize/2   #upper bound of panel
        return theta_offset, theta_lb, theta_ub

    def get_rbounds(self) -> Tuple[float,float]:
        """
            - method to compute bounds of the panel in radial direction

            Parameters
            ----------

            Raises
            ------

            Returns
            -------
                - `r_lb`
                    - `float`
                    - lower bound of the panel in x-direction (radially)
                    - located at `self.xlimdeadzone*self.LVPC.xlimrange`
                    - corresponds to `self.xlims[0]`
                - `r_ub`
                    - `float`
                    - upper bound of the panel in x-direction (radially)
                    - located at `self.LVPC.xlimrange`
                    - corresponds to `self.xlims[1]`

            Comments
            --------
        """        
        r_lb = self.LVPC.xlimdeadzone*self.LVPC.xlimrange
        r_ub = self.LVPC.xlimrange
        return r_lb, r_ub

    def get_yticks(self,
        theta_lb:float, theta_ub:float
        ) -> Tuple[List[float],List[Any]]:
        """
            - method to compute angular positions of the y-ticks angles measured from the x-axis counterclockwise (in radians)

            Parameters
            ----------
                - `theta_lb`
                    - `float`
                    - lower bound of the panel as an angle in radians
                    - corresponds to `self.ylims[0]`
                - `theta_ub`
                    - `float`
                    - upper bound of the panel as an angle in radians
                    - corresponds to `self.ylims[1]`

            Raises
            ------

            Returns
            -------
                - `ytickpos_th`
                    - `List[float]`
                    - tickpositions angles measured from the x-axis counterclockwise (in radians)
                - `yticklabs`
                    - `List[Any]`
                    - labels assigned to each tick
                    - same length as `ytickpos_th` 
            
            Comments
            --------
        """
        ytickpos_th = minmaxscale(self.yticks[0], theta_lb, theta_ub)
        yticklabs = self.yticks[1]

        return ytickpos_th, yticklabs

    def draw(self,
        ax:plt.Axes=None,
        ):
        """
            - method to draw the panel
                - just outlines, no data-series

            Parameters
            ----------
                - `ax`
                    - `plt.Axes`, optional
                    - axis to draw into
                    - the default is `None`
                        - will be set to `self.LVPC.ax` (axis of parent)

            Raises
            ------

            Returns
            -------
            
            Comments
            --------
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
        x_lb, y_lb  = polar2carth(r_bounds, theta_lb)
        x_ub, y_ub  = polar2carth(r_bounds, theta_ub)
        x_bounds = np.array([x_lb,x_ub])
        y_bounds = np.array([y_lb,y_ub])

        pad = self.yticklabelkwargs.pop("pad")   #padding for yticklabels
        r_, th_ = np.meshgrid(r_bounds, ytickpos_th)
        ytickpos_x, ytickpos_y              = polar2carth(r_, th_)
        yticklabelpos_x, yticklabelpos_y    = polar2carth((1+pad)*r_ub, ytickpos_th)

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
        ) -> Tuple[np.ndarray,np.ndarray,Dict]:
        """
            - method enforce axis limits onto the dataseries
                - only applies out-of-bounds cuts
                - removes any datapoints that are out of bounds in x- or y-direction

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the series to be plotted
                    - will serve as reference for enforcing `self.LVPC.xlims`
                - `y`
                    - `np.ndarray`
                    - y-values of the series to be plotted
                    - will serve as reference for enforcing `self.ylims`
                - `**kwargs`
                    - `kwargs` ultimately used when plotting `y` vs `x`
                    - also get modified accordingly i.e.,
                        - `"c"` needs to be set to same size as `x_cut` and `y_cut`
                        - `"s"` needs to be set to same size as `x_cut` and `y_cut`
                        - `"alpha"` needs to be set to same size as `x_cut` and `y_cut`

            Raises
            ------

            Returns
            -------
                - `x_cut`
                    - `np.ndarray`
                    - `x` after applying axis-limit cuts
                - `y_cut`
                    - `np.ndarray`
                    - `y` after applying axis-limit cuts
                - `kwargs`
                    - `Dict`
                    - `**kwargs` after applying axis-limit cuts

            
            Comments
            --------
        """

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
        r_min, th_min = carth2polar(self.LVPC.xlims[1], self.ylims[0])
        r_max, th_max = carth2polar(self.LVPC.xlims[1], self.ylims[1])

        #project x to obey axis-limits
        x_proj = minmaxscale(x,
            self.LVPC.xlimdeadzone*self.LVPC.xlimrange, self.LVPC.xlimrange,
            xmin_ref=self.LVPC.xlims[0], xmax_ref=self.LVPC.xlims[1],
        )

        #project y to fit into panel
        y_scaler = minmaxscale(x_proj,
            self.LVPC.xlimdeadzone, 1,
            xmin_ref=self.LVPC.xlimdeadzone*self.LVPC.xlimrange, xmax_ref=self.LVPC.xlimrange
        )
        y_proj = y_scaler * y
        
        #convert to polar coords for transformations
        r, theta = carth2polar(x_proj, y_proj)
        
        ##rescale theta (i.e., make sure y obeys axis limits)
        theta = minmaxscale(theta,
            theta_lb, theta_ub,
            xmin_ref=th_min, xmax_ref=th_max,
        )

        #convert back to carthesian coords for plotting
        #NOTE: use x_proj as radius because x is plotted in radial direction
        x_proj, y_proj = polar2carth(x_proj, theta)

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
                - can lead to unpredictable offsets in y-direction
        """

        #global variables
        theta_offset, theta_lb, theta_ub = self.get_thetabounds()

        #project x to obey axis-limits
        x_proj = minmaxscale(x,
            self.LVPC.xlimdeadzone*self.LVPC.xlimrange, self.LVPC.xlimrange,
            xmin_ref=self.LVPC.xlims[0], xmax_ref=self.LVPC.xlims[1],
        )

        #project y to fit into panel
        y_scaler = minmaxscale(x_proj,
            self.LVPC.xlimdeadzone, 1,
            xmin_ref=self.LVPC.xlimdeadzone*self.LVPC.xlimrange, xmax_ref=self.LVPC.xlimrange
        )
        y_proj = minmaxscale(y, 0, 1, xmin_ref=self.ylims[0], xmax_ref=self.ylims[1])
        y_proj = y_scaler * y_proj
        
        ##rescale to fill panel (considering bounds)
        y_slice = x_proj * np.tan(self.panelsize)
        y_proj = np.max(y_slice) * y_proj - y_slice/2
        
        #convert to polar coords for transformations
        r, theta = carth2polar(x_proj, y_proj)
        theta += theta_offset + np.pi

        #convert back to carthesian coords for plotting
        x_proj, y_proj = polar2carth(r, theta)

        return x_proj, y_proj
    
    def project_xy(self,
        x:np.ndarray, y:np.ndarray,
        y_projection_method:Literal["theta", "y"]="theta"
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
                - `y_projection_method`
                    - `Literal["theta","y"]`, optioal
                    - method to use for the projection
                    - the default is `theta`
                        - uses `self.project_xy_theta()`

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
                - generally `y_projection_method="theta"` is the preferred modus operandi
        """

        if y_projection_method == "theta":
            x_proj, y_proj = self.project_xy_theta(x, y)
        elif y_projection_method == "y":
            x_proj, y_proj = self.project_xy_y(x, y)

        return x_proj, y_proj
    

    #plotting methods
    def plot(self,
        x:np.ndarray, y:np.ndarray,
        ax:plt.Axes=None,
        **kwargs,
        ) -> List[plt.Line2D]:
        """
            - function to plot a series in the panel
            - similar to `ax.plot()`

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the series
                    - has to have same length as `y`
                - `y`
                    - `np.ndarray`
                    - y-values of the series
                    - has to have same length as `x`
                - `ax`
                    - `plt.Axes`, optional
                    - axis to draw into
                    - the default is `None`
                        - will be set to `self.LVPC.ax` (axis of parent)
                -`**kwargs`
                    - kwargs to pass to `ax.plot()`
                        
            Raises
            ------

            Returns
            -------
                - `line`
                    - `List[plt.Line2D]`
                    - list of created lines
                    - output from `ax.plot()`

            Comments
            --------        
        """

        #default parameters
        if ax is None: ax = self.LVPC.ax

        #draw canvas, panel if not done already
        if not self.LVPC.canvas_drawn: self.LVPC.draw()
        if not self.panel_drawn: self.draw()

        #apply axis limits
        x_cut, y_cut, kwargs = self.apply_axis_limits(x, y, **kwargs)

        #project x and y
        x_proj, y_proj = self.project_xy(x_cut, y_cut, self.y_projection_method)

        #plotting
        line = ax.plot(x_proj, y_proj, **kwargs)

        # #NOTE: temp
        # maxidx = np.argmax(y_cut)
        # ax.scatter(x_proj[maxidx], y_proj[maxidx])

        return line

    def scatter(self,
        x:np.ndarray, y:np.ndarray,
        ax:plt.Axes=None,
        **kwargs,
        ) -> PathCollection:
        """
            - function to plot a series in the panel
            - similar to `ax.scatter()`

            Parameters
            ----------
                - `x`
                    - `np.ndarray`
                    - x-values of the series
                    - has to have same length as `y`
                - `y`
                    - `np.ndarray`
                    - y-values of the series
                    - has to have same length as `x`
                - `ax`
                    - `plt.Axes`, optional
                    - axis to draw into
                    - the default is `None`
                        - will be set to `self.LVPC.ax` (axis of parent)
                -`**kwargs`
                    - kwargs to pass to `ax.scatter()`
                        
            Raises
            ------

            Returns
            -------
                - `sctr`
                    - `plt.PathCollection`
                    - list of created datapoints
                    - output from `ax.scatter()`

            Comments
            --------        
        """
        #default parameters
        if ax is None: ax = self.LVPC.ax

        #draw canvas, panel if not done already
        if not self.LVPC.canvas_drawn: self.LVPC.draw()
        if not self.panel_drawn: self.draw()

        #apply axis limits
        x_cut, y_cut, kwargs = self.apply_axis_limits(x, y, **kwargs)

        #project x and y
        x_proj, y_proj = self.project_xy(x_cut, y_cut, self.y_projection_method)

        #plotting
        sctr = ax.scatter(x_proj, y_proj, **kwargs)

        return sctr

