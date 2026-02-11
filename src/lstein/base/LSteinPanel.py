#%%imports
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Literal, Tuple

# from .LSteinCanvas import LSteinCanvas   #no import because leads to circular import
from ..utils import minmaxscale, polar2cart, cart2polar

import logging
logger = logging.getLogger(__name__)

#%%classes
class LSteinPanel:
    """represents a single panel in the `LSteinCanvas`

    - class defining a panel sitting within a `LSteinCanvas`

    Attributes
        - `LSC` -- see `__init__()`
        - `theta` -- see `__init__()`
        - `yticks` -- see `__init__()`
        - `panelsize` -- see `__init__()`
        - `show_panelbounds` -- see `__init__()`
        - `show_yticks` -- see `__init__()`
        - `y_projection_method` -- see `__init__()`
        - `ytickkwargs` -- see `__init__()`
        - `yticklabelkwargs` -- see `__init__()`
        - `panelboundskwargs` -- see `__init__()`
    
    Inferred Attributes
        - `ylims_data`
            - `Tuple[float,float]`
            - axis limits applied to `y`
        - `ylimrange_data`
            - `float`
            - range of y-values
            - convenience field for relative definitions of plot elements       
        - `ylims_plot`
            - `Tuple[float,float]`
            - limits used to plot the y-axis
            - sets the frame of reference for plotting
            - set to `(0,1)` for consistent results
        - `ylimrange_data`
            - `float`
            - range of plot values
            - convenience field for relative definitions of plot elements
        - `panel_drawn`
            - `bool`
            - flag denoting if the panel has been drawn already
            - to prevent drawing the panel several times when plotting
        - `dataseries`
            - `List[Dict[str,Any]]`
            - dataseries to be used for plotting
            - contains
                - `x`
                    - `np.ndarray`
                    - LSteinCanvastransformed (projected) dataseries in cartesian coordinates
                    - ready to be plotted
                - `y`
                    - `np.ndarray`
                    - transformed (projected) dataseries in cartesian coordinates
                    - ready to be plotted
                - `x_cut`
                    - `np.ndarray`
                    - original dataseries with axis-limits applied
                - `y_cut`
                    - `np.ndarray`
                    - original dataseries with axis-limits applied
            - `seriestype`
                - `Literal["scatter","line"]`
                - kind of the series to be displayed
                - used for implementation of plotting functions in backends
            - `kwargs`
                - kwargs to be passed to the respective plotting function in the backend
    
    Methods
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
        - `matplotlib`
        - `numpy`
        - `typing`
    """

    def __init__(self,
        LSC,#:LSteinCanvas,
        theta:float,
        yticks:Tuple[List[float],List[Any]],
        panelsize:float=np.pi/8,
        show_panelbounds:bool=False, show_yticks:bool=True,
        y_projection_method:Literal["theta","y"]="theta",
        ytickkwargs:dict=None, yticklabelkwargs:dict=None,
        panelboundskwargs:dict=None,
        ):
        """constructor

        - initializes class
        - computes inferred attributes
        
        Parameters
            - `LSC`
                - `LSteinCanvas`
                - parent canvas the panel is associated with
            - `theta`
                - `float`
                - theta value the panel is associated with
                - equivalent to 2.5th dimension of the dataset
                - similar to `pos` in `fig.add_subolot(pos)`
                - determines where on `LSC` the panel will be located
                    - created panel will be centered around `theta` with a width of `panelsize`
            - `yticks`
                `Tuple[List[float],List[Any]]`
                - ticks to draw for the y-axis
                - also defines axis limits applied to `y`
                    - i.e., bounds of the respective panel
                    - `yticks[0][0]` corresponds to the start of the panel
                    - `yticks[0][-1]` corresponds to the end of the panel
                    - `yticks[0]` will be used as tickpositions
                    - `yticks[1]` will be used as ticklabels
                - `yticks[0]` has to be sorted in ascending or descending order
                - to invert the y-axis pass `yticks[0]` in a reverse sorted manner
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
                - `Literal["theta","y"]`, optional
                - method to use for the projection
                - the default is `theta`
                    - uses `LSteinPanel.project_xy_theta()`
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

        Returns        
        """

        self.LSC               = LSC
        
        self.theta              = theta
        
        self.yticks             = (np.array(yticks[0]),yticks[1])
        
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

        #inferred attributes
        self.ylims_data = (self.yticks[0][0], self.yticks[0][-1])
        self.ylimrange_data = np.max(self.yticks[0]) - np.min(self.yticks[0])
        self.ylims_plot = (0, 1)
        self.ylimrange_plot = 1.0
        self.panel_drawn = False
        self.dataseries = []  #init list of dataseries to plot (List[Dict[str,Any]])

        return

    def __repr__(self) -> str:
        """returns string representation of the class"""
        return f"{self.__class__.__name__}(" + ", ".join([f"{attr}={val}" for attr, val in self.__dict__.items()]) + ")"

    #panel methods
    def get_thetabounds(self) -> Tuple[float,float,float]:
        """returns panel location and bounds as angles in radians

        - method to compute bounds of the panel as an angle measured from the x-axis counterclockwise (in radians)

        Parameters

        Raises

        Returns
            - `theta_lb`
                - `float`
                - lower bound of the panel as an angle in radians
                - corresponds to `self.ylims_plot[0]` and `self.ylims_data[0]`
            - `theta_ub`
                - `float`
                - upper bound of the panel as an angle in radians
                - corresponds to `self.ylims_plot[1]` and `self.ylims_data[1]`
            - `theta_offset`
                - `float`
                - offset of the panel w.r.t. the x-axis
                - defines the angular position of the central ray of the panel
                    - as an angle measured from the x-axis counterclockwise
                    - in radians
        """
        theta_offset = minmaxscale(self.theta, #panel position
            self.LSC.thetaplotlims[0], self.LSC.thetaplotlims[1],
            xmin_ref=self.LSC.thetaticks[0][0], xmax_ref=self.LSC.thetaticks[0][-1]
        )
        theta_lb = theta_offset - self.panelsize/2   #lower bound of panel
        theta_ub = theta_offset + self.panelsize/2   #upper bound of panel
        return theta_offset, theta_lb, theta_ub

    def get_rbounds(self) -> Tuple[float,float]:
        """returns panel bounds in radial direction

        - method to compute bounds of the panel in radial direction

        Parameters

        Raises

        Returns
            - `r_lb`
                - `float`
                - lower bound of the panel in x-direction (radially)
                - located at `self.xlimdeadzone*self.LSC.xlimrange_plot`
                - corresponds to `self.xlims[0]`
            - `r_ub`
                - `float`
                - upper bound of the panel in x-direction (radially)
                - located at `self.LSC.xlimrange_plot`
                - corresponds to `self.xlims[1]`

        Comments
        """        
        r_lb = self.LSC.xlimdeadzone*self.LSC.xlimrange_plot
        r_ub = self.LSC.xlimrange_plot
        return r_lb, r_ub

    def get_yticks(self,
        theta_lb:float, theta_ub:float
        ) -> Tuple[List[float],List[Any]]:
        """returns yticklabels and location of yticks

        - method to compute angular positions of the y-ticks angles measured from the x-axis counterclockwise (in radians)

        Parameters
            - `theta_lb`
                - `float`
                - lower bound of the panel as an angle in radians
                - corresponds to `self.ylims_plot[0]`
            - `theta_ub`
                - `float`
                - upper bound of the panel as an angle in radians
                - corresponds to `self.ylims_plot[1]`

        Raises

        Returns
            - `ytickpos_th`
                - `List[float]`
                - tickpositions angles measured from the x-axis counterclockwise (in radians)
            - `yticklabs`
                - `List[Any]`
                - labels assigned to each tick
                - same length as `ytickpos_th` 
        """
        ytickpos_th = minmaxscale(self.yticks[0], theta_lb, theta_ub, xmin_ref=self.ylims_data[0], xmax_ref=self.ylims_data[1])  #no use of min/max to allow inverted axis
        yticklabs = self.yticks[1]

        return ytickpos_th, yticklabs

    #dataseries methods
    def apply_axis_limits(self,
        x:np.ndarray, y:np.ndarray,
        **kwargs,
        ) -> Tuple[np.ndarray,np.ndarray,Dict]:
        """returns `x`, `y` and `**kwargs` after application of axis limits

        - method enforce axis limits onto the dataseries
            - only applies out-of-bounds cuts
            - removes any datapoints that are out of bounds in x- or y-direction

        Parameters
            - `x`
                - `np.ndarray`
                - x-values of the series to be plotted
                - will serve as reference for enforcing `self.LSC.xlims_plot`
            - `y`
                - `np.ndarray`
                - y-values of the series to be plotted
                - will serve as reference for enforcing `self.ylims_data`
            - `**kwargs`
                - `kwargs` ultimately used when plotting `y` vs `x`
                - also get modified accordingly i.e.,
                    - `"c"` needs to be set to same size as `x_cut` and `y_cut`
                    - `"s"` needs to be set to same size as `x_cut` and `y_cut`
                    - `"alpha"` needs to be set to same size as `x_cut` and `y_cut`

        Raises

        Returns
            - `x_cut`
                - `np.ndarray`
                - `x` after applying axis-limit cuts
            - `y_cut`
                - `np.ndarray`
                - `y` after applying axis-limit cuts
            - `kwargs`
                - `Dict`
                - `**kwargs` after applying axis-limit cuts
        """

        x_bool = (np.min(self.LSC.xlims_data)<=x)&(x<=np.max(self.LSC.xlims_data))
        y_bool = (np.min(self.ylims_data)<=y)&(y<=np.max(self.ylims_data))
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
    
    #projection methods
    def projection_preprocessing_(self,
        x:np.ndarray, y:np.ndarray,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """returns `x` and `y` after applying transformations preceding all projection methods
        
        - method applying preprocessing to `x` and `y`
        - applies transformation that precede all projection methods
        - only called from within the projection methods

        Parameters
            - `x`
                - `np.ndarray`
                - x-values of the series to be projected into the panel
            - `y`
                - `np.ndarray`
                - y-values of the series to be projected into the panel

        Raises

        Returns
            - `x_prep`
                - `np.ndarray`
                - `x` after application of preprocessing
            - `y_prep`
                - `np.ndarray`
                - `y` after application of preprocessing        
        """

        #rescale to plotting range (0,1) for more consistent results
        x_01 = minmaxscale(x,
            self.LSC.xlims_plot[0], self.LSC.xlims_plot[1],                         #0, 1,
            xmin_ref=self.LSC.xlims_data[0], xmax_ref=self.LSC.xlims_data[1],       #don't use min/max to allow for inverted axes
        )
        y_01 = minmaxscale(y,
            self.ylims_plot[0], self.ylims_plot[1],                                 #0, 1,
            xmin_ref=self.ylims_data[0], xmax_ref=self.ylims_data[1],               #don't use min/max to allow for inverted axes
        )

        #project x to obey axis-limits
        x_prep = minmaxscale(x_01,
            self.LSC.xlimdeadzone*self.LSC.xlimrange_plot, self.LSC.xlimrange_plot,
            xmin_ref=self.LSC.xlims_plot[0], xmax_ref=self.LSC.xlims_plot[1],
        )

        #project y to fit into panel/obey axis limits
        y_scaler = minmaxscale(x_prep,  #essentially scales x to [xlimdeadzone,1] #only needed if `self.xlims_plot[1] != 1`
            self.LSC.xlimdeadzone, 1,
            xmin_ref=self.LSC.xlimdeadzone*self.LSC.xlimrange_plot, xmax_ref=self.LSC.xlimrange_plot
        )
        y_prep = y_scaler * y_01

        return x_prep, y_prep
    
    def project_xy_theta(self,
        x:np.ndarray, y:np.ndarray,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """returns `x` and `y` after projection into the panel

        - method implementing a way to project `x` and `y` into the panel
        - operates in `theta`-space when projecting the series
        - advantages
            - more accurate representation of y-direction
        - downsides
            - more distorsion in x-direction

        Parameters
            - `x`
                - `np.ndarray`
                - x-values of the series to be projected into the panel
            - `y`
                - `np.ndarray`
                - y-values of the series to be projected into the panel

        Raises

        Returns
            - `x_proj`
                - `np.ndarray`
                - `x` after projection
            - `y_proj`
                - `np.ndarray`
                - `y` after projection
        """

        #global variables
        theta_offset, theta_lb, theta_ub = self.get_thetabounds()

        #convert ylims_plot to theta-values
        r_min, th_min = cart2polar(self.LSC.xlims_plot[1], self.ylims_plot[0])
        r_max, th_max = cart2polar(self.LSC.xlims_plot[1], self.ylims_plot[1])
        logger.debug(f"{th_min=}, {th_max=}")   #always np.pi+0 and np.pi+np.pi/4 since interval [0,1] chosen
        logger.debug(f"{self.LSC.xlims_plot=}, {self.ylims_plot=}")


        #preprocessing (applied to all projection methods)
        x_prep, y_prep = self.projection_preprocessing_(x, y)
        
        #convert to polar coords for transformations
        r, theta = cart2polar(x_prep, y_prep)
        
        ##rescale theta (i.e., make sure y obeys axis limits)
        theta = minmaxscale(theta,
            theta_lb, theta_ub,
            xmin_ref=th_min, xmax_ref=th_max,
        )

        #convert back to cartesian coords for plotting
        #NOTE: use x_prep as radius because x is plotted in radial direction
        x_proj, y_proj = polar2cart(x_prep, theta)

        return x_proj, y_proj
    
    def project_xy_y(self,
        x:np.ndarray, y:np.ndarray,
        ) -> Tuple[np.ndarray,np.ndarray, Dict]:
        """returns `x` and `y` after projection into the panel

        - method implementing a way to project `x` and `y` into the panel
        - operates in `y`-space when projecting the series
        - advantages
            - less distorsion in x-direction
        - downsides
            - can lead to unpredictable offsets in y-direction

        Parameters
            - `x`
                - `np.ndarray`
                - x-values of the series to be projected into the panel
            - `y`
                - `np.ndarray`
                - y-values of the series to be projected into the panel

        Raises

        Returns
            - `x_proj`
                - `np.ndarray`
                - `x` after projection
            - `y_proj`
                - `np.ndarray`
                - `y` after projection
        """

        #global variables
        theta_offset, theta_lb, theta_ub = self.get_thetabounds()

        #preprocessing (applied to all projection methods)
        x_prep, y_prep = self.projection_preprocessing_(x, y)
        
        #project y to obey panel bounds
        x_slice = x_prep                                                    #x-coordinate of slice at every datapoint
        y_slice = x_slice * np.tan(self.panelsize)                          #y-coordinate of slice at every datapoint considering panel size
        y_slice_ub_ylim = self.LSC.xlims_plot[1] * np.tan(self.panelsize)   #upper bound of the panel as defined by the y-limits
        y_slice_ub = max(np.max(y_slice), y_slice_ub_ylim)                  #actual upper bound of the panel (also considers dataseries being out of bounds)
        y_prep = y_slice_ub * y_prep - y_slice/2                            #adjust projection of datapoints in y #offset by half of the datapoints y_slice to make avoid overflows
        
        #convert to polar coords for transformations
        r, theta = cart2polar(x_prep, y_prep)
        theta += theta_offset + np.pi

        #convert back to cartesian coords for plotting
        x_proj, y_proj = polar2cart(r, theta)

        return x_proj, y_proj
    
    def project_xy(self,
        x:np.ndarray, y:np.ndarray,
        y_projection_method:Literal["theta","y"]="theta"
        ) -> Tuple[np.ndarray,np.ndarray]:
        """returns `x` and `y` after projection into the panel

        - method to project `x` and `y` into the panel using `y_projection_method`
        - calls upon `project_xy_...()` based on `y_projection_method`
        - generally `y_projection_method="theta"` is the preferred modus operandi

        Parameters
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

        Returns
            - `x_proj`
                - `np.ndarray`
                - `x` after projection
            - `y_proj`
                - `np.ndarray`
                - `y` after projection
        """

        if y_projection_method == "theta":
            x_proj, y_proj = self.project_xy_theta(x, y)
        elif y_projection_method == "y":
            x_proj, y_proj = self.project_xy_y(x, y)
        else:
            raise ValueError(f"`y_projection_method` has to be one of `'theta'`, `'y'` but got {y_projection_method}")

        return x_proj, y_proj
    

    #plotting methods
    def plot(self,
        x:np.ndarray, y:np.ndarray,
        seriestype:Literal["line","scatter"]="line",
        **kwargs,
        ):
        """attaches a dataseries to plot to the panel

        - method to add a series to the panel for plotting

        Parameters
            - `x`
                - `np.ndarray`
                - x-values of the series
                - has to have same length as `y`
            - `y`
                - `np.ndarray`
                - y-values of the series
                - has to have same length as `x`
            - `seriestpye`
                - `Literal["line","scatter"]`, optional
                - which style to use for plotting the series
                    `"line"` -- line plot
                    `"scatter"` -- scatter plot
                - the default is `"line"`
            -`**kwargs`
                - kwargs to pass to `ax.plot()`
                    
        Raises

        Returns
        """

        #apply axis limits
        x_cut, y_cut, kwargs = self.apply_axis_limits(x, y, **kwargs)

        #project x and y
        x_proj, y_proj = self.project_xy(x_cut, y_cut, self.y_projection_method)

        self.dataseries.append(dict(
            x=x_proj,
            y=y_proj,
            x_cut=x_cut,
            y_cut=y_cut,
            seriestype=seriestype,
            kwargs=kwargs,
        ))
        return

