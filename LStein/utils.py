
#%%imports
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from typing import Tuple, Union

#%%definitions
def carth2polar(
    x:float, y:float    
    ) -> Tuple[float, float]:
    """
        - function to convert carthesian coordinates into polar coordinates

        Parameters
        ----------
            - `x`
                - `float`
                - carthesian x-coordinate
            - `y`
                - `float`
                - carthesian y-coordinate

        Raises
        ------

        Returns
        -------
            - `r`
                - `float`
                - radius
            - `theta`
                - `float`
                - azimuthal angle

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------
    """
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + np.pi

    return r, theta

def polar2carth(
    r:float, theta:float
    ) -> Tuple[float, float]:
    """
        - function to convert polar coordinates into carthesian coordinates

        Parameters
        ----------
            - `r`
                - `float`
                - radius
            - `theta`
                - `float`
                - azimuthal angle

        Raises
        ------

        Returns
        -------
            - `x`
                - `float`
                - carthesian x-coordinate
            - `y`
                - `float`
                - carthesian y-coordinate

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------
    """    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def minmaxscale(
    x:np.ndarray,
    xmin:float, xmax:float,
    xmin_ref:float=None, xmax_ref:float=None,
    ) -> np.ndarray:
    """
        - function to apply min-max-scaling onto `x`
        - will rescale `x` to the interval `xmin`, `xmax`
        - considers `xmin_ref`, `xmax_ref` as reference boundaries of the original dataset that `x` is a part of
        - similar to [`sklearn.processing.MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
        
        Parameters
        ----------
            - `x`
                - `np.ndarray`
                - feature to be transformed
            - `xmin`
                - `float` 
                - lower bound of the target range
            - `xmax`
                - `float` 
                - upper bound of the target range
            - `xmin_ref`
                - `float`, optional
                - lower bound of the original range
                - the default is `None`
                    - will consider `x` to be representing the original range
                    - will be set to `np.nanmin(x)`
            - `xmax_ref`
                - `float`, optional
                - upper bound of the original range
                - the default is `None`
                    - will consider `x` to be representing the original range
                    - will be set to `np.nanmax(x)`

        Raises
        ------

        Returns
        -------
            - `x_scaled`
                - `np.ndarray`
                - same shape as `x`
                - `x` after the transformation

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------

    """
    if xmin_ref is None: xmin_ref = np.nanmin(x)
    if xmax_ref is None: xmax_ref = np.nanmax(x)

    x_scaled = (x - xmin_ref) / (xmax_ref - xmin_ref)
    x_scaled = x_scaled * (xmax - xmin) + xmin
    
    return x_scaled

def correct_labelrotation(theta:float) -> float:
    """
        - function to make labels always face the right way up
        - rotates any text that is upside down by 180 degrees
            - upside down meaning it has a rotation in the interval $(90,270)$

        Parameters
        ----------
            - `theta`
                - `float`
                - rotation angle in degrees
        
        Raises
        ------

        Returns
        -------
            - `theta`
                - `float`
                - `theta` after correcting upside down angles

        Dependencies
        ------------
            - `numpy`
            - `typing`

        Comments
        --------
    """
    
    if np.sin(np.radians(theta)) < 0:   #use `sin` to also account for negative values
        return theta + 180
    else:
        return theta
    
def get_colors(
    x:np.ndarray,
    cmap:Union[str,mcolors.Colormap]=None, norm=mcolors.Normalize,
    **kwargs
    ) -> np.ndarray:
    """
        - function to generate an array of colors mapping `x` onto `cmap`

        Parameters
        ----------
            - `x`
                - `np.ndarray`
                - data-series to be mapped onto `cmap`
            - `cmap`
                - `str`, `mcolors.Colormap`, optional
                - colormap to use
                - the default is `None`
                    - will use `plt.rcParams["image.cmap"]`
            - `norm`
                - `mcolors Norm`, optional
                - some instance of a `matplotlib.colors` Norm
                - normalization to use when mapping `x` to `cmap`
                - the default is `mcolors.Normalize`
            - `**kwargs`
                - kwargs to to pass to `norm()`


        Raies
        -----

        Returns
        -------
            - `colors`
                - `np.ndarray`
                - has shape `(x.shape[0],3)`
                    - one rgb-tuple for each value in `x`

        Dependencies
        ------------
            - `matplotlib`
            - `typing`
        
        Comments
        --------
    """
    
    #default parameters
    cmap = plt.rcParams["image.cmap"] if cmap is None else cmap
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    
    #get norm
    norm = norm(**kwargs)
    
    colors = cmap(norm(x))

    return colors
# %%

# theta = np.linspace(0,2*np.pi,10)
# r = 1


# x, y = polar2carth(r, theta)
# plt.plot(x,y)
# r, theta = carth2polar(x, y)
# x, y = polar2carth(r, theta)

# import matplotlib.pyplot as plt

# for i in range(len(x)): print(x[i], y[i], r[i], theta[i])
# plt.plot(x, y)
# # plt.plot(x,y)
# # plt.ylim(0,None)