
#%%imports
import numpy as np
from typing import Tuple

#%%definitions
def carth2polar(
    x:float, y:float    
    ) -> Tuple[float, float]:
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + np.pi

    return r, theta
def polar2carth(
    r:float, theta:float
    ) -> Tuple[float, float]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
def minmaxscale(
    x:np.ndarray,
    xmin:float, xmax:float,
    xmin_ref:float=None, xmax_ref:float=None,
    ) -> np.ndarray:
    if xmin_ref is None: xmin_ref = np.nanmin(x)
    if xmax_ref is None: xmax_ref = np.nanmax(x)

    x_scaled = (x - xmin_ref) / (xmax_ref - xmin_ref)
    x_scaled = x_scaled * (xmax - xmin) + xmin
    
    return x_scaled

def correct_labelrotation(theta:float) -> float:
    
    if np.sin(np.radians(theta)) < 0:   #use `sin` to also account for negative values
        return theta + 180
    else:
        return theta
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