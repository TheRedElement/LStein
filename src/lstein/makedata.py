"""module to create artificial data

- functions to create artificial data
- used for demonstrations

Classes

Functions
    - `gaussian_pdf()`  -- gaussian probability density function
    - `lc_sim()`        -- rudimentary non-physical lightcurve simulation
    - `sin_sim()`       -- simulation of sinusoidal signal
    - `simulate()`      -- interface to simulate data for LStein demos

Other Objects

"""

#%%imports
import numpy as np
from typing import Dict, Literal, Tuple

#%%definitions
def gaussian_pdf(x:float, mu:float, sigma:float) -> float:
    """evaluates gaussian normal distribution at `x`

    - function defining a gaussian normal distribution

    Parameters
        - `x`
            - `float`
            - x-value to evaluate the gaussian at
        - `mu`
            - `float`
            - mean of the gaussian
        - `sigma`
            - `float`
            - standard deviation of the gaussian

    Raises

    Returns
        - `y`
            - `float`
            - result when evaluating the gaussian with mean `mu` and standard deviation `sigma` at `x`
    
    Dependencies
        - `numpy`
        - `typing`
    """    
    y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))
    return y

def lc_sim(
    t:np.ndarray,
    t_peak:float, f_peak:float,
    stretch0:float, stretch1:float, stretch2:float,
    noiselevel:float=0.0,
    ) -> np.ndarray:
    """returns rudimentary (nonphysical) lightcurve

    - function to define a very simplistic phenomenological LC

    Parameters
        - `t`
            - `np.ndarray`
            - time i.e., x-values
        - `t_peak`
            - `float`
            - time of maximum flux
        - `f_peak`
            - `float`
            - maximum flux
        - `stretch0`
            - `float`
            - width of the entire peak
            - width between the two gaussians used to model in- and decreasing phase
        - `stretch1`
            - `float`
            - width of the gaussian used to model the increasing phase
        - `stretch2`
            - `float`
            - width of the gaussian used to model the decreasing phase
        - `noiselevel`
            - `float`, optional
            - how much noise to add to the generated dataseries
            - the default is `0.0`

    Raises

    Returns
        - `f`
            - `np.ndarray`
            - simulated flux values of the LC

    Dependencies
        - `numpy`
        - `typing`
    """
    f = (gaussian_pdf(t, t_peak - stretch0/2, stretch1) + gaussian_pdf(t, t_peak + stretch0/2, stretch2))
    f = f_peak * f / np.max(f) + noiselevel * np.random.randn(*t.shape)
    return f

def sin_sim(
    t:np.ndarray,
    f_minmax:float,
    p:float, offset:float=0.0,
    noiselevel:float=0.0
    ) -> float:
    """returns sin with period `p` and some `offset`

    - function to evaluate a sin with period `p` and `offset`

    Parameters
        - `t`
            - `np.ndarray`
            - time i.e., x-values
        - `f_minmax`
            - `float`
            - amplitude of minimum and maximum of the generated curve
        - `p`
            - `float`
            - period of the sine wave
        - `offset`
            - `float`, optional
            - offset of the sine wave
            - the default is `0.0`
        - `noiselevel`
            - `float`, optional
            - how much noise to add to the generated dataseries
            - the default is `0.0`

    Raises

    Returns
        - `f`
            - `np.ndarray`
            - sine wave with period `p` and offset `offset`

    Dependencies
        - `numpy`
        - `typing`
    """
    f = f_minmax * np.sin(t * 2*np.pi/p + offset) + noiselevel * np.random.randn(*t.shape)
    return f

def simulate(
    nobjects:int=6,
    opt:Literal["lc","sin"]="lc",
    theta:np.ndarray=None,
    ) -> Tuple[Dict,Dict]:
    """simulates `nobjects` objects using `opt`

    - function to simulate `nobjects` objects using the method specified in `opt`

    Parameters
        - `nobjects`
            - `int`, optional
            - how many objects to generate
        - `opt`
            - `Literal["lc","sin"]`, optional
            - method to use for generating the data
            - `"lc"` uses `lc_sim()`
            - `"sin"` uses `sin_sim()`
        - `theta`
            - `np.ndarrray`, optional
            - theta-values to use for generating the curves
            - setting `theta` overrides `nobjects`
            - the default is `None`
                - will be randomly generated

    Raises

    Returns
        - `raw`
            - `Dict`
            - contains simulated raw data
                - noisy
        - `pro`
            - `Dict`
            - contains simulated processed data
                - no noise


    Dependencies
        - `numpy`
        - `typing`
    """
    res = 500
    if theta is None:
        x = np.sort(np.random.choice(np.linspace(-50,100,res), size=(nobjects,res)), axis=1)
        theta_options = np.arange(10, 40, 0.2)
        theta = np.sort(np.random.choice(theta_options, size=nobjects, replace=False))
    else:
        nobjects = len(theta)
        x = np.sort(np.random.choice(np.linspace(-50,100,res), size=(nobjects,res)), axis=1)
    
    if opt == "lc":
        t_peak = np.linspace(0,40,nobjects) * 0
        y           = np.array([*map(lambda i: lc_sim(x[i], t_peak=t_peak[i], f_peak=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=1.0), range(nobjects))])
        y_nonoise   = np.array([*map(lambda i: lc_sim(x[i], t_peak=t_peak[i], f_peak=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=0.0), range(nobjects))])
    else:
        # theta *= 10
        y           = np.array([*map(lambda i: sin_sim(x[i], f_minmax=1, p=theta[i], offset=0.0, noiselevel=0.3), range(nobjects))])
        y_nonoise   = np.array([*map(lambda i: sin_sim(x[i], f_minmax=1, p=theta[i], offset=0.0, noiselevel=0.0), range(nobjects))])

    raw = dict(
            period=np.repeat(theta.reshape(-1,1), x.shape[1], axis=1).flatten(),
            time=x.flatten(),
            amplitude=y.flatten(),
            amplitude_e=np.nan,
            processing="raw",
        )
    pro = dict(
            period=np.repeat(theta.reshape(-1,1), x.shape[1], axis=1).flatten(),
            time=x.flatten(),
            amplitude=y_nonoise.flatten(),
            amplitude_e=np.nan,
            processing="nonoise",
        )
    
    return raw, pro
