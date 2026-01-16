"""module defining basic function for creating plots of the paper

- base-functions used to create plots for the traditional methods in the paper
- not relevant for user => no details in docstrings

Classes

Functions
    - `make_mesh()`                     -- creates components for plotting with `np.meshgrid()`
    - `plot_scatter_onepanel()`         -- scatter of different series in one panel
    - `plot_scatter_onepanel_offset()`  -- scatter of different series in one panel (series have offset)
    - `plot_scatter_multipanel()`       -- scatter of different series in one panel per series
    - `plot_scatter_multipanel_group()` -- scatter of different series (multiple panels, similar series grouped in a single panel)
    - `plot_heatmap()`                  -- heatmap of a set of series
    - `plot_3dsurface()`                -- 3d surface of a set of series

Other Objects

"""

#%%imports
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


#%%definitions
def make_mesh(
    th:np.ndarray, x:List[np.ndarray], y:List[np.ndarray],
    res=100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns components for plotting with `np.meshgrid`

    - function to create components for plotting with `np.meshgrid()` (`tt`, `xx`, `yy`)
    - will
        - generate `xx`: regularly spaced grid-values spanning from global min(x) to global max(x)
        - generate `tt`: same as `th` except spanning the grid
        - interpolate `y` to match the grid-shape spanned by `xx` and `tt` (fills with `np.nan` where out of interpolation range)
    """
    xmin = np.min([x_.min() for x_ in x])
    xmax = np.max([x_.max() for x_ in x])
    ymin = np.min([y_.min() for y_ in y])
    ymax = np.max([y_.max() for y_ in y])

    x_int = np.linspace(xmin, xmax, res)
    y_int = np.empty((len(th), res))
    th_int = th.copy()
    for idx, (thi, xi, yi) in enumerate(zip(th, x, y)):
        y_int[idx] = np.interp(x_int, xi, yi, right=np.nan, left=np.nan)
    
    #create meshgrid
    xx, tt = np.meshgrid(x_int, th_int)
    yy = y_int

    return tt, xx, yy 

def plot_scatter_onepanel(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    ax=None,
    ):
    """plots a scatter with all passbands in the same panel
    """    
    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=ylab)

    for i in range(len(theta_raw)):
        markers, caps, bars = ax.errorbar(x_raw[i], y_raw[i], yerr=y_raw_e[i], c=colors[i], ls="", marker="o", label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
        for bar in bars: bar.set_alpha(0.1)
    for i in range(len(theta_pro)):
        ax.plot(x_pro[i], y_pro[i], c="w", lw=3)
        ax.plot(x_pro[i], y_pro[i], c=colors[i])

    return ax

def plot_scatter_onepanel_offset(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors, offsets,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    ax=None,
    ):
    """plots a scatter with all passbands in the same panel
        
        - function to plot a scatter with all passbands in the same panel
        - passbands get offset
    """    
    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=ylab.replace("[]", " + Offset []"))

    offsets = np.cumsum(offsets)
    for i in range(len(theta_raw)):
        markers, caps, bars = ax.errorbar(x_raw[i], y_raw[i]+i*offsets[i], yerr=y_raw_e[i], c=colors[i], ls="", marker="o", label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
        for bar in bars: bar.set_alpha(0.1)
    for i in range(len(theta_pro)):
        ax.plot(x_pro[i], y_pro[i]+i*offsets[i], c="w", lw=3)
        ax.plot(x_pro[i], y_pro[i]+i*offsets[i], c=colors[i])

    return ax

def plot_scatter_multipanel(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    axs=None
    ):
    """plots a scatter with each passband in its own panel
    """
    if axs is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ncols_theta = 3
        nrows_theta = int(np.ceil(len(theta_raw)/ncols_theta))
        nrows = int(nrows_theta)
        ncols = int(ncols_theta)
        axs = [fig.add_subplot(nrows, ncols, i+1, xlabel=xlab, ylabel=ylab) for i in range(len(theta_raw))]

    for i in range(len(theta_raw)):
        markers, caps, bars = axs[i].errorbar(x_raw[i], y_raw[i],yerr=y_raw_e[i], c=colors[i], ls="", marker="o", label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
        for bar in bars: bar.set_alpha(0.1)
    for i in range(len(theta_pro)):
        axs[i].plot(x_pro[i], y_pro[i], c="w", lw=3)
        axs[i].plot(x_pro[i], y_pro[i], c=colors[i])
        
    return axs

def plot_scatter_multipanel_group(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    axs=None
    ):
    """plots a scatter with similar passbands grouped in same panel
    """
    if axs is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        nrows = 1
        ncols = 2
        axs = [fig.add_subplot(nrows, ncols, i+1, xlabel=xlab, ylabel=ylab) for i in range(ncols*nrows)]

    for i in range(0,3):    #redder passbands
        markers, caps, bars = axs[0].errorbar(x_raw[i], y_raw[i],yerr=y_raw_e[i], c=colors[i], ls="", marker="o", label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
        for bar in bars: bar.set_alpha(0.1)
    for i in range(0,min(3,len(theta_pro))):    #redder passbands
        axs[0].plot(x_pro[i], y_pro[i], c="w", lw=3)
        axs[0].plot(x_pro[i], y_pro[i], c=colors[i])
    for i in range(3,len(theta_raw)):    #bluer passbands
        markers, caps, bars = axs[1].errorbar(x_raw[i], y_raw[i],yerr=y_raw_e[i], c=colors[i], ls="", marker="o", label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
        for bar in bars: bar.set_alpha(0.1)
    for i in range(min(3,len(theta_pro)),len(theta_pro)):    #bluer passbands
        axs[1].plot(x_pro[i], y_pro[i], c="w", lw=3)
        axs[1].plot(x_pro[i], y_pro[i], c=colors[i])
        
    return axs

def plot_heatmap(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    ax=None,
    cmap=None, vmin=None, vmax=None,
    ):
    """plots series as a heatmap
    """

    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=thetalab)

    tt, xx, yy = make_mesh(theta_raw, x_raw, y_raw, res=100) 
    # tt, xx, yy = make_mesh(theta_pro, x_pro, y_pro, res=100) 
    vmin = np.nanmin(yy) if vmin is None else vmin
    vmax = np.nanmax(yy) if vmax is None else vmax

    mesh = ax.pcolormesh(xx, tt, yy, vmin=vmin, vmax=vmax, cmap=cmap)
    # for i in range(len(th_loc)):
    #     # ax.axhline(th_loc[i])
    #     ax.plot(x_loc, y_loc[i])
    # for i in range(len(theta_raw)):
    #     ax.scatter(x_raw[i], np.ones_like(x_raw[i])*theta_raw[i], c=y_raw[i], vmin=ymin, vmax=ymax, cmap=colors, ec="w", label="Raw Data"*(i==0))
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(ylab)

    return ax

def plot_3dsurface(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    ax=None,
    cmap=None,
    ):
    """plots data as 3d surface
    """
    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=thetalab, zlabel=ylab, projection="3d")

    tt, xx, yy = make_mesh(theta_raw, x_raw, y_raw, res=100) 
    # tt, xx, yy = make_mesh(theta_pro, x_pro, y_pro, res=100) 
    vmin = np.nanmin(yy)
    vmax = np.nanmax(yy)

    mesh = ax.plot_surface(xx, tt, yy, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9, zorder=0, linewidth=0)
    # for i in range(len(theta_raw)):
    #     ax.scatter(x_raw[i], np.ones_like(x_raw[i])*theta_raw[i], y_raw[i], c=y_raw[i],
    #         vmin=ymin, vmax=ymax,
    #         ec="w", label="Raw Data"*(i==0),
    #         zorder=4, depthshade=False)
    # cbar = fig.colorbar(mesh, ax=ax)
    # cbar.set_label(ylab)

    return ax
