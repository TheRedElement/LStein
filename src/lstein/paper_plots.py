
#%%imports
import matplotlib.pyplot as plt
import numpy as np

from lstein import lstein, utils as lsu, makedata as md


#%%definitions
def plot_scatter_onepanel(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    ax=None,
    ):
    """
        - function to plot a scatter with all passbands in the same panel
    """    
    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=ylab)

    for i in range(len(theta_raw)):
        markers, caps, bars = ax.errorbar(x_raw[i], y_raw[i], yerr=y_raw_e[i], c=colors[i], ls="", marker="o")
        for bar in bars: bar.set_alpha(0.1)
    for i in range(len(theta_pro)):
        ax.plot(x_pro[i], y_pro[i], c="w", lw=3)
        ax.plot(x_pro[i], y_pro[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")

    return ax

def plot_scatter_onepanel_offset(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors, offsets,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    ax=None,
    ):
    """
        - function to plot a scatter with all passbands in the same panel
        - passbands get offset
    """    
    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=ylab.replace("[]", " + Offset []"))

    offsets = np.cumsum(offsets)
    for i in range(len(theta_raw)):
        markers, caps, bars = ax.errorbar(x_raw[i], y_raw[i]+i*offsets[i], yerr=y_raw_e[i], c=colors[i], ls="", marker="o")
        for bar in bars: bar.set_alpha(0.1)
    for i in range(len(theta_pro)):
        ax.plot(x_pro[i], y_pro[i]+i*offsets[i], c="w", lw=3)
        ax.plot(x_pro[i], y_pro[i]+i*offsets[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")

    return ax

def plot_scatter_multipanel(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    axs=None
    ):
    """
        - function to plot a scatter with each passband in one panel
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
        markers, caps, bars = axs[i].errorbar(x_raw[i], y_raw[i],yerr=y_raw_e[i], c=colors[i], ls="", marker="o")
        for bar in bars: bar.set_alpha(0.1)
        axs[i].plot(x_pro[i], y_pro[i], c="w", lw=3)
        axs[i].plot(x_pro[i], y_pro[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
        
    return axs
def plot_scatter_multipanel_group(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    axs=None
    ):
    """
        - function to plot a scatter with each passband in one panel
    """
    if axs is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        nrows = 1
        ncols = 2
        axs = [fig.add_subplot(nrows, ncols, i+1, xlabel=xlab, ylabel=ylab) for i in range(ncols*nrows)]

    for i in range(0,3):    #redder passbands
        markers, caps, bars = axs[0].errorbar(x_raw[i], y_raw[i],yerr=y_raw_e[i], c=colors[i], ls="", marker="o")
        for bar in bars: bar.set_alpha(0.1)
        axs[0].plot(x_pro[i], y_pro[i], c="w", lw=3)
        axs[0].plot(x_pro[i], y_pro[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
    for i in range(3,len(theta_raw)):    #bluer passbands
        markers, caps, bars = axs[1].errorbar(x_raw[i], y_raw[i],yerr=y_raw_e[i], c=colors[i], ls="", marker="o")
        for bar in bars: bar.set_alpha(0.1)
        axs[1].plot(x_pro[i], y_pro[i], c="w", lw=3)
        axs[1].plot(x_pro[i], y_pro[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
        
    return axs

def plot_heatmap(
    theta_raw, x_raw, y_raw, y_raw_e,
    theta_pro, x_pro, y_pro, y_pro_e,
    colors,
    pb_mappings, otype, survey,
    thetalab, xlab, ylab,
    ax=None,
    cmap=None,
    ):
    """
        - function to plot as a heatmap
    """

    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=thetalab)

    res = 150
    x_pro_loc = np.array(x_pro.copy())
    y_pro_loc = np.array(y_pro.copy())
    xmin = np.min(x_pro_loc)
    xmax = np.max(x_pro_loc)
    ymin = np.min(y_pro_loc)
    ymax = np.max(y_pro_loc)
    thmin = np.min(theta_raw)
    thmax = np.max(theta_raw)
    x  = np.linspace(xmin, xmax, res)
    y = [np.interp(x, x_pro_loc[i], y_pro_loc[i]) for i in range(len(theta_raw))]
    th = np.linspace(thmin, thmax, len(theta_raw))
    
    yy = np.array(y)

    xx, tt = np.meshgrid(x, th)

    mesh = ax.pcolormesh(xx, tt, yy, vmin=ymin, vmax=ymax, cmap=cmap)
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
    """
        - function to plot data as 3d surface
    """
    if ax is None:
        fig = plt.figure(figsize=(9,5))
        fig.suptitle(f"{otype} ({survey})")
        ax = fig.add_subplot(111, xlabel=xlab, ylabel=thetalab, zlabel=ylab, projection="3d")
    res = 150
    x_pro_loc = np.array(x_pro.copy())
    y_pro_loc = np.array(y_pro.copy())
    xmin = np.min(x_pro_loc)
    xmax = np.max(x_pro_loc)
    ymin = np.min(y_pro_loc)
    ymax = np.max(y_pro_loc)
    thmin = np.min(theta_raw)
    thmax = np.max(theta_raw)
    x  = np.linspace(xmin, xmax, res)
    y = [np.interp(x, x_pro_loc[i], y_pro_loc[i]) for i in range(len(theta_raw))]
    th = np.linspace(thmin, thmax, len(theta_raw))
    
    yy = np.array(y)

    xx, tt = np.meshgrid(x, th)

    mesh = ax.plot_surface(xx, tt, yy, cmap=cmap, vmin=ymin, vmax=ymax, alpha=0.9, zorder=0, linewidth=0)
    # for i in range(len(theta_raw)):
    #     ax.scatter(x_raw[i], np.ones_like(x_raw[i])*theta_raw[i], y_raw[i], c=y_raw[i],
    #         vmin=ymin, vmax=ymax,
    #         ec="w", label="Raw Data"*(i==0),
    #         zorder=4, depthshade=False)
    # cbar = fig.colorbar(mesh, ax=ax)
    # cbar.set_label(ylab)

    return ax
