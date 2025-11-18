#%%imports
import glob
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import re
from typing import Literal

from lstein import lstein, utils as lsu, makedata as md, paper_plots as pp

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.transparent"] = True

#%%constants
SURVEY_MAPPING:dict = {"elasticc":"ELAsTiCC", "des":"DES"}
OTYPE_MAPPING:dict = {"snia":"SN Ia", "snii":"SN II", "snibc":"SN Ib/c", "tde":"TDE"}
SAVE:bool = True

#%%definitions
def get_data(fidx:int):
    """
        - function to load some data
    """

    xmin2zero = False   #whether to shift xvalues to start at 0

    #passbands
    df_pb = pl.read_csv("../data/passband_specs.csv")
    passbands = list(df_pb["name"])
    pb_mappings = dict(zip(df_pb["wavelength"], df_pb.select(pl.exclude("wavelength")).to_numpy()))

    #LCs
    fnames = sorted(glob.glob("../data/*_*.csv"))
    fnames = np.append(fnames, ["../data/lc_simulated.py", "../data/sin_simulated.py"])
    # print(fnames)
    fname = fnames[fidx]

    #deal with on-the-fly data generation (pseudo filenames)
    if fname == "../data/lc_simulated.py":
        raw, pro = md.simulate(5, opt="lc")
        df = pl.concat([pl.from_dict(raw), pl.from_dict(pro)])
        for t in df["period"].unique(): pb_mappings[t] = [np.round(t, 3)]
        legend = False
        thetalab = "Maximum Amplitude"
        xlab = "Time [d]"
        ylab = "Amplitude []"
    elif fname == "../data/sin_simulated.py":
        raw, pro = md.simulate(5, opt="sin")
        df = pl.concat([pl.from_dict(raw), pl.from_dict(pro)])
        for t in df["period"].unique(): pb_mappings[t] = [np.round(t, 3)]
        legend = False
        thetalab = "Period [s]"
        xlab = "Time [s]"
        ylab = "Amplitude []"
    else:
        df = pl.read_csv(fname, comment_prefix="#")
        df = df.sort(pl.col(df.columns[1]))
        legend = True
        thetalab = "Wavelength [nm]"
        xlab = "MJD-min(MJD) [d]" if "mjd" in df.columns else "Period [d]"
        ylab = "m [mag]" if "mag" in df.columns else "Fluxcal []"

    # df = df.drop_nans()

    #sigma clipping
    df = df.filter(
        pl.col(df.columns[2]).median()-3*pl.col(df.columns[2]).std() <= pl.col(df.columns[2]),
        pl.col(df.columns[2]) <= pl.col(df.columns[2]).median()+3*pl.col(df.columns[2]).std(),
    )

    parts = re.split(r"[/\_\.]", fname)
    survey = parts[-2]
    otype = parts[-3]

    df_raw = df.filter(pl.col("processing")=="raw")
    df_pro = df.filter(pl.col("processing")!="raw")
    theta_raw = np.sort(np.unique(df_raw[:,0]))
    df_raw_p = df_raw.partition_by(df_raw.columns[0], maintain_order=True)
    x_raw = [df[:,1].to_numpy().astype(np.float64) for df in df_raw_p]
    x_raw = [xi - xmin2zero*np.nanmin(xi) for xi in x_raw]
    y_raw = [df[:,2].to_numpy().astype(np.float64) for df in df_raw_p]
    y_raw_e = [df[:,3].to_numpy().astype(np.float64) for df in df_raw_p]
    theta_pro = np.sort(np.unique(df_pro[:,0]))
    df_pro_p = df_pro.partition_by(df_pro.columns[0], maintain_order=True)
    x_pro = [df[:,1].to_numpy().astype(np.float64) for df in df_pro_p]
    x_pro = [xi - xmin2zero*np.nanmin(xi) for xi in x_pro]
    y_pro = [df[:,2].to_numpy().astype(np.float64) for df in df_pro_p]
    y_pro_e = [df[:,3].to_numpy().astype(np.float64) for df in df_raw_p]
    
    #artificial, large x-values
    # x_raw = [np.linspace(10000,10010,len(xi)) for xi in x_raw]
    # x_pro = [np.linspace(10000,10010,len(xi)) for xi in x_pro]

    return (
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        legend, thetalab, xlab, ylab, fname,
        pb_mappings, otype, survey,
    )

def get_stats(theta_raw, x_raw, y_raw, fname):
    """
        - function to get stats for plot specifications
    """
    unique_thetas = np.unique(theta_raw)
    thetaticks = np.round(np.linspace(np.floor(np.min(theta_raw)), np.ceil(np.max(theta_raw)), 4),0).astype(int)
    xticks = np.round(np.linspace(np.floor(np.min(np.concat(x_raw))), np.ceil(np.max(np.concat(x_raw))), 4), decimals=0).astype(int)
    yticks = np.round(np.linspace(np.floor(np.min(np.concat(y_raw))), np.ceil(np.max(np.concat(y_raw))), 4), decimals=0).astype(int)
    # yticks = np.sort(np.append(yticks, [-10, 80]))
    panelsize = np.pi/10
    vmin = 300 if ".py" not in fname else None
    colors = lsu.get_colors(theta_raw, cmap="nipy_spectral", vmin=vmin)
    return (
        unique_thetas,
        thetaticks, xticks, yticks,
        panelsize,
        colors,
    )

def plot_lstein(
    ):
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(-np.pi/2,2*np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,2*np.pi/2-panelsize/2),
        xlimdeadzone=0.2,
        thetalabel=thetalab, xlabel=xlab, ylabel=ylab,
        thetaarrowpos_th=None, ylabpos_th=np.min(theta_raw),
        thetatickkwargs=None, thetaticklabelkwargs=dict(pad=0.3), thetalabelkwargs=dict(rotation=40, textcoords="offset fontsize", xytext=(-1,-1)),
        xtickkwargs=None, xticklabelkwargs=dict(xytext=(-2,0)), xlabelkwargs=None,
    )

    #adding all the series (will initialize panels for you)
    LSC.plot(theta_raw, x_raw, y_raw, seriestype="scatter", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=dict(s=3, alpha=0.5))
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="plot", series_kwargs=dict(lw=3, c="w"))
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="plot", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=[dict(lw=_/theta_pro.max(), ls="-") for _ in theta_pro])

    fig = lstein.draw(LSC, figsize=(5,9))
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/lstein_lc.png")
    return

def plot_scatter_onepanel(
    ):
    ax = pp.plot_scatter_onepanel(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        colors,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,        
    )
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_onepanel.png")
    return

def plot_scatter_onepanel_offset(
    ):
    
    offset = [10]*len(theta_raw)
    offset = theta_raw / 100
    ax = pp.plot_scatter_onepanel_offset(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        colors, offset,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,        
    )
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_onepanel_offset.png")
    return

def plot_scatter_multipanel(
    ):

    axs = pp.plot_scatter_multipanel(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        colors,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,        
    )
    for ax in axs: ax.legend()
    fig = axs[0].get_figure()
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_multipanel.png")
    return

def plot_scatter_multipanel_group(
    ):

    axs = pp.plot_scatter_multipanel_group(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        colors,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,        
    )
    for ax in axs: ax.legend()
    fig = axs[0].get_figure()
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_multipanel_group.png")
    return

def plot_heatmap(
    ):
    ax = pp.plot_heatmap(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        colors,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,
    )
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/heatmap.png")
    return

def plot_3dsurface(
    ):
    ax = pp.plot_3dsurface(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        colors,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,
    )
    fig = ax.get_figure()
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/surface3d.png")
    return ax

def plot_projection_methods(
    context:Literal["theta","y"]="theta"
    ):
    if context == "y":          fidx = 0    #displays projection effects in y-direction with `y_projection_method="y"`
    elif context == "theta":    fidx = -1   #displays projection effects in x-projection with `y_projection_method="theta"`
    else:                       fidx = 8    #displays going haywire for large x-values

    #get data
    theta_raw, x_raw, y_raw, y_raw_e, \
    theta_pro, x_pro, y_pro, y_pro_e, \
    legend, thetalab, xlab, ylab, fname, \
        pb_mappings, otype, survey = get_data(fidx)

    unique_thetas, \
    thetaticks, xticks, yticks, \
    panelsize, \
    colors = get_stats(theta_raw, x_raw, y_raw, fname)

    #init canvas (similar to `fig = plt.figure()`)
    LSC1 = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(0,np.pi/2), thetaplotlims=(0+panelsize/2,np.pi/2-panelsize/2),
    )
    LSC2 = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(0,np.pi/2), thetaplotlims=(0+panelsize/2,np.pi/2-panelsize/2),
    )

    #plotting all the series (similar to `plt.plot()`)
    LSC1.plot(theta_raw[::1], x_raw[::1], y_raw[::1], seriestype="scatter", panel_kwargs=dict(y_projection_method="theta"),)
    LSC1.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="plot", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=dict(lw=3, c="w"))
    LSC1.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="plot", panel_kwargs=dict(y_projection_method="theta"),)
    LSC2.plot(theta_raw[::1], x_raw[::1], y_raw[::1], seriestype="scatter", panel_kwargs=dict(y_projection_method="y"),)
    LSC2.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="plot", panel_kwargs=dict(y_projection_method="y"), series_kwargs=dict(lw=3, c="w"))
    LSC2.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="plot", panel_kwargs=dict(y_projection_method="y"),)

    #plotting
    fig = plt.figure(figsize=(9,9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("`y_projection_method=\"theta\"`", y=1.1)
    ax2.set_title("`y_projection_method=\"y\"`", y=1.1)

    lstein.LSteinMPL(LSC1).show(ax1)
    lstein.LSteinMPL(LSC2).show(ax2)
    fig.tight_layout()
    
    if SAVE: fig.savefig(f"../report/gfx/projectionmethods_{context}.png")
    return fig
#%%main
def main():
    #declare as global so no arguments have to be passed to nested functions
    global theta_raw, x_raw, y_raw, y_raw_e
    global theta_pro, x_pro, y_pro, y_pro_e
    global legend, thetalab, xlab, ylab, fname
    global pb_mappings, otype, survey
    
    global unique_thetas
    global thetaticks, xticks, yticks
    global panelsize
    global colors

    #load data
    theta_raw, x_raw, y_raw, y_raw_e, \
    theta_pro, x_pro, y_pro, y_pro_e, \
    legend, thetalab, xlab, ylab, fname, \
        pb_mappings, otype, survey = get_data(7)    

    unique_thetas, \
    thetaticks, xticks, yticks, \
    panelsize, \
    colors = get_stats(theta_raw, x_raw, y_raw, fname)

    #plots
    plot_lstein()
    plot_scatter_onepanel()
    plot_scatter_onepanel_offset()
    plot_scatter_multipanel()
    plot_scatter_multipanel_group()
    plot_heatmap()
    plot_3dsurface()
    plot_projection_methods(context="theta")
    plot_projection_methods(context="y")

    plt.show()
    return

if __name__ == "__main__":
    main()