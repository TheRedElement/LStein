#%%imports
import brian2
import brian2.numpy_ as np_
from brian2 import NeuronGroup, Network
from brian2 import TimedArray
from brian2 import StateMonitor, SpikeMonitor
from brian2 import Gohm, ms, mV, pA, pF, second
from cycler import cycler
import glob
import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
import re
from typing import Literal, Tuple
from LuStCodeSnippets_py.Styles import PlotStyles

from lstein import lstein, utils as lsu, makedata as md, paper_plots as pp
importlib.reload(pp)

#setup plotting style
_ = PlotStyles.tre_light()


color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
prop_cycle = (
    cycler(linestyle=["-"]*len(color_cycle)) +
    cycler(color=color_cycle)
)
plt.rcParams["font.size"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams["legend.framealpha"] = 0.8
plt.rcParams["legend.facecolor"] = "w"
plt.rcParams["axes.prop_cycle"] = prop_cycle        #constant linestyle
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.transparent"] = True
plt.rcParams["text.usetex"] = True

#seeds
np.random.seed(0)

#%%constants
SURVEY_MAPPING:dict = {"elasticc":"ELAsTiCC", "des":"DES"}
OTYPE_MAPPING:dict = {"snia":"SN Ia", "snii":"SN II", "snibc":"SN Ib/c", "tde":"TDE"}
CMAP:str = "plasma"
CMAP_BANDS:str = "nipy_spectral"
SAVE:bool = True
YLIM:Tuple[int,int] = (-5,None)

#%%definitions
def get_data(fidx:int):
    """
        - function to load some data
    """

    xmin2zero = True   #whether to shift xvalues to start at 0

    #passbands
    df_pb = pl.read_csv("../data/passband_specs.csv")
    passbands = list(df_pb["name"])
    pb_mappings = dict(zip(df_pb["wavelength"], df_pb.select(pl.exclude("wavelength")).to_numpy()))

    #LCs
    fnames = sorted(glob.glob("../data/*_*.csv"))
    fnames = np.append(fnames, ["../data/lc_simulated.py", "../data/sin_simulated.py"])
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
        raw, pro = md.simulate(5, opt="sin", theta=np.linspace(10, 36, 3))
        df = pl.concat([pl.from_dict(raw), pl.from_dict(pro)])
        for t in df["period"].unique(): pb_mappings[t] = [np.round(t, 3)]
        legend = False
        thetalab = "Period"
        xlab = "Time"
        ylab = "Amplitude"
    else:
        df = pl.read_csv(fname, comment_prefix="#")
        df = df.sort(pl.col(df.columns[1]))
        legend = True
        # thetalab = "Wavelength [nm]"
        # xlab = "MJD-min(MJD) [d]" if "mjd" in df.columns else "Period [d]"
        # ylab = "m [mag]" if "mag" in df.columns else "Flux [FLUXCAL]"
        thetalab = "Wavelength [nm]"
        xlab = "Time [d]" if "mjd" in df.columns else "Period [d]"
        ylab = "Flux [FLUXCAL]"

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
    y_pro_e = [df[:,3].to_numpy().astype(np.float64) for df in df_pro_p]
    
    # #avoid negative values
    # y_raw_mask = [(y_raw[idx]) for idx in range(len(y_raw))]
    # y_raw = [y_raw[idx][y_raw_mask[idx]] for idx in range(len(y_raw))]
    # x_raw = [x_raw[idx][y_raw_mask[idx]] for idx in range(len(x_raw))]
    # y_raw_e = [y_raw_e[idx][y_raw_mask[idx]] for idx in range(len(y_raw_e))]
    # y_pro_mask = [(y_pro[idx]>-1) for idx in range(len(y_pro))]
    # y_pro = [y_pro[idx][y_pro_mask[idx]] for idx in range(len(y_pro))]
    # x_pro = [x_pro[idx][y_pro_mask[idx]] for idx in range(len(x_pro))]
    # y_pro_e = [y_pro_e[idx][y_pro_mask[idx]] for idx in range(len(y_pro_e))]

    #artificial, large x-values
    # x_raw = [np.linspace(10000,10010,len(xi)) for xi in x_raw]
    # x_pro = [np.linspace(10000,10010,len(xi)) for xi in x_pro]

    return (
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        legend, thetalab, xlab, ylab, fname,
        pb_mappings, otype, survey,
    )

def get_stats(theta_raw, x_raw, y_raw, fname,
    nthticks=4, nxticks=4, nyticks=4,
    ):
    """
        - function to get stats for plot specifications
    """
    ymin = YLIM[0] if YLIM[0] is not None else np.floor(np.min(np.concat(y_raw)))
    ymax = YLIM[1] if YLIM[1] is not None else np.ceil(np.max(np.concat(y_raw)))
    
    unique_thetas = np.unique(theta_raw)
    thetaticks = np.round(np.linspace(np.floor(np.min(theta_raw)), np.ceil(np.max(theta_raw)), nthticks),0).astype(int)
    xticks = np.round(np.linspace(np.floor(np.min(np.concat(x_raw))), np.ceil(np.max(np.concat(x_raw))), nxticks), decimals=0).astype(int)
    yticks = np.round(np.linspace(ymin, ymax, nyticks), decimals=0).astype(int)
    # yticks = np.sort(np.append(yticks, [-10, 80]))
    panelsize = np.pi/10
    vmin = 300 if ".py" not in fname else None
    colors = lsu.get_colors(theta_raw, cmap=CMAP_BANDS, vmin=vmin)
    return (
        unique_thetas,
        thetaticks, xticks, yticks,
        panelsize,
        colors,
    )

def plot_lstein_snii():
    
    #load data
    theta_raw, x_raw, y_raw, y_raw_e, \
    theta_pro, x_pro, y_pro, y_pro_e, \
    legend, thetalab, xlab, ylab, fname, \
        pb_mappings, otype, survey = get_data(7)    

    unique_thetas, \
    thetaticks, xticks, yticks, \
    panelsize, \
    colors = get_stats(theta_raw, x_raw, y_raw, fname)


    thetaguidelims=(0*np.pi/2,2*np.pi/2)
    xticks = xticks[::-1]
    yticks = yticks[::-1]

    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        # thetaguidelims=(0*np.pi/2,2*np.pi/2), thetaplotlims=(0*np.pi/2+panelsize/2,2*np.pi/2-panelsize/2),
        thetaguidelims=thetaguidelims, thetaplotlims=(thetaguidelims[0]+panelsize/2,thetaguidelims[1]-panelsize/2),
        xlimdeadzone=0.3,
        panelsize=panelsize,
        thetalabel=thetalab, xlabel=xlab, ylabel=ylab,
        thetaarrowpos_th=None, ylabpos_th=np.min(theta_raw),
        thetatickkwargs=None, thetaticklabelkwargs=dict(pad=0.3), thetalabelkwargs=dict(rotation=0, textcoords="offset fontsize", xytext=(0,-0.7)),
        xtickkwargs=None, xticklabelkwargs=dict(xytext=(0,-1)), xlabelkwargs=dict(xytext=(-7.5,-2)),
        ylabelkwargs=dict(rotation=-83, textcoords="offset fontsize", xytext=(0,-3)),
    )

    #adding all the series (will initialize panels for you)
    LSC.plot(theta_raw, x_raw, y_raw, seriestype="scatter", panel_kwargs=dict(y_projection_method="theta", show_panelbounds=True), series_kwargs=[dict(s=20, alpha=0.5, c=colors[thidx]) for thidx in range(len(theta_raw))])
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", series_kwargs=dict(lw=4, c="w"))
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=[dict(lw=3, ls="-", c=colors[thidx]) for thidx in range(len(theta_raw))])

    fig = lstein.draw(LSC, figsize=(9,5))
    fig.tight_layout()

    if SAVE: fig.savefig(f"../report/gfx/lstein_{otype}_{survey}.pdf")
    return LSC

def plot_lstein_tde(
    ):
    
    #load data
    theta_raw, x_raw, y_raw, y_raw_e, \
    theta_pro, x_pro, y_pro, y_pro_e, \
    legend, thetalab, xlab, ylab, fname, \
        pb_mappings, otype, survey = get_data(21)    

    unique_thetas, \
    thetaticks, xticks, yticks, \
    panelsize, \
    colors = get_stats(theta_raw, x_raw, y_raw, fname, nyticks=3)
    
    thetaguidelims=(0.1*np.pi/4,2.5*np.pi/4)

    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=thetaguidelims, thetaplotlims=(thetaguidelims[0]+panelsize/2,thetaguidelims[1]-panelsize/2),
        xlimdeadzone=0.3,
        panelsize=np.pi/15,
        thetalabel=thetalab, xlabel=xlab, ylabel=ylab,
        thetaarrowpos_th=None, ylabpos_th=np.min(theta_raw),
        thetatickkwargs=None, thetaticklabelkwargs=dict(pad=0.3), thetalabelkwargs=dict(rotation=0, textcoords="offset fontsize", xytext=(-1.5,1)),
        xtickkwargs=None, xticklabelkwargs=dict(xytext=(0,-1)), xlabelkwargs=dict(rotation=3, xytext=(-7,-2.5)),
        ylabelkwargs=dict(rotation=-83, textcoords="offset fontsize", xytext=(0,-3)),
    )

    #adding all the series (will initialize panels for you)
    LSC.plot(theta_raw, x_raw, y_raw, seriestype="scatter", panel_kwargs=dict(y_projection_method="theta", show_panelbounds=True), series_kwargs=[dict(s=20, alpha=0.5, c=colors[thidx]) for thidx in range(len(theta_raw))])
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", series_kwargs=dict(lw=3, c="w"))
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=[dict(lw=3, ls="-", c=colors[thidx]) for thidx in range(len(theta_raw))])

    fig = lstein.draw(LSC, figsize=(7.5,7.5))
    fig.tight_layout()

    if SAVE: fig.savefig(f"../report/gfx/lstein_{otype}_{survey}.pdf")
    return LSC

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
    ax.set_ylim(YLIM)
    fig = ax.get_figure()
    fig.suptitle("")
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_onepanel.pdf")
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
    ax.set_ylim(YLIM)
    fig = ax.get_figure()
    fig.suptitle("")
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_onepanel_offset.pdf")
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
    for idx, ax in enumerate(axs):
        ax.legend(loc="upper left")
        ax.set_ylim(YLIM)
        # if idx%3 != 0: ax.set_ylabel("")
        ax.set_ylabel("")
        if idx < 3: ax.set_xlabel("")
    fig = axs[0].get_figure()
    fig.text(0, 0.5, ylab, rotation=90, va="center")
    fig.suptitle("")
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_multipanel.pdf", bbox_inches="tight")
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
    for ax in axs:
        ax.legend()
        ax.set_ylim(YLIM)
    axs[1].set_ylabel("")
    fig = axs[0].get_figure()
    fig.suptitle("")
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/scatter_multipanel_group.pdf")
    return

def plot_heatmap(
    ):
    ax = pp.plot_heatmap(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        CMAP,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,
        cmap=CMAP, vmin=-1, vmax=None,
    )
    # ax.legend()
    # ax.set_ylim(YLIM)
    fig = ax.get_figure()
    fig.suptitle("")
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/heatmap.pdf")
    return

def plot_3dsurface(
    ):

    fig, ax = plt.subplots(1,1, figsize=(9,9), subplot_kw=dict(projection="3d", xlabel=xlab, ylabel=thetalab, zlabel=ylab))
    ax = pp.plot_3dsurface(
        theta_raw, x_raw, y_raw, y_raw_e,
        theta_pro, x_pro, y_pro, y_pro_e,
        colors,
        pb_mappings, otype, survey,
        thetalab, xlab, ylab,
        ax=ax,
        cmap=CMAP,
    )
    # ax.set_ylim(YLIM)
    fig = ax.get_figure()
    ax.set_box_aspect(None, zoom=0.89)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15
    fig.suptitle("")
    fig.subplots_adjust(left=0.0)
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/surface3d.pdf", bbox_inches="tight")
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

    thetaguidelims = (0*np.pi/2,1*np.pi/2)
    if context=="theta":
        yticks = np.linspace(-1.5, 1.5, 3)
        xlimdeadzone = 0.3
        xlabelkwargs = None
        ylabelkwargs = dict(rotation=-80, textcoords="offset fontsize", xytext=(0,-1.5))
        thetalablekwargs = dict(rotation=0, textcoords="offset fontsize", xytext=(0.5,1))
        thetaticklabelkwargs = None
        xticklabelkwargs = None
    elif context=="y":
        yticks = np.round(np.linspace(np.floor(np.min(np.concat(y_raw))), np.ceil(np.max(np.concat(y_raw))), 4), decimals=0).astype(int)
        xlimdeadzone = 0.35
        xlabelkwargs = dict(xytext=(-3,-2))
        ylabelkwargs = dict(rotation=-80, textcoords="offset fontsize", xytext=(0,-2.7))
        thetalablekwargs = dict(rotation=-45, textcoords="offset fontsize", xytext=(1.5,0))
        xticklabelkwargs = dict(xytext=(0,-1))
        thetaticklabelkwargs = dict(rotation=45)

    #init canvas (similar to `fig = plt.figure()`)
    # panelsize = np.pi/12
    LSC1 = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=thetaguidelims, thetaplotlims=(thetaguidelims[0]+1.3*panelsize/2,thetaguidelims[1]-1.3*panelsize/2),
        xlimdeadzone=xlimdeadzone,
        thetalabel=thetalab, xlabel=xlab, #ylabel=ylab, 
        thetalabelkwargs=thetalablekwargs,
        xlabelkwargs=xlabelkwargs,
        ylabelkwargs=ylabelkwargs,
        xticklabelkwargs=xticklabelkwargs,
        thetaticklabelkwargs=thetaticklabelkwargs,
    )
    LSC2 = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=thetaguidelims, thetaplotlims=(thetaguidelims[0]+1.3*panelsize/2,thetaguidelims[1]-1.3*panelsize/2),
        xlimdeadzone=xlimdeadzone,
        thetalabel=thetalab, xlabel=xlab, ylabel=ylab, 
        thetalabelkwargs=thetalablekwargs,
        xlabelkwargs=xlabelkwargs,
        ylabelkwargs=ylabelkwargs,
        xticklabelkwargs=xticklabelkwargs,
        thetaticklabelkwargs=thetaticklabelkwargs,
    )

    #plotting all the series (similar to `plt.plot()`)
    LSC1.plot(theta_raw[::1], x_raw[::1], y_raw[::1], seriestype="scatter", panel_kwargs=dict(y_projection_method="theta", show_panelbounds=True), series_kwargs=[dict(s=5, alpha=0.5, c=colors[thidx]) for thidx in range(len(theta_raw))])
    LSC1.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="line", panel_kwargs=dict(y_projection_method="theta", show_panelbounds=True), series_kwargs=dict(lw=5, c="w"))
    LSC1.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="line", panel_kwargs=dict(y_projection_method="theta", show_panelbounds=True), series_kwargs=[dict(lw=3, ls="-", c=colors[thidx]) for thidx in range(len(theta_raw))])
    LSC2.plot(theta_raw[::1], x_raw[::1], y_raw[::1], seriestype="scatter", panel_kwargs=dict(y_projection_method="y", show_panelbounds=True), series_kwargs=[dict(s=5, alpha=0.5, c=colors[thidx]) for thidx in range(len(theta_raw))])
    LSC2.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="line", panel_kwargs=dict(y_projection_method="y", show_panelbounds=True), series_kwargs=dict(lw=5, c="w"))
    LSC2.plot(theta_pro[::1], x_pro[::1], y_pro[::1], seriestype="line", panel_kwargs=dict(y_projection_method="y", show_panelbounds=True), series_kwargs=[dict(lw=3, ls="-", c=colors[thidx]) for thidx in range(len(theta_raw))])

    #plotting
    fig = plt.figure(figsize=(9,9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # ax1.set_title("`y_projection_method=\"theta\"`", y=1.1)
    # ax2.set_title("`y_projection_method=\"y\"`", y=1.1)

    lstein.LSteinMPL(LSC1).show(ax1)
    lstein.LSteinMPL(LSC2).show(ax2)
    fig.tight_layout()
    
    if SAVE: fig.savefig(f"../report/gfx/projectionmethods_{context}.pdf")
    return fig

def plot_spectra():

    #load data
    datadir = "../data/Pessi2023_SN2017cfo_spectra/"
    df_obs = pl.read_csv(f"{datadir}wiserep_spectra.csv").select("IAU name", "Obs-date", "JD", "Ascii file", "Spec. ID")
    df_obs = df_obs.filter(~pl.col("Spec. ID").is_in([77750,77746,77741,77739]))  #remove very close observations
    dfs_spec = [pl.read_csv(f"{datadir}{df_obs['Ascii file'][idx]}", separator="\t", has_header=False).with_columns(pl.lit(df_obs["Spec. ID"][idx]).alias("Spec. ID")) for idx in range(df_obs.height)]
    dfs_spec = [df.rename({"column_1":"wavelength", "column_2":"flux"}) for df in dfs_spec]

    #preprocessing
    t_explosion = 57822.2   #from Pessi2023
    mjd_offset =  2400000.5
    flux_factor = 1e15
    theta = df_obs["JD"]-mjd_offset - t_explosion
    # theta = df_obs["index"]
    X = [df["wavelength"].to_numpy().flatten() for df in dfs_spec]
    Y = [df["flux"].to_numpy().flatten()*flux_factor for df in dfs_spec]

    #sigma clipping
    sc_mask = lambda x, n=5: (np.median(x)-n*x.std()<x)&(x<np.median(x)+n*x.std())
    X = [X[i][sc_mask(Y[i])] for i in range(len(X))]
    Y = [Y[i][sc_mask(Y[i])] for i in range(len(Y))]

    thetaticks = np.round(np.linspace(theta.min(), theta.max(), 5)).astype(int)
    xticks = np.array([[np.min(xi), np.max(xi)] for xi in X])
    # xticks = np.round(np.linspace(xticks[:,0].min(), xticks[:,1].max(), 5)).astype(int)
    xticks = np.linspace(4000, 9000, 5).astype(int)
    yticks = np.array([[np.min(yi), np.max(yi)] for yi in Y])
    yticks = np.round(np.array([yticks[:,0].min(), yticks[:,1].max()])).astype(int)

    colors = lsu.get_colors(theta, cmap=CMAP)
    panelsize = np.pi/25
    # erg cm(-2) sec(-1) Ang(-1)
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(-np.pi/2,1*np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,1*np.pi/2-panelsize/2), panelsize=panelsize,
        # thetalabel=df.columns[0], xlabel=df.columns[1], ylabel=df.columns[y1idx],
        thetalabel="Time Since\nExplosion [d]", xlabel="Wavelength $[\\mathrm{\AA}]$", ylabel="Flux $\\left[\\frac{\mathrm{erg}}{\\mathrm{cm^2\,s\,}\\mathrm{\AA}}\\right]$",
        thetalabelkwargs=dict(rotation=0, xytext=(1.5,1.5)),
        xlabelkwargs=dict(rotation=-90, textcoords="offset fontsize", xytext=(-3.3,0)),
        xticklabelkwargs=dict(textcoords="offset fontsize", xytext=(-2,-0.5)),
        ylabelkwargs=dict(rotation=0, textcoords="offset fontsize", xytext=(5.5,1.2)),
    )
    for i in range(len(theta)):
        LSP = LSC.add_panel(
            theta[i],
            panelsize=panelsize,
            show_panelbounds=True,
            yticklabelkwargs=dict(rotation=np.linspace(panelsize/2, np.pi/2-panelsize/2, len(theta))[i]*180/np.pi),
        )
        # LSP.plot(X[i], Y[i],  c=colors[i], label=f"{theta[i]}: {thetalabs[i]}")
        LSP.plot(X[i], Y[i],  c=colors[i], label=f"")

    fig = lstein.draw(LSC, figsize=(5,9))
    fig.tight_layout()
    # fig.legend(bbox_to_anchor=(1.0,0.95), fontsize=10)
    if SAVE: fig.savefig(f"../report/gfx/spectra.pdf")
    


    return

def plot_hypsearch():
    df = pl.read_csv("../data/autoencoder_hypsearch.csv")
    df = (df
        .with_columns(
            pl.col("hyperparameters").cast(pl.Enum(sorted(df["hyperparameters"].unique()))).to_physical().alias("hyperparam combination")
        )
        .select(
            pl.col("hyperparam combination"),
            pl.exclude("hyperparameters", "hyperparam combination"),
            pl.col("hyperparameters"),
        )
        .filter(
            pl.col("hyperparam combination") < 6
        )
        .sort(pl.col("hyperparam combination"))
    )
    
    dfs = df.partition_by(df.columns[0], maintain_order=True)

    y1idx = 2
    y2idx = 3
    theta       = np.array([d[0,0] for d in dfs])
    X           = np.array([d[:,1].to_numpy().flatten() for d in dfs])
    Y           = np.array([d[:,y1idx].to_numpy().flatten() for d in dfs])
    Y2          = np.array([d[:,y2idx].to_numpy().flatten() for d in dfs])
    thetalabs   = [d[0,-1] for d in dfs]

    thetaticks  = np.round(np.linspace(df[:,0].min(), df[:,0].max(), 5), decimals=0).astype(int)
    xticks      = np.round(np.linspace(df[:,1].min(), df[:,1].max(), 5), decimals=0).astype(int)
    yticks      = np.round(np.linspace(df[:,y1idx].min(), df[:,y1idx].max(), 5), decimals=0).astype(int)

    colors = lsu.get_colors(theta, cmap=CMAP)
    panelsize = np.pi/12
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(0,np.pi/2), thetaplotlims=(0+panelsize/2,np.pi/2-panelsize/2), panelsize=panelsize,
        # thetalabel=df.columns[0], xlabel=df.columns[1], ylabel=df.columns[y1idx],
        thetalabel="Hyperparameter\nCombination", xlabel="Epoch", ylabel="Loss",
        thetalabelkwargs=dict(rotation=-45, xytext=(1.5,1.5)),

    )
    for i in range(len(theta)):
        LSP = LSC.add_panel(
            theta[i], yticks=np.unique(np.linspace(np.floor(Y[i].min()), np.ceil(Y[i].max()), 5).astype(int)),
            panelsize=panelsize,
            show_panelbounds=True,
            yticklabelkwargs=dict(rotation=np.linspace(panelsize/2, np.pi/2-panelsize/2, len(theta))[i]*180/np.pi),
        )
        LSP.plot(X[i], Y[i],  c=colors[i], label=f"{theta[i]}: {thetalabs[i]}")
        LSP.plot(X[i], Y2[i], c=colors[i],  ls="--")

    fig = lstein.draw(LSC, figsize=(7,7))
    fig.tight_layout()
    fig.legend(bbox_to_anchor=(1.0,0.95), fontsize=10)
    if SAVE: fig.savefig(f"../report/gfx/hypsearch.pdf")
    
    return

def plot_snn():

    #simulation
    ##neuron params
    n_neurons   = 3
    u_rest      = -65*mV
    # u_rest      = 0*mV
    u_reset     = -75*mV
    u_th        = -50*mV #(=theta)
    C_m         = 750*pF
    R_m         = 0.02*Gohm
    tau_m       = R_m*C_m
    a0          = 3e-1*(1/mV)   #qif specific
    u_c         = -50*mV        #qif specific
    Delta_T     = 3*mV          #eif specific
    theta_rh    = -50*mV        #eif specific
    delta_abs   = 0*ms  #refractory period

    ##simulation specs
    t_sim = .2 * second
    dt = .1 * ms
    
    ##config brian2
    brian2.defaultclock.dt = dt

    ##neuron inputs
    t_in = np_.arange(0, t_sim, dt)
    I_in = TimedArray(np_.linspace([800]*len(t_in), 1000, n_neurons).T * pA, dt=dt)

    ##init network
    brian2.start_scope()
    net1 = Network()
    
    ##setup neurons
    def get_lif():
        eqs = dict(
            model=(
                'du/dt = -(u-u_rest)/tau_m + (R_m*I)/tau_m : volt (unless refractory)\n'
                'I = I_in(t, i) : amp \n'
            ),
            threshold="u>u_th",
            reset="u=u_reset",
            refractory=delta_abs,
        )
        G = NeuronGroup(n_neurons, **eqs, method="euler", dt=dt)
        G.u = u_reset
        state_mon = StateMonitor(G, ["u","I"], record=True)
        return G, state_mon
    def get_eif():
        eqs = dict(
            model=(
                'du/dt = (-(u-u_rest)/tau_m) + Delta_T * exp((u - theta_rh)/Delta_T)/tau_m'
                ' + R_m*I/tau_m'
                '\n'
                '   : volt (unless refractory) \n'
                'I = I_in(t, i) : amp \n'
            ),
            threshold="u>u_th",
            reset="u=u_reset",
            refractory=delta_abs,
        )
        G = NeuronGroup(n_neurons, **eqs, method="euler", dt=dt)
        G.u = u_reset
        state_mon = StateMonitor(G, ["u","I"], record=True)
        return G, state_mon
    def get_qif():
        eqs = dict(
            model=(
                'du/dt = (a0*(u-u_rest)*(u-u_c)/tau_m)'
                ' + R_m*I/tau_m'
                '\n'
                '   : volt (unless refractory) \n'
                'I = I_in(t, i) : amp \n'
            ),
            threshold="u>u_th",
            reset="u=u_reset",
            refractory=delta_abs,
        )
        G = NeuronGroup(n_neurons, **eqs, method="euler", dt=dt)
        G.u = u_reset
        state_mon = StateMonitor(G, ["u","I"], record=True)
        return G, state_mon

    G_lif, state_mon_lif = get_lif()
    G_eif, state_mon_eif = get_eif()
    G_qif, state_mon_qif = get_qif()
    net1.add([
        G_lif, state_mon_lif,
        G_eif, state_mon_eif,
        G_qif, state_mon_qif,
    ])

    ##simulate
    net1.run(t_sim)

    #plotting
    ##traditional
    fig, axs = plt.subplots(1,1, figsize=(9,5), subplot_kw=dict(xlabel="Time [ms]"))
    axs.set_ylabel("$u_\mathrm{membrane}$ [mV]")
    colors = lsu.get_colors(np.unique(state_mon_lif.I), cmap=CMAP)
    sm = plt.cm.ScalarMappable(mcolors.Normalize(vmin=state_mon_lif.I.min()/pA, vmax=state_mon_lif.I.max()/pA))
    for n in range(n_neurons):
        axs.plot(state_mon_lif.t/ms, state_mon_lif.u[n]/mV, color=colors[n], ls="-", label=f"LIF"*(n == 0))
        axs.plot(state_mon_eif.t/ms, state_mon_eif.u[n]/mV, color=colors[n], ls="--", label=f"EIF"*(n == 0))
        axs.plot(state_mon_qif.t/ms, state_mon_qif.u[n]/mV, color=colors[n], ls="-.", label=f"QIF"*(n == 0))
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label("$I_\mathrm{ext}$ [pA]")
    axs.legend(loc="upper left")
    fig.tight_layout()

    ##lstein
    theta = np.concat([
        np.unique(state_mon_lif.I/pA),
        np.unique(state_mon_eif.I/pA),
        np.unique(state_mon_qif.I/pA)
    ])
    x = np.concat([
        [state_mon_lif.t/ms]*n_neurons,
        [state_mon_eif.t/ms]*n_neurons,
        [state_mon_qif.t/ms]*n_neurons,
    ])
    y = np.concat([
        state_mon_lif.u/mV,
        state_mon_eif.u/mV,
        state_mon_qif.u/mV,
    ])
    thetaticks  = np.round(np.linspace(theta.min(), theta.max(), 5), decimals=0).astype(int)
    xticks      = np.round(np.linspace(x.min(), x.max(), 5), decimals=0).astype(int)
    yticks      = np.round(np.linspace(y.min(), y.max(), 5), decimals=0).astype(int)
    colors = np.repeat(np.array(lsu.get_colors(np.arange(n_neurons), cmap=CMAP)).reshape(-1,1), len(np.unique(theta)))     #one color per NEURON TYPE
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(0,np.pi/2), thetaplotlims=(0+panelsize/2,np.pi/2-panelsize/2), panelsize=panelsize,
        xlimdeadzone=0.35,
        thetalabel="$I_\mathrm{ext}$ [pA]", xlabel="Time [ms]", ylabel="$u_\mathrm{membrane}$ [mV]",
        thetalabelkwargs=dict(textcoords="offset fontsize", xytext=(2,2)),
        xlabelkwargs=dict(xytext=(-3.5,-2)),
        ylabelkwargs=dict(rotation=-82, textcoords="offset fontsize", xytext=(0,-3)),
        thetaticklabelkwargs=dict(rotation=30),
    )

    LSC.plot(theta, x, y, seriestype="line", series_kwargs=[dict(c=colors[i], label=[f"LIF","","", f"EIF","","", f"QIF","",""][i]) for i in range(len(theta))])

    fig = lstein.draw(LSC, figsize=(9,9))
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    if SAVE: fig.savefig(f"../report/gfx/snn.pdf")

    return

def plot_errorband():

    # fig, axs = plt.subplots(1,1)
    # # axs.errorbar(x_raw[-1], y_raw[-1], yerr=y_raw_e[-1], ls="", marker="o")
    # axs.scatter(x_raw[-1], y_raw[-1])
    # axs.plot(x_raw[-1], y_raw[-1]-y_raw_e[-1])
    # axs.plot(x_raw[-1], y_raw[-1]+y_raw_e[-1])
    # # axs.plot(x_pro[-1], y_pro[-1])
    # axs.invert_xaxis()
    # axs.invert_yaxis()
    # plt.show()

    yticks=np.linspace(-100, 100, 5).astype(int)
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(-np.pi/2,2*np.pi/2), thetaplotlims=(-np.pi/2+1.2*panelsize/2,2*np.pi/2-1.2*panelsize/2),
        xlimdeadzone=0.2, 
        thetalabel=thetalab, xlabel=xlab, ylabel=ylab,
        thetaarrowpos_th=None, ylabpos_th=np.min(theta_raw),
        thetatickkwargs=None, thetaticklabelkwargs=dict(pad=0.3), thetalabelkwargs=dict(rotation=40, textcoords="offset fontsize", xytext=(-2,-1.5)),
        xtickkwargs=None, xticklabelkwargs=dict(xytext=(-0.2,0), ha="right", va="center"), xlabelkwargs=dict(rotation=90, textcoords="offset fontsize", xytext=(-3,-0)),
        ylabelkwargs=dict(textcoords="offset fontsize", xytext=(-0,-2)),
    )

    #adding all the series (will initialize panels for you)
    LSC.plot(theta_raw, x_raw, y_raw, seriestype="scatter", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=[dict(s=10, alpha=1.0, c=colors[thidx]) for thidx in range(len(theta_raw))])
    LSC.plot(theta_raw, x_raw, [y-np.abs(ye) for y, ye in zip(y_raw, y_raw_e)], seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=[dict(lw=2, ls="--", c=colors[thidx], alpha=0.5) for thidx in range(len(theta_raw))])
    LSC.plot(theta_raw, x_raw, [y+np.abs(ye) for y, ye in zip(y_raw, y_raw_e)], seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=[dict(lw=2, ls="--", c=colors[thidx], alpha=0.5) for thidx in range(len(theta_raw))])
    # LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", panel_kwargs=dict(y_projection_method="y"), seres_kwargs=[dict(alpha=1.0, c=colors[thidx]) for thidx in range(len(theta_raw))])
    
    # LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=dict(lw=3, c="w"))
    # LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=dict(ls="-"))
    # LSC.plot(theta_pro, x_pro, [y-ye for y, ye in zip(y_pro, y_pro_e)], seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=dict(ls="--", alpha=0.5))
    # LSC.plot(theta_pro, x_pro, [y+ye for y, ye in zip(y_pro, y_pro_e)], seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=dict(ls="--", alpha=0.5))

    fig = lstein.draw(LSC, figsize=(9,9))
    fig.tight_layout()

    if SAVE: fig.savefig("../report/gfx/lstein_errorband.pdf")
    return
#%%main
def plot_graphical_abstract(LSCs):

    fig = plt.figure(figsize=(16,9))
    axs = [
        fig.add_axes([0.0,0.0,0.55,1.0]),
        fig.add_axes([0.55,0.0,0.45,1.0]),
    ]
    for (ax, LSC) in zip(axs, LSCs):
        LSC.reset()
        lstein.LSteinMPL(LSC).show(ax)

    fig.savefig("graphical_abstract.pdf", dpi=180)

    return

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
    # for i in range(42):
    #     try: plot_lstein(i); plt.close()
    #     except: pass
    LSCa = plot_lstein_snii()
    LSCb = plot_lstein_tde()
    plot_graphical_abstract([LSCa,LSCb])
    # plot_projection_methods(context="theta")
    # plot_projection_methods(context="y")
    # plot_spectra()
    # plot_hypsearch()
    # plot_snn()
    # plot_errorband()

    # plot_scatter_multipanel()

    # #plots with increased fontsize (one column)
    # plt.rcParams["font.size"] = 25
    # plot_scatter_onepanel()
    # plot_scatter_onepanel_offset()
    # plot_scatter_multipanel_group()
    # plot_heatmap()
    # plot_3dsurface()
    

    plt.show()
    return

if __name__ == "__main__":
    main()