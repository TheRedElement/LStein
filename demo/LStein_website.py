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
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import polars as pl
import re
from LuStCodeSnippets_py.Styles import PlotStyles

from lstein import lstein, utils as lsu, makedata as md, paper_plots as pp
importlib.reload(pp)

#setup plotting style
tre_dark = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(
            # size=14,
            color="#FFFFFF"
        ),
        colorway=[
            "#ffffff",
            "#9E9E9E"
        ],
        legend=go.Legend(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#4d4d4d",
        )    
    ),
    data=dict(
        scatter=[
            go.Scatter(
                line=dict(width=1)   # default line width for all scatter lines
            )
        ],
    ),
)
#seeds

#%%constants
SURVEY_MAPPING:dict = {"elasticc":"ELAsTiCC", "des":"DES"}
OTYPE_MAPPING:dict = {"snia":"SN Ia", "snii":"SN II", "snibc":"SN Ib/c", "tde":"TDE"}
CMAP:str = "autumn"
CMAP_BANDS:str = "nipy_spectral"
SAVE:bool = True

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
        ylab = "Brightness [FLUXCAL]"

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
    unique_thetas = np.unique(theta_raw)
    thetaticks = np.round(np.linspace(np.floor(np.min(theta_raw)), np.ceil(np.max(theta_raw)), nthticks),0).astype(int)
    xticks = np.round(np.linspace(np.floor(np.min(np.concat(x_raw))), np.ceil(np.max(np.concat(x_raw))), nxticks), decimals=0).astype(int)
    yticks = np.round(np.linspace(np.floor(np.min(np.concat(y_raw))), np.ceil(np.max(np.concat(y_raw))), nyticks), decimals=0).astype(int)
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
        thetatickkwargs=None, thetaticklabelkwargs=dict(pad=0.3, font=dict(color=tre_dark.layout.font.color)), thetalabelkwargs=dict(yshift=-10, font=dict(color=tre_dark.layout.font.color)),
        xtickkwargs=dict(line=dict(color=tre_dark.layout.colorway[1])), xticklabelkwargs=dict(yshift=-10, font=dict(color=tre_dark.layout.font.color)), xlabelkwargs=dict(yshift=-20, font=dict(color=tre_dark.layout.font.color)),
        ylabelkwargs=dict(textangle=85, xshift=7, font=dict(color=tre_dark.layout.font.color)),
    )

    #adding all the series (will initialize panels for you)
    print(pb_mappings)
    LSC.plot(theta_raw, x_raw, y_raw, seriestype="scatter",
        panel_kwargs=dict(y_projection_method="theta", show_panelbounds=True,
            panelboundskwargs=dict(line=dict(color=tre_dark.layout.colorway[0])),
            ytickkwargs=dict(line=dict(color=tre_dark.layout.colorway[1])),
            yticklabelkwargs=dict(font=dict(color=tre_dark.layout.font.color)),
        ),
        series_kwargs=[dict(name=pb_mappings[tr][0]) for tr in theta_raw]
    )
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", series_kwargs=dict(line=dict(color=tre_dark.layout.font.color, width=tre_dark.data.scatter[0].line.width+2), showlegend=False))
    LSC.plot(theta_pro, x_pro, y_pro, seriestype="line", panel_kwargs=dict(y_projection_method="theta"), series_kwargs=dict(showlegend=False))

    fig = lstein.draw(LSC, backend="plotly")
    fig.update_layout(
        template=tre_dark,
    )
    fig.show()

    if SAVE: fig.write_json(f"../gfx/lstein_website_{otype}_{survey}.json", pretty=True)
    return
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
    plot_lstein_snii()

    return

if __name__ == "__main__":
    main()