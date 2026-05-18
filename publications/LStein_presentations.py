#%%imports
import numpy as np
import polars as pl
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Tuple

from lstein import lstein, utils as lsu, makedata as md, paper_plots as pp

pio.templates.default = "plotly_dark"

#%%definitions
def get_passbands() -> dict:
    df_pb = pl.read_csv("../data/passband_specs.csv")
    # passbands = list(df_pb["name"])
    pb_mappings = dict(zip(df_pb["wavelength"], df_pb.select(pl.exclude("wavelength")).to_numpy()))
    
    #translate markers to plotly
    for pb in pb_mappings.keys():
        pb_mappings[pb][2] = {
            "o":"circle",
            "^":"triangle-up",
            "v":"triangle-down",
            "s":"square",
            "*":"star",
            "p":"pentagon",
        }[pb_mappings[pb][2]]

    return pb_mappings

def load_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(f"../data/72147108_snii_elasticc.csv", comment_prefix="#")
    
    print(df)
    df_raw = df.filter(pl.col("processing")=="raw")
    df_pro = df.filter(pl.col("processing")=="gp")

    pb_raw, x_raw, y_raw, y_raw_e = df_raw[:,:4].to_numpy().T
    pb_pro, x_pro, y_pro, y_pro_e = df_pro[:,:4].to_numpy().T

    #normalize
    pb_pro_r = (pb_pro==622.3)
    x_peak_r = x_pro[pb_pro_r][np.argmax(y_pro[pb_pro_r])]
    y_peak = y_pro.max()
    x_raw -= x_peak_r
    x_pro -= x_peak_r

    y_raw /= y_peak
    y_pro /= y_peak
    y_raw_e /= y_peak
    y_pro_e /= y_peak


    return (
        (pb_raw, x_raw, y_raw, y_raw_e),
        (pb_pro, x_pro, y_pro, y_pro_e),
    )

def plot_onepanel(
    pb_raw:np.ndarray, x_raw:np.ndarray, y_raw:np.ndarray, y_raw_e:np.ndarray,
    pb_pro:np.ndarray, x_pro:np.ndarray, y_pro:np.ndarray, y_pro_e:np.ndarray,
    pb_mappings:dict,
    ) -> None:
    fig = make_subplots(1,1,
        x_title="Time [d]",
        y_title="Flux [Arbitrary Units]",
    )

    fig.update_layout(
        margin=dict(
            l=70,
            r=0,
            t=0,
            b=60,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )        
    )

    fig.add_traces([
        dict(
            x=x_raw[(pb_raw==pb)],
            y=y_raw[(pb_raw==pb)],
            error_y=dict(
                type="data",
                array=y_raw_e[(pb_raw==pb)],
                visible=True,
            ),
            type="scatter", mode="markers",
            name=f"{pb_mappings[pb][4].upper()} {pb_mappings[pb][0]} ({pb} nm)",
            marker=dict(
                color=pb_mappings[pb][1],
                symbol=pb_mappings[pb][2],
            )
        )
    for pb in np.unique(pb_raw)])
    fig.add_traces([
        dict(
            x=x_pro[(pb_pro==pb)],
            y=y_pro[(pb_pro==pb)],
            type="scatter", mode="lines",
            name=pb_mappings[pb][0],
            showlegend=False,
            marker=dict(
                color=pb_mappings[pb][1],
            )
        )
    for pb in np.unique(pb_pro)])
    fig.add_traces([
        #GP error-bands
        dict(
            x=np.append(x_pro[(pb_pro==pb)],x_pro[(pb_pro==pb)][::-1]),
            y=np.append(y_pro[(pb_pro==pb)]-y_pro_e[(pb_pro==pb)], (y_pro[(pb_pro==pb)]+y_pro_e[(pb_pro==pb)])[::-1]),
            type="scatter",
            fill="toself",
            name=pb_mappings[pb][0],
            showlegend=False,
            visible=False,
            marker=dict(
                color=pb_mappings[pb][1],
            )
        )
    for pb in np.unique(pb_pro)])

    pio.write_json(fig, "../gfx/ScatterOnepanel.json", pretty=True)

    fig.show()
    return

def plot_lstein(
    pb_raw:np.ndarray, x_raw:np.ndarray, y_raw:np.ndarray, y_raw_e:np.ndarray,
    pb_pro:np.ndarray, x_pro:np.ndarray, y_pro:np.ndarray, y_pro_e:np.ndarray,
    pb_mappings:dict,        
    ) -> None:

    thticks = np.linspace(pb_raw.min(), pb_raw.max(), 5).astype(int)
    xticks  = np.arange(-20, 100, 20).astype(int)
    yticks  = np.linspace(y_pro.min(), y_pro.max(), 3).round(1)

    LSC = lstein.LSteinCanvas(
        thticks, xticks, yticks,
        thetaguidelims=(-1*np.pi/2,1*np.pi/2),
        panelsize=np.pi/6,
        xticklabelkwargs=dict(c="#ffffff", xshift=-13, yshift=0),
        thetaticklabelkwargs=dict(c="#ffffff"),
        xlabel="Time [d]", xlabelkwargs=dict(c="w", textangle=90, xshift=-30, yshift=20),
        ylabel="Normalized flux", ylabelkwargs=dict(c="w", textangle=0),
        thetalabel="Wavelength [nm]", thetalabelkwargs=dict(c="w", xanchor="right", xshift=30),
    )
    for idx, pb in enumerate(np.unique(pb_raw)):
        
        LSP = LSC.add_panel(pb,
            yticklabelkwargs=dict(c="#ffffff"),
            yticks=(yticks if idx==0 else (yticks, [""]*len(yticks))),
            y_projection_method="theta",
            # ytickkwargs=dict(c="w"),
            show_panelbounds=True,
            panelboundskwargs=dict(c="w"),
        )
        LSP.plot(x_raw[(pb_raw==pb)], y_raw[(pb_raw==pb)], seriestype="scatter", marker=dict(color=pb_mappings[pb][1], symbol=pb_mappings[pb][2]))
        LSP.plot(x_pro[(pb_pro==pb)], y_pro[(pb_pro==pb)], seriestype="line", c=pb_mappings[pb][1])

    fig = lstein.draw(LSC, backend="plotly")
    fig.update_layout(
        margin=dict(
            t=0,
            b=0,
            l=0,
            r=0,
        ),
        font=dict(
            size=10,
        ),
    )
    pio.write_json(fig, "../gfx/LsteinSnii.json", pretty=True)
    fig.show()
    return
#%%main
def main():

    (pb_raw, x_raw, y_raw, y_raw_e), \
        (pb_pro, x_pro, y_pro, y_pro_e), = load_data()
    pb_mappings = get_passbands()

    # plot_onepanel(
    #     pb_raw, x_raw, y_raw, y_raw_e,
    #     pb_pro, x_pro, y_pro, y_pro_e,
    #     pb_mappings,
    # )
    plot_lstein(
        pb_raw, x_raw, y_raw, y_raw_e,
        pb_pro, x_pro, y_pro, y_pro_e,
        pb_mappings,
    )




if __name__ == "__main__":
    main()
