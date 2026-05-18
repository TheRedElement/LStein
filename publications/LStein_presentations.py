#%%imports
import numpy as np
import polars as pl
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Tuple

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
    
    df_raw = df.filter(pl.col("processing")=="raw")
    df_pro = df.filter(pl.col("processing")=="gp")

    pb_raw, x_raw, y_raw = df_raw[:,:3].to_numpy().T
    pb_pro, x_pro, y_pro = df_pro[:,:3].to_numpy().T

    #normalize
    pb_pro_r = (pb_pro==622.3)
    x_peak_r = x_pro[pb_pro_r][np.argmax(y_pro[pb_pro_r])]
    x_raw -= x_peak_r
    x_pro -= x_peak_r

    y_raw /= y_pro.max()
    y_pro /= y_pro.max()


    return (
        (pb_raw, x_raw, y_raw),
        (pb_pro, x_pro, y_pro),
    )

def plot_onepanel(
    pb_raw:np.ndarray, x_raw:np.ndarray, y_raw:np.ndarray,
    pb_pro:np.ndarray, x_pro:np.ndarray, y_pro:np.ndarray,
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

    pio.write_json(fig, "../gfx/ScatterOnepanel.json", pretty=True)

    fig.show()
    return

#%%main
def main():

    (pb_raw, x_raw, y_raw), \
        (pb_pro, x_pro, y_pro), = load_data()
    pb_mappings = get_passbands()

    plot_onepanel(
        pb_raw, x_raw, y_raw,
        pb_pro, x_pro, y_pro,          
        pb_mappings
    )




if __name__ == "__main__":
    main()
