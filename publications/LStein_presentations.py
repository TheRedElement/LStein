#%%imports
import brian2
import brian2.numpy_ as np_
from brian2 import NeuronGroup, Network
from brian2 import TimedArray
from brian2 import StateMonitor, SpikeMonitor
from brian2 import Gohm, ms, mV, pA, pF, second
import json
import numpy as np
import polars as pl
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Tuple

from lstein import lstein, utils as lsu, makedata as md, paper_plots as pp

pio.templates.default = "plotly_dark"

#%%constants
CMAP:str = "plasma"
CMAP_PB:str = "jet"


#%%definitions

def binning(
    x, y, dx,
    func=np.nanmean,
    ):

    xbin = np.array([])
    ybin = np.array([])
    
    #init inteval bounds
    xlb = np.nanmin(x)
    xub = xlb+dx
    #apply binning
    while xub <= np.nanmax(x):
        mask = (xlb <= x)&(x < xub)  #current inteval
        xbin = np.append(xbin, func(x[mask]))
        ybin = np.append(ybin, func(y[mask]))
        
        #update
        xlb += dx
        xub += dx


    return xbin, ybin

def get_passbands() -> dict:
    df_pb = pl.read_csv("../data/passband_specs.csv")
    # passbands = list(df_pb["name"])
    pb_mappings = dict(zip(df_pb["wavelength"], df_pb.select(pl.exclude("wavelength")).to_numpy()))
    
    #adjust colors
    bands = list("ugrizy")
    pb_colors = lsu.get_colors(range(len(bands)), cmap=CMAP_PB, vmin=-0.5, vmax=5)
    pb_colors = {pb:c for pb, c in zip(bands, pb_colors)}
    
    #adjustments
    for pb in pb_mappings.keys():
        #translate markers to plotly
        pb_mappings[pb][2] = {
            "o":"circle",
            "^":"triangle-up",
            "v":"triangle-down",
            "s":"square",
            "*":"star",
            "p":"pentagon",
        }[pb_mappings[pb][2]]
        if pb_mappings[pb][0] in bands:
            pb_mappings[pb][1] = pb_colors[pb_mappings[pb][0]]

    return pb_mappings

def load_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(f"../data/0901_snii_elasticc.csv", comment_prefix="#")
    
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

def load_rubin(pb_mappings:dict) -> Tuple:

    obj, sntype = "313998569623257167", "snii"
    obj, sntype = "314003014107006318", "snic"
    # obj, sntype = "170107660764446767", "snii"

    pb_inv = {v[0]:k for k, v in pb_mappings.items()}   #invert to get wavelength of passband

    cols = {
        "r:midpointMjdTai":"time [d]", 
        "r:scienceFlux":"flux_science", "r:scienceFluxErr":"flux_science_e",
        "r:templateFlux":"flux_template", "r:templateFluxErr":"flux_template_e",
        "r:band":"band",
    }
    with open(f"../data/fink_lsst_alerts/{obj}.json", "r") as f:
        data = json.load(f)
        df_lc = pl.from_dicts([
            {v:alert[k] for k, v in cols.items()}
        for alert in data])

    df_lc = (df_lc
        .with_columns(
            (pl.col("flux_science") - pl.col("flux_template")).alias("flux_difference"),
            (pl.col("flux_science_e") + pl.col("flux_template_e")).alias("flux_difference_e"),
            (pl.col("band").replace(pb_inv).cast(pl.Float64)).alias("Wavelength [nm]"),
        )
    )

    pb_raw = df_lc["Wavelength [nm]"].to_numpy().flatten()
    x_raw = (df_lc["time [d]"] - df_lc["time [d]"].min()).to_numpy().flatten()
    y_raw = df_lc["flux_difference"].to_numpy().flatten() * 1e-3
    y_raw_e = df_lc["flux_difference_e"].to_numpy().flatten() * 1e-3
    return (obj, sntype, 
        pb_raw, x_raw, y_raw, y_raw_e
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
                visible=False,
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

def plot_lstein_snii(
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

def run_brian2():

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

    return n_neurons, state_mon_lif, state_mon_eif, state_mon_qif

def plot_lstein_snn():
    n_neurons, state_mon_lif, state_mon_eif, state_mon_qif = run_brian2()

    panelsize = np.pi/8

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
    thetaticks  = np.round(np.linspace(theta.min(), theta.max(), 3), decimals=0).astype(int)
    xticks      = np.round(np.linspace(x.min(), x.max(), 5), decimals=0).astype(int)
    yticks      = np.round(np.linspace(y.min(), y.max(), 5), decimals=0).astype(int)
    colors = np.repeat(np.array(lsu.get_colors(np.arange(n_neurons), cmap=CMAP)).reshape(-1,1), len(np.unique(theta)))     #one color per NEURON TYPE
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(0,np.pi/2), thetaplotlims=(0+panelsize/2,np.pi/2-panelsize/2), panelsize=panelsize,
        xlimdeadzone=0.35,
        # thetalabel="$I_\mathrm{ext}$ [pA]", xlabel="Time [ms]", ylabel="$u_\mathrm{membrane}$ [mV]",
        thetalabel="External current [pA]", xlabel="Time [ms]", ylabel="Membrane potential [mV]",
        thetalabelkwargs=dict(c="w", textangle=45, yshift=20, xshift=20),
        xlabelkwargs=dict(c="w", yshift=-20),
        ylabelkwargs=dict(c="w", textangle=80, xshift=5),
        thetaticklabelkwargs=dict(c="w"),
        xticklabelkwargs=dict(c="w", yshift=-10),
    )

    LSC.plot(theta, x, y, seriestype="line",
        series_kwargs=[dict(
            c=colors[i],
            name=[f"LIF","","", f"EIF","","", f"QIF","",""][i],
            showlegend=[True,False,False,True,False,False,True,False,False][i],
            ) for i in range(len(theta))],
        panel_kwargs=[dict(
            show_yticks=[True,True,True,False,False,False,False,False,False][i],
            yticklabelkwargs=dict(c="w"),
        ) for i in range(len(theta))],
    )

    fig = lstein.draw(LSC, backend="plotly")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
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
    pio.write_json(fig, f"../gfx/LsteinSnn.json", pretty=True)
    fig.show()
    return

def plot_lstein_pulsar():

    data = np.load("../data/pulsar_data/J0437-4715_2021-02-03.npz")     #very bright
    data = np.load("../data/pulsar_data/J1804-2858_2024-03-19.npz")     #very faint
    data = np.load("../data/pulsar_data/J2145-0750_2023-03-07.npz")     #RFI contaminated
    data = np.load("../data/pulsar_data/J2145-0750_2023-03-30.npz")     #cleaned (follow-up observation)
    print(data.files)
    freq    = np.linspace(0, data["bandwidth"], data["nchan"]) + data["freq_ctr"]-data["bandwidth"]/2
    phase   = np.linspace(0, 1, data["nbin"])
    print(data["nsubint"])
    subint  = np.linspace(0, 8, data["nsubint"])

    #define dimensions
    theta = freq
    X = np.repeat(phase.reshape(1,-1), theta.shape[0], axis=0)
    Y = data["freq_phase"][:theta.shape[0]]
    # plt.plot(np.nanmean(X, axis=0), np.nanmean(Y, axis=0), zorder=10)
    # for i in range(10):
    #     plt.plot(X[i,:], Y[i,:])
    # plt.show()

    #normalize
    # Y = Y / np.nanmedian(Y, axis=1, keepdims=True)

    #introduce missing freqs
    fidx = np.random.choice(range(Y.shape[0]), size=10, replace=False)
    Y[fidx,:] = np.nan

    #filter NaNs
    nanmask = np.any(np.isfinite(Y), axis=1)
    theta = theta[nanmask]
    X = X[nanmask]
    Y = Y[nanmask]

    subset = slice(0,None,1)
    theta = theta[subset]
    X = X[subset,:]
    Y = Y[subset,:]
    X = [xi for xi in X]
    Y = [yi for yi in Y]
    thetaticks = np.round(np.linspace(theta.min(), theta.max(), 5)).astype(int)
    xticks = np.array([[np.nanmin(xi), np.nanmax(xi)] for xi in X])
    xticks = np.round(np.linspace(np.nanmin(xticks[:,0]), np.nanmax(xticks[:,1]), 5), 1)#.astype(int)
    # xticks = np.linspace(4000, 9000, 5).astype(int)
    yticks = np.array([[np.nanmin(yi), np.nanmax(yi)] for yi in Y])
    yticks = np.array([np.floor(np.nanmin(yticks[:,0])), np.ceil(np.nanmax(yticks[:,1]))]).astype(int)

    colors = lsu.get_colors(theta, cmap=CMAP)
    panelsize = np.pi/12
    plotlims = [-np.pi/2, 1*np.pi/2]
    # plotlims = [np.pi/2, 3*np.pi/2]
    # xticks = xticks[::-1]
    # yticks = yticks[::-1]
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=(plotlims[0],1*plotlims[1]), thetaplotlims=(plotlims[0]+panelsize/2,1*plotlims[1]-panelsize/2), panelsize=panelsize,
        # thetalabel=df.columns[0], xlabel=df.columns[1], ylabel=df.columns[y1idx],
        thetalabel="Frequency<br>[MHz]", xlabel="Phase", ylabel="Flux",
        thetalabelkwargs=dict(c="w", textangle=0,),
        thetaticklabelkwargs=dict(c="w", pad=0.25),
        xlabelkwargs=dict(c="w", textangle=90, xshift=-25, yshift=10),
        xticklabelkwargs=dict(c="w", xshift=-10),
        ylabelkwargs=dict(c="w", textangle=0, xshift=170, yshift=350),
    )
    for i in range(len(theta)):
        # show_y_guides = (i==16) #only for one specific LSP
        show_y_guides = (i==80) #only for one specific LSP

        LSP = LSC.add_panel(
            theta[i],
            panelsize=panelsize,
            show_panelbounds=show_y_guides,
            show_yticks=show_y_guides,
            y_projection_method="theta",
            # yticklabelkwargs=dict(rotation=np.linspace(panelsize/2, np.pi/2-panelsize/2, len(theta))[i]*180/np.pi),
            yticklabelkwargs=dict(c="w", textangle=38,),
            panelboundskwargs=dict(zorder=100, c="w")
        )
        # LSP.plot(X[i], Y[i],  c=colors[i], label=f"{theta[i]}: {thetalabs[i]}")
        LSP.plot(X[i], Y[i],  c=colors[i], label=f"", lw=1, alpha=0.9, showlegend=False)

    fig = lstein.draw(LSC, backend="plotly")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
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
    pio.write_json(fig, "../gfx/LsteinPulsar.json", pretty=True)
    fig.show()

def plot_lstein_spectra():

    datadir = "../data/wiserep_SN2023ixf_spectra/"
    lbda_factor = 1     #1/10  #convert AA -> nm
    #load data
    df_obs = (pl.read_csv(f"{datadir}wiserep_spectra.csv", comment_prefix="#")
        .select("IAU name", "Obs-date", "JD", "Ascii file", "Phase (days)", "Spec. ID")
        .filter(pl.col("Ascii file").str.contains("Mayall"))   #different format -> ignore
        .sort("Ascii file")
        # .sort("Phase (days)")
    )
    print(df_obs["Phase (days)"].to_numpy())
    print(df_obs.height)
    df_obs = df_obs[[0,5,8,11,15,17],:] #wiserep

    # df_obs = df_obs.filter((pl.col("Phase (days)").abs() < 1)) #wiserep
    dfs_spec = [] 
    for idx in range(df_obs.height):
        df_spec = (
            pl.read_csv(f"{datadir}{df_obs['Ascii file'][idx]}", comment_prefix="#", separator=" ")
                .rename({"WAVE":"wavelength","FLUX":"flux"})
        )
        dfs_spec.append(df_spec)

    #preprocessing
    flux_factor = 1e-3#1e15
    theta = df_obs["Phase (days)"]
    # theta = df_obs["index"]
    X = [df["wavelength"].to_numpy().flatten()*lbda_factor for df in dfs_spec]
    Y = [df["flux"].to_numpy().flatten()*flux_factor for df in dfs_spec]

    #continuum removal
    XY_cont = [binning(xi, yi, 500, np.nanmedian) for xi, yi in zip(X, Y)]
    X_cont = [np.linspace(X[i].min(), X[i].max(), X[i].shape[0]) for i in range(len(X))]
    Y_cont = [np.interp(np.linspace(X[i].min(), X[i].max(), X[i].shape[0]), XY_cont[i][0], XY_cont[i][1]) for i in range(len(X))]
    
    Y = [Y[i] - Y_cont[i] for i in range(len(Y))]
    # for i in range(len(X)):
    #     plt.plot(X[i], Y[i])
    #     plt.plot(X_cont[i], Y_cont[i])

    # #wavelength constraint
    Y = [Y[i][((5000*lbda_factor<X[i]) & (X[i]<np.inf))] for i in range(len(X))]
    X = [X[i][((5000*lbda_factor<X[i]) & (X[i]<np.inf))] for i in range(len(X))]

    #sigma clipping
    sc_mask = lambda x, n=5: (np.median(x)-n*x.std()<x)&(x<np.median(x)+n*x.std())
    X = [X[i][sc_mask(Y[i])] for i in range(len(X))]
    Y = [Y[i][sc_mask(Y[i])] for i in range(len(Y))]

    #binning
    XY = [binning(xi, yi, 10, np.nanmean) for xi, yi in zip(X, Y)]
    X = [XYi[0] for XYi in XY]
    Y = [XYi[1] for XYi in XY]

    # for i in range(len(theta)):
    #     plt.plot(X[i], Y[i])

    #X as an offset (to ensure computation with smaller values => minimize projection effects)
    Xmin = np.min([np.min(xi) for xi in X])
    Xmax = np.max([np.max(xi) for xi in X])
    Xmin = 5000*lbda_factor
    Xmax = 9500*lbda_factor
    X = [lsu.minmaxscale(xi, 0, 10, Xmin, Xmax) for xi in X]

    thetaticks = np.round(np.linspace(theta.min(), theta.max(), 5)).astype(int)
    xticks = np.array([[np.min(xi), np.max(xi)] for xi in X])
    xticks = np.round(np.linspace(xticks[:,0].min(), xticks[:,1].max(), 5), 5).astype(int)
    # xticks = (xticks,np.linspace(Xmin,Xmax,5).astype(int))   #make sure ticklabels display correct value
    yticks = np.array([[np.min(yi), np.max(yi)] for yi in Y])
    yticks = np.round(np.array([yticks[:,0].min(), yticks[:,1].max()]), 1).astype(float)

    colors = lsu.get_colors(theta, cmap=CMAP)
    panelsize = np.pi/10
    # guidelims = (-np.pi/2,1*np.pi/2)
    guidelims = (3*np.pi/2,np.pi/2)
    xticks = (xticks[::-1], np.linspace(Xmin,Xmax,5).astype(int)[::-1])
    yticks = yticks[::-1]
    # erg cm(-2) sec(-1) Ang(-1)
    LSC = lstein.LSteinCanvas(
        thetaticks, xticks, yticks,
        thetaguidelims=guidelims, thetaplotlims=(guidelims[0]-1.3*panelsize+1.3*panelsize/2,guidelims[1]+panelsize-panelsize/2), panelsize=panelsize,
        # thetalabel=df.columns[0], xlabel=df.columns[1], ylabel=df.columns[y1idx],
        thetalabel="Time since peak [d]", xlabel="Wavelength [&#xC5;]", ylabel=r"Flux &#x00b7; 10<sup>-20</sup> [erg/cm<sup>2</sup>/s/&#xC5;]",
        thetalabelkwargs=dict(c="w", textangle=0, xshift=10),
        xlabelkwargs=dict(c="w", textangle=-90, xshift=40, yanchor="bottom"),
        xticklabelkwargs=dict(c="w", xshift=20),
        ylabelkwargs=dict(c="w", textangle=0, xshift=0, yshift=-10),
    )
    for i in range(len(theta)):
        rot = lsu.minmaxscale(theta[i], *LSC.thetaplotlims, *LSC.thetalims)*180/np.pi + 180 #rotating labels
        LSP = LSC.add_panel(
            theta[i],
            panelsize=panelsize,
            show_panelbounds=True,
            y_projection_method="theta",
            panelboundskwargs=dict(c="w"),
            yticklabelkwargs=dict(c="w"),
        )
        LSP.plot(X[i], Y[i],  c=colors[i], showlegend=False)

    fig = lstein.draw(LSC, backend="plotly")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
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
    pio.write_json(fig, f"../gfx/LsteinSpectra.json", pretty=True)
    fig.show()
    
    return

def plot_lstein_rubin(
    obj:str, sntype:str,
    pb_rubin:np.ndarray, x_rubin:np.ndarray, y_rubin:np.ndarray, y_rubin_e:np.ndarray,
    pb_mappings:dict,
    ) -> None:

    thticks = np.linspace(pb_rubin.min(), pb_rubin.max(), 5).astype(int)
    xticks  = np.arange(np.floor(x_rubin.min()), np.ceil(x_rubin.max()), 20).astype(int)
    yticks  = np.linspace(y_rubin.min(), y_rubin.max(), 3).round(1)

    LSC = lstein.LSteinCanvas(
        thticks, xticks, yticks,
        thetaguidelims=(-1*np.pi/2,1*np.pi/2),
        panelsize=np.pi/6,
        xticklabelkwargs=dict(c="#ffffff", xshift=-13, yshift=0),
        thetaticklabelkwargs=dict(c="#ffffff"),
        xlabel="Time [d]", xlabelkwargs=dict(c="w", textangle=90, xshift=-30, yshift=20),
        ylabel="Difference flux [&#xb5;Jy]", ylabelkwargs=dict(c="w", textangle=0),
        thetalabel="Wavelength [nm]", thetalabelkwargs=dict(c="w", xanchor="right", xshift=30),
    )
    for idx, pb in enumerate(np.unique(pb_rubin)):
        
        LSP = LSC.add_panel(pb,
            yticklabelkwargs=dict(c="#ffffff"),
            yticks=(yticks if idx==0 else (yticks, [""]*len(yticks))),
            y_projection_method="theta",
            # ytickkwargs=dict(c="w"),
            show_panelbounds=True,
            panelboundskwargs=dict(c="w"),
        )
        LSP.plot(x_rubin[(pb_rubin==pb)], y_rubin[(pb_rubin==pb)], seriestype="scatter", marker=dict(color=pb_mappings[pb][1], symbol=pb_mappings[pb][2]))

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
    pio.write_json(fig, f"../gfx/LsteinRubin{sntype.capitalize()}.json", pretty=True)
    fig.show()        

    return

def plot_onepanel_rubin(
    obj:str, sntype:str,
    pb_rubin:np.ndarray, x_rubin:np.ndarray, y_rubin:np.ndarray, y_rubin_e:np.ndarray,
    pb_mappings:dict,
    ):

    fig = make_subplots(1,1,
        x_title="Time [d]",
        y_title="Flux [&#xb5;Jy]",
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
            x=x_rubin[(pb_rubin==pb)],
            y=y_rubin[(pb_rubin==pb)],
            error_y=dict(
                type="data",
                array=y_rubin_e[(pb_rubin==pb)],
                visible=True,
            ),
            type="scatter", mode="markers",
            name=f"{pb_mappings[pb][4].upper()} {pb_mappings[pb][0]} ({pb} nm)",
            marker=dict(
                color=pb_mappings[pb][1],
                symbol=pb_mappings[pb][2],
            )
        )
    for pb in np.unique(pb_rubin)])

    pio.write_json(fig, f"../gfx/ScatterOnepanelRubin{sntype.capitalize()}.json", pretty=True)

    fig.show()    

    return
#%%main
def main():

    (pb_raw, x_raw, y_raw, y_raw_e), \
        (pb_pro, x_pro, y_pro, y_pro_e), = load_data()
    pb_mappings = get_passbands()

    obj, sntype, \
        pb_rubin_rubin, x_rubin, y_rubin, y_rubin_e = load_rubin(pb_mappings)

    plot_onepanel(
        pb_raw, x_raw, y_raw, y_raw_e,
        pb_pro, x_pro, y_pro, y_pro_e,
        pb_mappings,
    )
    plot_lstein_snii(
        pb_raw, x_raw, y_raw, y_raw_e,
        pb_pro, x_pro, y_pro, y_pro_e,
        pb_mappings,
    )
    plot_lstein_snn()
    plot_lstein_pulsar()
    plot_lstein_spectra()
    plot_lstein_rubin(obj, sntype, pb_rubin_rubin, x_rubin, y_rubin, y_rubin_e, pb_mappings)
    plot_onepanel_rubin(obj, sntype, pb_rubin_rubin, x_rubin, y_rubin, y_rubin_e, pb_mappings)



if __name__ == "__main__":
    main()
