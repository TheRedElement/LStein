
#%%imports
import glob
import importlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import os
import polars as pl
import re
import sys
from typing import Literal

sys.path.append("../")
from src_py import LVisPCanvas, utils as lvisu

importlib.reload(LVisPCanvas)
# plt.style.use("dark_background")

#%%definitions
def gaussian_pdf(x, mu, sigma):
    """
        - function defining a gaussian normal distribution
    """    
    y = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))
    return y
def lc_sim(
    t:np.ndarray,
    t_peak:float, f_peak:float,
    stretch0:float, stretch1:float, stretch2:float,
    lbda:float=1.0,
    noiselevel:float=0.0,
    ) -> np.ndarray:
    """
        - function to define a very simplistic phenomenological LC simulation
    """
    f = (gaussian_pdf(t, t_peak - stretch0/2, stretch1) + gaussian_pdf(t, t_peak + stretch0/2, stretch2))
    f = f_peak * f / np.max(f) + noiselevel * np.random.randn(*t.shape)
    f *= lbda
    return f
def sin_sim(
    t:np.array,
    f_peak:float,
    p:float, offset:float=0.,
    noiselevel:float=0.0
    ) -> float:
    """
        - function to evaluate a sin with period `p` and `offset`
    """
    f = f_peak * np.sin(t * 2*np.pi/p + offset) + noiselevel * np.random.randn(*t.shape)
    return f
def simulate(
    nobjects:int=6,
    opt:Literal["lc","sin"]="lc"
    ):
    res = 500
    x = np.sort(np.random.choice(np.linspace(-50,100,res), size=(nobjects,res)), axis=1)
    theta_options = np.arange(0.2, 4, 0.2)
    theta = np.sort(np.random.choice(theta_options, size=nobjects, replace=False))
    
    if opt == "lc":
        t_peak = np.linspace(0,40,nobjects) * 0
        y           = np.array([*map(lambda i: lc_sim(x[i], t_peak=t_peak[i], f_peak=20, lbda=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=1.0), range(nobjects))])
        y_nonoise   = np.array([*map(lambda i: lc_sim(x[i], t_peak=t_peak[i], f_peak=20, lbda=theta[i], stretch0=5, stretch1=15, stretch2=40, noiselevel=0.0), range(nobjects))])
    else:
        theta *= 10
        y           = np.array([*map(lambda i: sin_sim(x[i], f_peak=1, p=theta[i], offset=0.0, noiselevel=0.1), range(nobjects))])
        y_nonoise   = np.array([*map(lambda i: sin_sim(x[i], f_peak=1, p=theta[i], offset=0.0, noiselevel=0.0), range(nobjects))])

    df_raw = pl.from_dict(dict(
            period=np.repeat(theta.reshape(-1,1), x.shape[1], axis=1).flatten(),
            time=x.flatten(),
            amplitude=y.flatten(),
            amplitude_e=np.nan,
            processing="raw",
        ))
    df_pro = pl.from_dict(dict(
            period=np.repeat(theta.reshape(-1,1), x.shape[1], axis=1).flatten(),
            time=x.flatten(),
            amplitude=y_nonoise.flatten(),
            amplitude_e=np.nan,
            processing="nonoise",
        ))
    
    df = pl.concat([df_raw, df_pro])
    

    return df

#%%data loading
#passbands
df_pb = pl.read_csv("../data/passband_specs.csv")
passbands = list(df_pb["name"])
pb_mappings = dict(zip(df_pb["wavelength"], df_pb.select(pl.exclude("wavelength")).to_numpy()))

#LCs
fnames = sorted(glob.glob("../data/*_elasticc.csv"))
fnames = np.append(fnames, ["../data/lc_simulated.py", "../data/sin_simulated.py"])
print(fnames)
fname = fnames[15]

#deal with on-the-fly data generation (pseudo filenames)
if fname == "../data/lc_simulated.py":
    df = simulate(9, opt="lc")
    for t in df["period"].unique(): pb_mappings[t] = [np.round(t, 3)]
    legend = False
    thetalab = "Maximum Scale"
    xlab = "Time [d]"
    ylab = "Amplitude []"
elif fname == "../data/sin_simulated.py":
    df = simulate(9, opt="sin")
    for t in df["period"].unique(): pb_mappings[t] = [np.round(t, 3)]
    legend = False
    thetalab = "Period [s]"
    xlab = "Time [s]"
    ylab = "Amplitude []"
else:
    df = pl.read_csv(fname, comment_prefix="#")
    legend = True
    thetalab = "Wavelength [nm]"
    xlab = "MJD-min(MJD) [d]" if "mjd" in df.columns else "Period [d]"
    ylab = "m [mag]" if "mag" in df.columns else "Fluxcal []"

# df = df.drop_nans()


parts = re.split(r"[/\_\.]", fname)
survey = parts[-2]
otype = parts[-3]

df_raw = df.filter(pl.col("processing")=="raw")
df_pro = df.filter(pl.col("processing")!="raw")
theta_raw = np.sort(np.unique(df_raw[:,0]))
df_raw_p = df_raw.partition_by(df_raw.columns[0], maintain_order=True)
x_raw = [df[:,1].to_numpy().astype(np.float64) for df in df_raw_p]
x_raw = [xi - np.nanmin(xi) for xi in x_raw]
y_raw = [df[:,2].to_numpy().astype(np.float64) for df in df_raw_p]
theta_pro = np.sort(np.unique(df_pro[:,0]))
df_pro_p = df_pro.partition_by(df_pro.columns[0], maintain_order=True)
x_pro = [df[:,1].to_numpy().astype(np.float64) for df in df_pro_p]
x_pro = [xi - np.nanmin(xi) for xi in x_pro]
y_pro = [df[:,2].to_numpy().astype(np.float64) for df in df_pro_p]

# print(theta_raw.shape, len(x_raw), len(y_raw))
# print(theta_pro.shape, len(x_pro), len(y_pro))

#%%get stats
unique_thetas = np.unique(theta_raw)
thetaticks = np.round(np.linspace(np.floor(np.min(theta_raw)), np.ceil(np.max(theta_raw)), 4),0).astype(int)
xticks = np.round(np.linspace(np.floor(np.min(np.concat(x_raw))), np.ceil(np.max(np.concat(x_raw))), 4), decimals=0).astype(int)
yticks = np.round(np.linspace(np.floor(np.min(np.concat(y_raw))), np.ceil(np.max(np.concat(y_raw))), 4), decimals=0).astype(int)
# yticks = np.sort(np.append(yticks, [-10, 80]))
panelsize = np.pi/10
vmin = 300 if ".py" not in fname else None
colors = lvisu.get_colors(theta_raw, cmap="nipy_spectral", vmin=vmin)

#%%plotting
#LVisP
fig = plt.figure(figsize=(12,9))
fig.suptitle(f"{otype} ({survey})")
ax = fig.add_subplot(121)
LVPC = LVisPCanvas.LVisPCanvas(ax,
    thetaticks, xticks, yticks,
    thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
    xlimdeadzone=0.3,
    thetalabel=thetalab, xlabel=xlab, ylabel=ylab,
    thetaarrowpos_th=None, ylabpos_th=None,
    thetatickkwargs=dict(c=plt.rcParams["text.color"]), thetaticklabelkwargs=None, thetalabelkwargs=None,
    xtickkwargs=None, xticklabelkwargs=dict(textcoords="offset fontsize", xytext=(-2,0)), xlabelkwargs=dict(rotation=-90,  textcoords="offset fontsize", xytext=(-3.5,0)),
    ylabelkwargs=dict(rotation=0)
)
LVPC.scatter(theta_raw, x_raw, y_raw,
    panel_kwargs=[dict(
        y_projection_method="y",
        panelsize=panelsize,
        show_panelbounds=True, show_yticks=True
    ) for _ in theta_raw],
    scatter_kwargs=[dict(
        c=mcolors.to_hex(colors[i]), label=pb_mappings[theta_raw[i]][0],
    ) for i in range(len(theta_raw))],
)
LVPC.plot(theta_pro, x_pro, y_pro,
    plot_kwargs=[dict(lw=3, c="w") for _ in theta_pro]
)
LVPC.plot(theta_pro, x_pro, y_pro,
    plot_kwargs=[dict(c=mcolors.to_hex(colors[i])) for i in range(len(theta_pro))],
)
# if legend: ax.legend()
ax.legend()


#traditional
axt1 = fig.add_subplot(222, xlabel=xlab, ylabel=ylab)

for i in range(len(theta_raw)):
    axt1.scatter(x_raw[i], y_raw[i], c=colors[i])
for i in range(len(theta_pro)):
    axt1.plot(x_pro[i], y_pro[i], c="w", lw=3)
    axt1.plot(x_pro[i], y_pro[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
if legend: axt1.legend()

for i in range(len(theta_raw)):
# for i in range(6):
    ncols_theta = 3
    nrows_theta = int(np.ceil(len(theta_raw)/ncols_theta))
    nrows = int(2*nrows_theta)
    ncols = int(2*ncols_theta)
    pos = (1 + nrows_theta * ncols + ncols_theta) + i%ncols_theta + ncols*(i//ncols_theta)
    axt2 = fig.add_subplot(nrows, ncols, pos, xlabel=xlab, ylabel=ylab)
    # axt2 = fig.add_subplot(5, 1, i+1, xlabel="MJD-min(MJD) [d]", ylabel="Fuxcal []")
    axt2.scatter(x_raw[i], y_raw[i], c=colors[i])
    axt2.plot(x_pro[i], y_pro[i], c="w", lw=3)
    axt2.plot(x_pro[i], y_pro[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
    if legend: axt2.legend()
    
fig.tight_layout()
fig.savefig(fname.replace("./data/","./gfx/").replace(".csv",".png").replace(".py",".png"), bbox_inches="tight")

plt.show()