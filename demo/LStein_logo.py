
#%%imports
import glob
import importlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import polars as pl
import re

from lstein import lstein, utils as lsu, makedata as md

importlib.reload(lstein)
importlib.reload(md)

theme = "dark"
if theme == "dark": plt.style.use("dark_background")
plt.rcParams["savefig.transparent"] = True

np.random.seed(0)
#%%definitions

#%%data loading
#passbands
df_pb = pl.read_csv("../data/passband_specs.csv")
passbands = list(df_pb["name"])
pb_mappings = dict(zip(df_pb["wavelength"], df_pb.select(pl.exclude("wavelength")).to_numpy()))

#LCs
fnames = sorted(glob.glob("../data/*_*.csv"))
fnames = np.append(fnames, ["../data/lc_simulated.py", "../data/sin_simulated.py"])
print(fnames)
fname = fnames[3]   #snib
fname = fnames[7]   #snii
# fname = fnames[11]   #snia
# fname = fnames[21]   #tde
fname = fnames[-1]

#deal with on-the-fly data generation (pseudo filenames)
if fname == "../data/lc_simulated.py":
    raw, pro = md.simulate(9, opt="lc")
    df = pl.concat([pl.from_dict(raw), pl.from_dict(pro)])
    for t in df["period"].unique(): pb_mappings[t] = [np.round(t, 3)]
    legend = False
    thetalab = "Maximum Amplitude"
    xlab = "Time [d]"
    ylab = "Amplitude []"
elif fname == "../data/sin_simulated.py":
    raw, pro = md.simulate(6, opt="sin", theta=np.linspace(10,40,6))
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
    thetalab = "Wavelength\n[nm]"
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
x_raw = [xi - np.nanmin(xi) for xi in x_raw]
y_raw = [df[:,2].to_numpy().astype(np.float64) for df in df_raw_p]
y_raw_e = [df[:,3].to_numpy().astype(np.float64) for df in df_raw_p]
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
thetaticks = (thetaticks, [""]*len(thetaticks))
xticks = (xticks[::-1], [""]*len(xticks))
yticks = (yticks[::-1], [""]*len(yticks))
# yticks = np.sort(np.append(yticks, [-10, 80]))
panelsize = np.pi/8
vmin = 300 if ".py" not in fname else 0.9*np.min(theta_raw)
vmax = 1000 if ".py" not in fname else 1.1*np.max(theta_raw)
colors = lsu.get_colors(theta_raw,
    cmap="nipy_spectral",
    norm=mcolors.LogNorm,
    vmin=vmin, vmax=vmax,
)

#%%plotting
#LStein
thetalab = "LStein"
LSC = lstein.LSteinCanvas(
    thetaticks, xticks, yticks,
    xlimdeadzone=0.3,
    thetalabel=thetalab, 
    # xlabel=xlab, ylabel=ylab,
    thetaarrowpos_th=None, ylabpos_th=None,
    #top hemicircle
    thetaguidelims=(2*np.pi,1*np.pi), thetaplotlims=(2*np.pi-panelsize/2,1*np.pi+panelsize/2),
    thetatickkwargs=dict(c=plt.rcParams["text.color"]), thetaticklabelkwargs=dict(pad=0.15), thetalabelkwargs=dict(fontsize=28, textcoords="offset fontsize", xytext=(0.0,-0.5), weight="bold"),
    xtickkwargs=None, xticklabelkwargs=dict(textcoords="offset fontsize", xytext=(-1,-1)), xlabelkwargs=dict(rotation=0,  textcoords="offset fontsize", xytext=(-8,-2.5)),
    ylabelkwargs=dict(rotation=90, textcoords="offset fontsize", xytext=(0.5,-4))
)
LSC.plot(theta_raw, x_raw, y_raw, seriestype="scatter",
    panel_kwargs=[dict(
        y_projection_method="theta",
        panelsize=panelsize,
        show_panelbounds=True, show_yticks=False
    ) for _ in theta_raw],
    series_kwargs=[dict(
        c=mcolors.to_hex(colors[i]), label=pb_mappings[theta_raw[i]][0],
    ) for i in range(len(theta_raw))],
)
LSC.plot(theta_pro, x_pro, y_pro,
    series_kwargs=[dict(lw=3, c="w") for _ in theta_pro]
)
LSC.plot(theta_pro, x_pro, y_pro,
    series_kwargs=[dict(c=mcolors.to_hex(colors[i])) for i in range(len(theta_pro))],
)

fig = plt.figure(figsize=(14,9))
ax = fig.add_subplot(121)
ax.set_title(r"$\mathbf{L}\mathrm{inking~}\mathbf{S}\mathrm{eries~}\mathbf{t}\mathrm{o~}\mathbf{e}\mathrm{nvision ~}\mathbf{i}\mathrm{nformation~}\mathbf{n}\mathrm{eatly~}$", fontsize=19)

lstein.LSteinMPL(LSC).show(ax)

fig.tight_layout()
fig.savefig(
    # "../gfx/lstein_logo.pdf", bbox_inches="tight",
    "../gfx/lstein_logo.svg", bbox_inches="tight",
    transparent=True, dpi=300
)

plt.show()