
#%%imports
import glob
import importlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import polars as pl
import re

from LStein import LSteinCanvas, utils as lsu, makedata as md

importlib.reload(LSteinCanvas)
# plt.style.use("dark_background")

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
# fname = fnames[-1]

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
    raw, pro = md.simulate(9, opt="sin")
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
xticks = xticks[::-1]
yticks = yticks[::-1]
# yticks = np.sort(np.append(yticks, [-10, 80]))
panelsize = np.pi/8
vmin = 300 if ".py" not in fname else 0
colors = lsu.get_colors(theta_raw,
    cmap="nipy_spectral",
    norm=mcolors.LogNorm,
    vmin=300, vmax=1000
)

#%%plotting
#LStein
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(121)
ax.set_title(f"SN II (ELAsTiCC)")
LSC = LSteinCanvas.LSteinCanvas(ax,
    thetaticks, xticks, yticks,
    xlimdeadzone=0.3,
    thetalabel=thetalab, xlabel=xlab, ylabel=ylab,
    thetaarrowpos_th=None, ylabpos_th=None,
    #bottom hemicircle
    thetaguidelims=(np.pi,2*np.pi), thetaplotlims=(2*np.pi-panelsize/2,np.pi+panelsize/2),
    thetatickkwargs=dict(c=plt.rcParams["text.color"]), thetaticklabelkwargs=dict(pad=0.25), thetalabelkwargs=dict(textcoords="offset fontsize", xytext=(0.0,0)),
    xtickkwargs=None, xticklabelkwargs=dict(textcoords="offset fontsize", xytext=(-1,0.3)), xlabelkwargs=dict(rotation=0,  textcoords="offset fontsize", xytext=(-2,1.5)),
    ylabelkwargs=dict(rotation=0, textcoords="offset fontsize", xytext=(-4,-9))
    # #noflip
    # thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
    # thetatickkwargs=dict(c=plt.rcParams["text.color"]), thetaticklabelkwargs=None, thetalabelkwargs=dict(textcoords="offset fontsize", xytext=(1.0,0)),
    # xtickkwargs=None, xticklabelkwargs=dict(textcoords="offset fontsize", xytext=(0.3,0)), xlabelkwargs=dict(rotation=-90,  textcoords="offset fontsize", xytext=(-2,0)),
    # ylabelkwargs=dict(rotation=0, textcoords="offset fontsize", xytext=(-3,-1))
    # #flipxy
    # thetaguidelims=(3*np.pi/2,np.pi/2), thetaplotlims=(3*np.pi/2-panelsize/2,np.pi/2+panelsize/2),
    # thetatickkwargs=dict(c=plt.rcParams["text.color"]), thetaticklabelkwargs=None, thetalabelkwargs=dict(textcoords="offset fontsize", xytext=(-1,0)),
    # xtickkwargs=None, xticklabelkwargs=dict(textcoords="offset fontsize", xytext=(0.3,0)), xlabelkwargs=dict(rotation=-90,  textcoords="offset fontsize", xytext=(2,0)),
    # ylabelkwargs=dict(rotation=0, textcoords="offset fontsize", xytext=(0,-1))
)
LSC.scatter(theta_raw, x_raw, y_raw,
    panel_kwargs=[dict(
        y_projection_method="theta",
        panelsize=panelsize,
        show_panelbounds=True, show_yticks=True
    ) for _ in theta_raw],
    sctr_kwargs=[dict(
        c=mcolors.to_hex(colors[i]), label=pb_mappings[theta_raw[i]][0],
    ) for i in range(len(theta_raw))],
)
LSC.plot(theta_pro, x_pro, y_pro,
    plot_kwargs=[dict(lw=3, c="w") for _ in theta_pro]
)
LSC.plot(theta_pro, x_pro, y_pro,
    plot_kwargs=[dict(c=mcolors.to_hex(colors[i])) for i in range(len(theta_pro))],
)
# if legend: ax.legend()
# ax.legend(loc="upper left", bbox_to_anchor=(0.88, 1.05), framealpha=0.2)
# ax.legend(loc="upper left", bbox_to_anchor=(-.06, 1.05), framealpha=0.2)
ax.legend(
    ncol=2,
    columnspacing=0.5, handletextpad=0.1, borderpad=0.0, labelspacing=0.1,
    bbox_to_anchor=(0.18, 0.2), framealpha=0.2
)


fig.tight_layout()
fig.savefig(
    # "../gfx/lstein_poster.pdf", bbox_inches="tight",
    "../gfx/lstein_poster.png", bbox_inches="tight",
    transparent=True, dpi=300
)

plt.show()