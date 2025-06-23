
#%%imports
import glob
import importlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import polars as pl
import re
import sys

from LStein import LSteinCanvas, utils as lsu, makedata as md

importlib.reload(LSteinCanvas)
plt.style.use("dark_background")

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
# yticks = np.sort(np.append(yticks, [-10, 80]))
panelsize = np.pi/8
vmin = 300 if ".py" not in fname else 0
colors = lsu.get_colors(theta_raw, cmap="nipy_spectral", vmin=vmin)

#%%plotting
#LStein
fig = plt.figure(figsize=(12,9))
fig.suptitle(f"{otype} ({survey})")
ax = fig.add_subplot(121)
LVPC = LSteinCanvas.LSteinCanvas(ax,
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
        y_projection_method="theta",
        panelsize=panelsize,
        show_panelbounds=True, show_yticks=True
    ) for _ in theta_raw],
    sctr_kwargs=[dict(
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
    markers, caps, bars = axt1.errorbar(x_raw[i], y_raw[i], yerr=y_raw_e[i], c=colors[i], ls="", marker="o")
    for bar in bars: bar.set_alpha(0.1)
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
    markers, caps, bars = axt2.errorbar(x_raw[i], y_raw[i],yerr=y_raw_e[i], c=colors[i], ls="", marker="o")
    for bar in bars: bar.set_alpha(0.1)
    axt2.plot(x_pro[i], y_pro[i], c="w", lw=3)
    axt2.plot(x_pro[i], y_pro[i], c=colors[i], label=f"{pb_mappings[theta_raw[i]][0]} ({int(np.round(theta_raw[i], decimals=0))} nm)")
    if legend: axt2.legend()
    
fig.tight_layout()
fig.savefig(fname.replace("./data/","./gfx/").replace(".csv",".png").replace(".py",".png"), bbox_inches="tight")

plt.show()