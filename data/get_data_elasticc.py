
#%%imports
from astropy.io import fits
from astropy.timeseries import LombScargleMultiband
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
import sys

#%%global definitions

#from https://github.com/lsst/tutorial-notebooks/blob/main/DP0.2/08_Truth_Tables.ipynb
passbands = ["u", "g", "r", "i", "z", "y"]
colors_passbands  = dict(u="#0c71ff", g= "#49be61", r="#c61c00", i="#ffc200", z="#f341a2", Y="#5d0000")
markers_passbands = dict(u="o", g= "^", r="v", i="s", z="*", Y="p")

assert len(sys.argv) > 1, "This script needs to be called as follows: `python3 get_data_elasticc.py fname [pmin] [pmax] [objidx]`" 
fname = sys.argv[1]
pmin = eval(sys.argv[2]) if len(sys.argv) > 2 else None
pmax = eval(sys.argv[3]) if len(sys.argv) > 3 else None
objidx = eval(sys.argv[4]) if len(sys.argv) > 4 else None
print(fname, pmin, pmax, objidx)

fmax = 1/pmin if pmin is not None else None
fmin = 1/pmax if pmax is not None else None

#%%
# elasticc_mapping = pd.read_csv("~/Downloads/elasticc_origmap.txt", sep=r"\s+", comment="#", header=None).to_dict("split")
# elasticc_mapping = {kv[0]:kv[1] for kv in elasticc_mapping["data"]}
# print(elasticc_mapping)

# df_head = pd.read_csv("~/Downloads/elasticc_objects_nonia_truth.csv", nrows=1000000)
# df_phot = pl.read_csv("~/Downloads/elasticc_alerts_nonia_truth.csv", separator=",", n_rows=10000)
# df_phot = df_phot.rename({c:c.strip() for c in df_phot.columns})
# df_phot = df_phot.with_columns(
#     pl.col(pl.Utf8).str.strip_chars(" ")
# ).with_columns(
#     pl.col("SourceID").cast(pl.Utf8),
#     pl.col("MJD", "TRUE_GENMAG").cast(pl.Float64),
#     pl.col("DETECT", "TRUE_GENTYPE").cast(int),
# )
# print(df_phot.columns)
# print(df_head.columns)
# display(df_head["GENTYPE"].value_counts())
# display(df_head["SNID"].value_counts())
# display(df_phot["SourceID"].value_counts())
#%%
hdul = fits.open(fname)

df = pl.DataFrame(
    [list(hdul[1].data[c]) for c in hdul[1].columns.names],
    schema=list(hdul[1].columns.names),
)
hdul.close()
# for c in sorted(df.columns): print(c)

df = df.filter(pl.col("PHOTFLAG")==0).with_columns(pl.col("BAND").cast(pl.Categorical))
# print(df.columns)
# print(df["PHOTPROB"].value_counts())
# print(df["PHOTFLAG"].value_counts())
idxs = np.where(df["MJD"]==-777)[0]
dfs = [df[i1:i2].filter(pl.col("MJD") != -777) for i1, i2 in zip(np.append([0], idxs[:-1]), idxs)]

oidxs = np.array([objidx]).flatten() if objidx is not None else np.random.randint(0,len(dfs), 20)
# for oidx in range(0,5):
for oidx in oidxs:
    dfo = dfs[oidx]
    if len(dfo) > 10 and dfo["BAND"].n_unique() >=6:
        LS = LombScargleMultiband(dfo["MJD"], dfo["FLUXCAL"], dfo["BAND"])
        f, p = LS.autopower(minimum_frequency=fmin, maximum_frequency=fmax)
        f = f[np.argmin(p)]
        # print(f)

        #saving the data
        dfo = dfo.with_columns(
            (pl.col("FLUXCAL")/pl.col("FLUXCALERR")).alias("s2n"),
            ((pl.col("MJD")/(1/f)).mod(1) * 1/f).alias("period"),
            (pl.col("MJD")-pl.col("MJD").min()).alias("deltamjd"),
            pl.lit(oidx).alias("oid"),
        )
        fname_save = fname.replace("-Templates", "").replace("SALT3", "").lower()
        dfo = dfo.rename({c:c.lower() for c in dfo.columns})
        # if objidx is not None: dfo.write_csv(f"./{objidx:04d}_{fname_save.replace('.fits.gz','_elasticc.csv')}")

        #plotting
        fig = plt.figure()
        fig.suptitle(f"{oidx:04d}")
        ax1 = fig.add_subplot(121, xlabel="MJD [d]", ylabel="FLUXCAL []")
        ax2 = fig.add_subplot(122, xlabel="Period [d]", ylabel="FLUXCAL []")
        for b in dfo["band"].unique():
            dfo_b = dfo.filter(
                (pl.col("band")==b),
                # (pl.col("FLUXCAL")/pl.col("FLUXCALERR") > 5)
            )
            ax1.errorbar(dfo_b["deltamjd"], dfo_b["fluxcal"] - 0*dfo_b["rdnoise"] - 0*dfo_b["sim_fluxcal_hosterr"],
                yerr=dfo_b["fluxcalerr"],
                c=colors_passbands[b], marker=markers_passbands[b], ls="",
                ecolor=(*mcolors.to_rgb(colors_passbands[b]), 0.3),
                label=b,
            )
            ax2.errorbar(dfo_b["period"], dfo_b["fluxcal"],
                yerr=dfo_b["fluxcalerr"],
                c=colors_passbands[b], marker=markers_passbands[b], ls="",
                ecolor=(*mcolors.to_rgb(colors_passbands[b]), 0.3),
                label=b,             
            )
        ax1.legend()
        fig.tight_layout()
    else:
        print("Not enough data")
plt.show()