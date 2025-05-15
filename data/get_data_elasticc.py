
#%%imports
from astropy.io import fits
from astropy.timeseries import LombScargleMultiband
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd

#%%global definitions

#from https://github.com/lsst/tutorial-notebooks/blob/main/DP0.2/08_Truth_Tables.ipynb
passbands = ["u", "g", "r", "i", "z", "y"]
colors_passbands  = dict(u="#0c71ff", g= "#49be61", r="#c61c00", i="#ffc200", z="#f341a2", Y="#5d0000")
markers_passbands = dict(u="o", g= "^", r="v", i="s", z="*", Y="p")


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
# hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0001_PHOT.FITS.gz") #CEPH
# hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0002_PHOT.FITS.gz") #EB
# hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0003_PHOT.FITS.gz") #SNII
# hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0004_PHOT.FITS.gz") #SNIb
hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0005_PHOT.FITS.gz") #RRLyr

df = pl.DataFrame(
    [list(hdul[1].data[c]) for c in hdul[1].columns.names],
    schema=list(hdul[1].columns.names),
)
hdul.close()


df = df.filter(pl.col("PHOTFLAG")==0).with_columns(pl.col("BAND").cast(pl.Categorical))
print(df.columns)
print(df["PHOTPROB"].value_counts())
print(df["PHOTFLAG"].value_counts())
idxs = np.where(df["MJD"]==-777)[0]
dfs = [df[i1:i2].filter(pl.col("MJD") != -777) for i1, i2 in zip(np.append([0], idxs[:-1]), idxs)]

# for oidx in range(0,5):
for oidx in np.random.randint(0,len(dfs), 5):
    dfo = dfs[oidx]
    if len(dfo) > 10:
        LS = LombScargleMultiband(dfo["MJD"], dfo["FLUXCAL"], dfo["BAND"])
        # f, p = LS.autopower(minimum_frequency=1/1.2, maximum_frequency=1/0.2)
        f, p = LS.autopower()
        f = f[np.argmin(p)]
        # print(f)
        fig = plt.figure()
        ax1 = fig.add_subplot(121, xlabel="MJD [d]", ylabel="FLUXCAL []")
        ax2 = fig.add_subplot(122, xlabel="Phase", ylabel="FLUXCAL []")
        for b in dfo["BAND"].unique():
            dfo_b = dfo.filter(
                (pl.col("BAND")==b),
                # (pl.col("FLUXCAL")/pl.col("FLUXCALERR") > 3)
            )
            ax1.scatter(dfo_b["MJD"], dfo_b["FLUXCAL"] - 0*dfo_b["RDNOISE"] - dfo_b["SIM_FLUXCAL_HOSTERR"], c=colors_passbands[b], marker=markers_passbands[b], label=b)
            ax2.scatter(dfo_b["MJD"]%(1/f), dfo_b["FLUXCAL"], c=colors_passbands[b], marker=markers_passbands[b])
        ax1.legend()
        fig.tight_layout()
    else:
        print("Not enough data")
plt.show()