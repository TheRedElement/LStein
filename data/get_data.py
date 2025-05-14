
#%%imports
from astropy.io import fits
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

#%%
# hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0001_PHOT.FITS.gz") #CEPH
# hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0002_PHOT.FITS.gz") #EB
# hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0003_PHOT.FITS.gz") #SNII
hdul = fits.open("~/Downloads/ELASTICC2_TRAIN_02_NONIaMODEL0-0004_PHOT.FITS.gz") #SNIb

df = pl.DataFrame(
    [list(hdul[1].data[c]) for c in hdul[1].columns.names],
    schema=list(hdul[1].columns.names),
)
hdul.close()


df = df.filter(pl.col("PHOTFLAG")==0).with_columns(pl.col("BAND").cast(pl.Categorical))
print(df.columns)
print(df["PHOTPROB"].value_counts())
idxs = np.where(df["MJD"]==-777)[0]
dfs = [df[i1:i2].filter(pl.col("MJD") != -777) for i1, i2 in zip(np.append([0], idxs[:-1]), idxs)]

for oidx in range(0,5):
    dfo = dfs[oidx]
    if len(dfo) > 10:
        LS = LombScargle(dfo["MJD"], dfo["FLUXCAL"])
        f, p = LS.autopower()
        f = f[np.argmin(p)]
        print(f)
        fig = plt.figure()
        # plt.scatter(dfo["MJD"]%(1/f), dfo["FLUXCAL"], c=dfo["BAND"].to_physical())
        plt.scatter(dfo["MJD"], dfo["FLUXCAL"], c=dfo["BAND"].to_physical())
        plt.show()
    else:
        print("Not enough data")