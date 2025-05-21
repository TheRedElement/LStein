
#%%imports
from astropy.io import fits
from astropy.timeseries import LombScargleMultiband
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import sys
import warnings

# warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#%%definitions
def gaussian_process_interpolate_lc(
    x:np.ndarray, y:np.ndarray, y_err:np.ndarray,
    n_interp:int=100,
    ) -> pl.DataFrame:
    """
        - function to execute gaussion process interpolation on one single lc
    """
    
    #default parameters
    kernel = kernels.Matern(x.std(), (1e-2*x.std(),1e2*x.std()), nu=3/2)
    kernel *= kernels.ConstantKernel(y.var(), (1e-3*y.var(),1e3*y.var()))
    kernel += kernels.WhiteKernel(2*y.std(), noise_level_bounds=(1e-6*y.std(),y.std()))

    #fit gaussian process
    GPR = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=0, n_targets=1,
        alpha=y_err.flatten()**2,
        normalize_y=False,
    )
    GPR.fit(x, y)

    #make prediction (interpolate)
    x_pred = np.linspace(x.min(), x.max(), n_interp).reshape(-1,1)
    y_pred, y_pred_std = GPR.predict(x_pred, return_std=True)

    #save to dataframe
    df_lc_gp = pl.DataFrame(dict(
        time=x_pred.flatten(),
        fluxcal=y_pred,
        fluxcalerr=y_pred_std,
        processing="gp",
    ))

    return df_lc_gp

#%%global definitions
df_pb = pl.read_csv("passband_specs.csv").filter(pl.col("mission")=="lsst")
passbands = list(df_pb["name"]) + ["-"]
pb_mappings = dict(zip(df_pb["name"], df_pb.select(pl.exclude("name")).to_numpy()))

assert len(sys.argv) > 1, "This script needs to be called as follows: `python3 get_data_elasticc.py fname [pmin] [pmax] [objidx]`" 
fname = sys.argv[1]
pmin = eval(sys.argv[2]) if len(sys.argv) > 2 else None
pmax = eval(sys.argv[3]) if len(sys.argv) > 3 else None
objidx = eval(sys.argv[4]) if len(sys.argv) > 4 else None
print(fname, pmin, pmax, objidx)

fmax = 1/pmin if pmin is not None else None
fmin = 1/pmax if pmax is not None else None

#%%
hdul = fits.open(fname)

df = pl.DataFrame(
    [list(hdul[1].data[c]) for c in hdul[1].columns.names],
    schema=list(hdul[1].columns.names),
)
hdul.close()

df = df.filter(pl.col("PHOTFLAG")==0).with_columns(pl.col("BAND").cast(pl.Enum(passbands)))
idxs = np.where(df["MJD"]==-777)[0]
dfs = [df[i1:i2].filter(pl.col("MJD") != -777) for i1, i2 in zip(np.append([0], idxs[:-1]), idxs)]

oidxs = np.array([objidx]).flatten() if objidx is not None else np.random.randint(0,len(dfs), 20)
# for oidx in range(0,5):
for oidx in oidxs:
    dfo = dfs[oidx]
    dfs = []  #list of output dataframes
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
            pl.lit("raw").alias("processing"),
        )
        fname_save = fname.replace("-Templates", "").replace("-SALT3", "").lower()
        dfo = dfo.rename({c:c.lower() for c in dfo.columns})

        #plotting
        fig = plt.figure()
        fig.suptitle(f"{oidx:04d}")
        ax1 = fig.add_subplot(121, xlabel="MJD [d]", ylabel="FLUXCAL []")
        ax2 = fig.add_subplot(122, xlabel="Period [d]", ylabel="FLUXCAL []")
        for b in dfo["band"].unique():
            dfo_b = dfo.filter(
                (pl.col("band")==b),
                # (pl.col("FLUXCAL")/pl.col("FLUXCALERR") > 5)
            ).with_columns(
                pl.col(pl.Float32).cast(pl.Float64),
            )

            #gaussian process interpolation
            ##decide what to use as x-values
            if fmin is None:    #if transient
                dfo_b = dfo_b.with_columns((pl.col("mjd")-pl.col("mjd").min()).alias("time"))
            else:               #if periodic variable
                dfo_b = dfo_b.with_columns(pl.col("period").alias("time"))
            
            ##exec interpolation
            dfo_b = dfo_b.sort("time")
            df_gp = gaussian_process_interpolate_lc(
                dfo_b.select(pl.col("time")).to_numpy(),
                dfo_b.select(pl.col("fluxcal")).to_numpy(),
                dfo_b.select(pl.col("fluxcalerr")).to_numpy(),
            ).with_columns(
                pl.lit(b, pl.Enum(passbands)).alias("band"),
                pl.col(pl.Float32).cast(pl.Float64),
            ).select(pl.col("band","time","fluxcal","fluxcalerr","processing"))
            
            #add current passband to output dataframes (one for each passband)
            dfs.append(pl.concat([
                dfo_b.select(pl.col("band","time","fluxcal","fluxcalerr","processing")).with_columns(pl.lit("raw").alias("processing")),
                df_gp,
            ], how="vertical"))

            #testplot
            ax1.errorbar(dfo_b["time"], dfo_b["fluxcal"] - 0*dfo_b["rdnoise"] - 0*dfo_b["sim_fluxcal_hosterr"],
            # ax1.errorbar(dfo_b["deltamjd"], dfo_b["fluxcal"] - 0*dfo_b["rdnoise"] - 0*dfo_b["sim_fluxcal_hosterr"],
                yerr=dfo_b["fluxcalerr"],
                c=pb_mappings[b][1], marker=pb_mappings[b][2], ls="",
                ecolor=(*mcolors.to_rgb(pb_mappings[b][1]), 0.3),
                label=b,
            )
            ax2.errorbar(dfo_b["period"], dfo_b["fluxcal"],
                yerr=dfo_b["fluxcalerr"],
                c=pb_mappings[b][1], marker=pb_mappings[b][2], ls="",
                ecolor=(*mcolors.to_rgb(pb_mappings[b][1]), 0.3),
                label=b,             
            )
            ax1.plot(df_gp["time"], df_gp["fluxcal"],
                c=pb_mappings[b][1], ls="-",
            )
            ax1.fill_between(df_gp["time"], df_gp["fluxcal"]-df_gp["fluxcalerr"], df_gp["fluxcal"]+df_gp["fluxcalerr"],
                color=pb_mappings[b][1], alpha=0.3,
            )
        ax1.legend()
        fig.tight_layout()
        plt.close()

        #saving
        df_out = pl.concat(dfs) #merge passbands
        df_out = df_out.with_columns(
            pl.col("band").cast(pl.Utf8).replace(df_pb["name"], df_pb["wavelength"]).cast(pl.Float64)
        )
        if fmin is None:    df_out = df_out.rename({"time":"mjd"})
        else:               df_out = df_out.rename({"time":"period"})
        if objidx is not None:
            s_fn = f"./{objidx:04d}_{fname_save.replace('.fits.gz','_elasticc.csv')}"
            file_doc = (
                "#band: central wavelength of the passband of the observation in nm\n"
                "#mjd: time of the observation as modified julian date in days\n" * (fmin is None) + ""
                "#period: folded  time of the object shown over one period in days\n" * (fmin is not None) + ""
                "#fluxcal: simulated observed flux (from SNANA)\n"
                "#fluxcalerr: error of fluxcal\n"
                "#processing: which processing was done to get the datapoint\n"
            )
            df_out.write_csv(s_fn)
            with open(s_fn, "r+") as f:
                content = f.read()
                content = file_doc + content
                f.seek(0,0)
                f.write(content)

    else:
        print("Not enough data")


plt.show()