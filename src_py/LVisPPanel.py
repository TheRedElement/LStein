#%%imports
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
from typing import Any, List, Tuple

import utils as lvisu

#%%classes
class LVisPPanel:

    def __init__(self,
        yticks:Tuple[List[float],List[Any]],
        thetaplotlims:Tuple[float,float]=None, panelsize:float=np.pi/8,
        panelbounds:bool=False, ygrid:bool=True,
        ygridkwargs:dict=None,
        yticklabelkwargs:dict=None,
        ylabelkwargs:dict=None,
        panelboundskwargs:dict=None,
        ):
        
        self.yticks         = (yticks, yticks) if isinstance(yticks, (list, np.ndarray)) else yticks
        
        self.thetaplotlims  = thetaguidelims if thetaplotlims is None else thetaplotlims
        self.panelsize      = panelsize

        self.panelbounds    = panelbounds
        self.ygrid          = ygrid

        self.ygridkwargs            = dict(c=plt.rcParams["grid.color"], ls=plt.rcParams["grid.linestyle"], lw=plt.rcParams["grid.linewidth"]) if ygridkwargs is None else ygridkwargs
        self.yticklabelkwargs       = dict() if yticklabelkwargs is None else yticklabelkwargs
        self.ylabelkwargs           = dict() if ylabelkwargs is None else ylabelkwargs
        self.panelboundskwargs      = dict() if panelboundskwargs is None else panelboundskwargs

        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" + ", ".join([f"{attr}={val}" for attr, val in self.__dict__.items()]) + ")"

#%%
from LVisPCanvas import LVisPCanvas
panelsize = np.pi/5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()

LVPC = LVisPCanvas(
    [-3,0,7], [-20,0,100], [0, 1, 10],
    thetaguidelims=(-np.pi/2,np.pi/2), thetaplotlims=(-np.pi/2+panelsize/2,np.pi/2-panelsize/2),
    xlimdeadzone=0.3,
    panelsize=panelsize,
    thetalabel=r"$\theta$-label", xlabel=r"$x$-label", ylabel=r"$y$-label",
    thetaarrowlength=np.pi/2,
    thetatickkwargs=None, thetaticklabelkwargs=None, thetalabelkwargs=None,
    xtickkwargs=None, xticklabelkwargs=None, xlabelkwargs=None,
)
LVPP = LVisPPanel(
    
)
LVPC.plot_LVisPCanvas(ax)

fig.tight_layout()
