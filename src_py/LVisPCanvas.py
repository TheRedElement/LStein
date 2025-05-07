
#%%imports
import numpy as np
from typing import Any, List, Tuple


#%%classes
class LVisPCanvas:

    def __init__(self,
        thetalims:Tuple[float,float], xticks:Tuple[List[float],List[Any]],
        thetaguidelims:Tuple[float,float]=(0,2*np.pi), thetaplotlims:Tuple[float,float]=None, xlimdeadzone:float=0.3, panelsize:float=np.pi/8,
        # thetalabel::String="", xlabel::String="", ylabel::String="",
        # th_arrowlength::Real=pi/4,
        # panelbounds::Bool=false, ygrid::Bool=true,
        # fontsizes::Union{NamedTuple,Nothing}=nothing,
        # thetaarrowkwargs::NamedTuple=(color=:black, alpha=0.3),
        # thetaticklabelkwargs::NamedTuple=(halign=:center,),
        # thetalabelkwargs::NamedTuple=(halign=:center,),
        # xtickkwargs::NamedTuple=(linecolor=:black, linealpha=0.3,),
        # xticklabelkwargs::NamedTuple=(rotation=-90, halign=:right, valign=:bottom),
        # xlabelkwargs::NamedTuple=(halign=:center,),
        # ygridkwargs::NamedTuple=(linecolor=:black, linealpha=0.3, linestyle=:solid,),
        # yticklabelkwargs::NamedTuple=NamedTuple(),
        # ylabelkwargs::NamedTuple=NamedTuple(),
        # panelboundskwargs::NamedTuple=(linecolor=:black, linealpha=0.4, linestyle=:solid,),
        ):
        pass

    def __repr__(self):
        pass

    