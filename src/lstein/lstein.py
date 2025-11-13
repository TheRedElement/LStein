#%%imports
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Any, Literal

#import to expose to user
from .base.LSteinCanvas import LSteinCanvas
from .base.LSteinPanel import LSteinPanel

from .backends.matplotlib import LSteinMPL
from .backends.plotly import LSteinPlotly

#%%definitions
def draw(
    LSC:LSteinCanvas,
    backend:Literal["matplotlib","plotly"]="matplotlib",
    reset:bool=True,
    **kwargs
    ) -> Any:
    """
        - convenience function to draw an existing instance of `LSteinCanvas` using some `backend`
        
        Parameters
        ----------
            - `LSC`
                - `LSteinCanvas`
                - instance of `LSteinCanvas` to be drawn
            - `backend`
                - `Literal["matplotlib"]`, optional
                - backend to use for plotting
                - the default is "matplotlib"
            - `reset`
                - `bool`, optional
                - whether to clear all flags
                - ensures that
                    - canvas is drawn (again)
                    - panels are drawn (again)
            - `kwargs`
                - kwargs to be passed to figure instantiation methods of `backend`
                    - `"matplotlib"`: `plt.subplots()`
                    - `"plotly"`: `make_subplots()`
        
        Raises
        ------


        Returns
        -------
            - `fig`
                - type depends on `backend`
                - created figure that can be displayed

        Dependencies
        ------------
            - `matplotlib`
            - `typing`

        Comments
        --------
    """

    backends = ["matplotlib", "plotly"]
    assert backend in backends, f"`backend` has to be one of {backends} but is {backend}"

    #reset draw flags
    if reset: LSC.reset()

    #plotting
    if backend == "matplotlib":
        fig, ax = plt.subplots(1,1, **kwargs)
        LSteinMPL(LSC).show(ax)
        return fig
    elif backend == "plotly":
        fig = make_subplots(
            rows=1, cols=1,
            column_widths=[1.0], row_heights=[1.0],
            **kwargs,
        )
        fig = LSteinPlotly(LSC).show(fig, 1, 1)
        return fig
    
