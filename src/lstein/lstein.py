#%%imports
import matplotlib.pyplot as plt
from typing import Any, Literal

#import to expose to user
from .base.LSteinCanvas import LSteinCanvas
from .base.LSteinPanel import LSteinPanel

from .backends.matplotlib import LSteinMPL

#%%definitions
def draw(
    LSC:LSteinCanvas,
    backend:Literal["matplotlib"]="matplotlib",
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
            - `kwargs`
                - kwargs to be passed to figure instantiation methods of `backend`
        
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

    backends = ["matplotlib"]
    assert backend in backends, f"`backend` has to be one of {backends} but is {backend}"

    if backend == "matplotlib":
        fig, ax = plt.subplots(1,1, **kwargs)
        LSteinMPL(LSC).show(ax)

        return fig
    
