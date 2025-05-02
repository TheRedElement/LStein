
"""
    - building blocks that are used in multiple files of the project
    - use by simply importing

    ```python
    import importlib
    
    import _projectbuildingblocks as pbb
    importlib.reload(pbb)
    ```    
"""

#%%imports

#%%definitions
class ClassTemplate:
    """
        - class description

        Attributes
        ----------
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
        
        Infered Attributes
        ------------------
            - 

        Methods
        -------
            - 

        Dependencies
        ------------
            -

        Comments
        --------
    """
    def __init__(self,
        verbose:int=0
        ) -> None:
        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'            
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))
    
    def __call__(self, *args, **kwds):
        pass
    
def function_template():
    """
        - function description

        Parameters
        ----------
            - `param_name`
                - `type`, <optional>
                - description
                - the default is `<default>`


        Raises
        ------
            - `error_type`
                - 

        Returns
        -------
            - `varable_name`
                - `type`, <optional>
                - description
                
        Dependencies
        ------------

        Comments
        --------
    """
    return
