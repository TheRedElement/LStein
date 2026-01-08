
#%%imports
import importlib

#%%definitions


#%%main
metadata = importlib.metadata.metadata("LStein")
project = metadata["Name"]
# version = metadata["Version"]
# release = version
# description = metadata["Description"]


#extensions
extensions = [
    #sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]