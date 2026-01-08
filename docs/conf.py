
#%%imports
import importlib

#%%definitions


#%%main
metadata = importlib.metadata.metadata("LStein")
author = metadata["Author"]
copyright = "2026" + "Lukas Steinwender"
description = metadata["Description"]
project = metadata["Name"]
title = metadata["Name"]
version = metadata["Version"]
release = metadata["Version"] + ""  #full release (includes alpha, beta, ...)


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