#configuration for sphinx document builder
# details: https://www.sphinx-doc.org/en/master/usage/configuration.html


#%%imports
import importlib
import os
import sys


#%%definitions
def add_paths():
    """
        - function to add directories to be documented to the path
        - make sure to make the path absolute!
    """
    sys.path.insert(0, os.path.abspath("../src"))   #default for uv `project/src/pkg` layout

    return

def set_metadata():
    #expose to global scope
    global author
    global copyright
    global description
    global project
    global title
    global version
    global release    

    #read metadata from pyproject.toml
    metadata = importlib.metadata.metadata("LStein")

    #set metadata
    author = metadata["Author"]
    copyright = f"2026, Lukas Steinwender"
    description = metadata["Description"]
    project = metadata["Name"]
    title = metadata["Name"]
    version = metadata["Version"]
    release = metadata["Version"] + ""  #full release (includes alpha, beta, ...)

    return

#%%main
add_paths()
set_metadata()

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