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
        - all paths are relative to `conf.py`
        - make sure to make the path absolute!
    """
    global html_static_path

    sys.path.insert(0, os.path.abspath("../src"))   #default for uv `project/src/pkg` layout

    #paths to custom static files
    #loaded after builtin static files => will override => `default.css` overrides internal `default.css`
    html_static_path = [
        "_static",
        os.path.abspath("../gfx"),
    ]
    return

def make_themes():
    global html_theme
    
    html_theme = "sphinx_rtd_theme"
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
#extensions
extensions = [
    #sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    #custom extensions
    # "sphinx_copybutton",
]


add_paths()
set_metadata()

html_static_path = [
    "assets",
    os.path.abspath("../gfx"),
]
