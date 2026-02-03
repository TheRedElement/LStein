#configuration for sphinx document builder
# details: https://www.sphinx-doc.org/en/master/usage/configuration.html


#%%imports
import importlib
import os
import sys
import shutil


#%%definitions
def add_paths():
    """adds directories to be documented to the path
        
    - function to add directories to be documented to the path
    - all paths are relative to `conf.py`
    - make sure to make the path absolute!
    """
    global html_static_path

    sys.path.insert(0, os.path.abspath("../src"))   #default for uv `project/src/pkg` layout

    #paths to custom static files
    #loaded after builtin static files => will override => `default.css` overrides internal `default.css`
    html_static_path = [
        "assets",
        os.path.abspath("../gfx"),
        os.path.abspath("../demo"),
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

def override_style():
    """overrides `html_theme` with custom theme
        
    - overrides `html_theme` with custom theme
    - depends on `/docs/_static/custom.css`

    """
    global html_theme
    global html_static_path
    global html_css_files
    global html_theme_options

    c_bg = "#000000"
    c_body_text = "#ffffff"
    c_link = "#c80000"
    c_hover = "#c94242"
    c_code_bg = "#414141"
    html_theme = "alabaster"
    html_theme_options = {
        # "description": description,
        "logo": "_gfx/lstein_logo.svg",
        "github_user": "TheRedElement",
        "github_repo": "LStein",
        "github_banner": True,
        "github_button": True,
        "sidebar_collapse": True,
        "body_text": c_body_text,
        "gray_1": c_body_text,
        "gray_2": "#646464",
        "gray_3": "#535353",
        "link_hover": c_hover,
        "link": c_link,
        "pre_bg": c_code_bg,
    }
    html_static_path = ["_static"]
    html_css_files = ["custom.css"]
    return
# def modify_docstrings(app, what, name, obj, options, lines):
#     """
#         - function to apply modifications to docstrings
#     """

#     #mapping to downgrade headings
#     heading_mapping = {
#         "=": "~",
#         "-": "~",
#         "~": '~',
#         "^": '~',
#     }

#     for i, line in enumerate(lines[:-1]):
#         next_line = lines[i + 1]
        
#         #downgrade headings
#         for char in heading_mapping.keys():
#             if set(next_line) == {char} and len(next_line) >= len(line):
#                 lines[i + 1] = heading_mapping[char] * len(next_line)

#     return


#%%sphinx internal functions
def setup(app):
    """makes some global setups pre build
    """

    paths = {
        # "../gfx":"_gfx",
    }
    #copy some paths to html
    for path in paths.keys():
        src = os.path.abspath(path)
        dst = os.path.join(app.outdir, paths[path])
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # app.connect("autodoc-process-docstring", modify_docstrings)


    
#%%main
#extensions
extensions = [
    #sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    #custom extensions
    "nbsphinx",
    "sphinx_copybutton",    #adds copy button to all code-blocks
    # "myst_parser",          #markdown support
    "myst_nb",              #ipynb support (only use when `myst_parser` not used)
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst_nb",
    ".ipynb": "myst_nb",
}

autodoc_default_options = {
    "members": True,
    "member-order": "alphabetical",
}
# autoclass_content = "both"
autosummary_generate = True

#myst_parser extensions
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

#executions (order matters)
add_paths()
set_metadata()
override_style()

todo_include_todos = True

