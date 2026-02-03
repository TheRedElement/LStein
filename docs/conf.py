#configuration for sphinx document builder
# details: https://www.sphinx-doc.org/en/master/usage/configuration.html


#%%imports
import importlib
import os
from pathlib import Path
import re
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
        os.path.abspath("../README.md"),
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

def readme2index():
    readme_path = "../README.md"

    with open(readme_path, "r") as readme:
        text = readme.read()
    
    # replacements = {
    #     r"\[\!WARNING\]": "{warning}",
    #     r"\[\!INFO\]": "{info}",
    #     r"^\> ": ""
    # }

    # for ex, rep in replacements.items():
    #     re.sub

    with open("../index.md", "w") as f:
        f.write(text)

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

def copy_files(app):
    """copies files before the build process starts

    - ensures that files are available for build process
    - convenience since files only have to be present in one location
    """
    root = Path("../")
    docs = Path(app.srcdir)

    #README.md => index.md
    src = root
    dst = docs
    dst.mkdir(parents=True, exist_ok=True)
    f = "README.md"
    shutil.copy2(src / f, dst / f)
    with open(dst / f, "r+") as outfile:
        #deal with some markdown conversions ([!WARNING], [!INFO], ... blocks)
        text = outfile.read()
        text = re.sub(r"^>\s", "", text)
        text = re.sub(r"\[!(\w+)\]", "\1", text)
        text = f"```{text}\n```"
        outfile.write(text)

    #graphics
    src = root / "gfx"
    dst = docs / "gfx"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ["0901_snii_elasticc.png", "1189_snia_elasticc.png", "2025_tde_elasticc.png", "lstein_logo.svg", "sin_simulated.png"]:
        shutil.copy2(src / f, dst / f)

    return
#%%sphinx internal functions
def setup(app):
    """makes some global setups pre build
    """
    app.connect("builder-inited", copy_files)


    # paths = {
    #     # "../gfx":"_gfx",
    #     # "../README.md": "pages/README.md"
    # }
    # #copy some paths to html
    # for path in paths.keys():
    #     src = os.path.abspath(path)
    #     dst = os.path.join(app.outdir, paths[path])
    #     if os.path.exists(src):
    #         if os.path.isdir(src):
    #             shutil.copytree(src, dst, dirs_exist_ok=True)
    #         else:
    #             shutil.copyfile(src, dst)

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
    "myst_parser",          #markdown support
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    # ".ipynb": "markdown",
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

