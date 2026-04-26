"""LStein package

- packages to display 2.5 dimensional data in 2 dimensions
- originally developed for astronomical observations (brightness(wavelength,time))

Modules
- `lstein`
- `makedata`
- `paper_plots`
- `utils`

Subpackages
- `backends`
- `base`

"""
from importlib.metadata import metadata, version

#expose metadata
meta = metadata("lstein")
__modulename__ = "LStein"
__version__ = version("lstein")
__author__ = meta["Author"]
__author_email__ = meta["Author-email"]
__maintainer__ = meta["Maintainer"]
__maintainer_email__ = meta["Maintainer-email"]
__url__ = meta.get_all("Project-URL")