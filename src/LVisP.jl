"""
LVisP package
"""

module LVisP

#%%imports
using Dates

#metadata
const __modulename__ = "LVisP" 
const __version__ = "0.1.0"
const __author__ = "Lukas Steinwender"
const __author_email__ = ""
const __maintainer__ = "Lukas Steinwender"
const __maintainer_email__ = ""
const __url__ = "https://github.com/TheRedElement/LVisP"
const __credits__ = ""
const __last_changed__ = string(Dates.today())

#add submodules (make visible to parent module)
include("../src_jl/LVisP.jl")

#load submodules (make visible to parent module)
using .LVisP

#reexport submodules (make accesible to user)
export LVisP

end #modle