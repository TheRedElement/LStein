#!/bin/bash

#   - building blocks that are used in multiple files of the project
#   - use by simply sourcing and calling the functions
#
#   ```bash
#       source /fred/oz300/lukas/SNClassification/code/_projectbuildingblocks.sh
#   ``` 

#%%definitions
function_template() {
    #   - function description
    #
    #Usage
    #-----
    #   ```bash
    #   ```
    #
    #Parameters
    #----------
    #   - $1 (`local name`)
    #       - type
    #       - description
    #   - $3+ (`arr1` `arr2` ...)
    #       - (types) variable arguments
    #
    #Example
    #-------
    #   ```bash
    #   ```
    #
    #Output
    #------
    #   ```bash
    #   ```
    #Comments
    #--------
    #

    
}
#prevent direct execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This is a library file. Source it in your scripts: source <path/to/_projectbuildingblocks.sh>"
fi