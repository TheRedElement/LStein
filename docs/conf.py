
#%%imports
import importlib

#%%definitions


#%%main
metadata = importlib.metadata.metadata("LStein")
print(metadata)
project = metadata["Name"]
version = metadata["Version"]
release = version
description = metadata["Description"]
