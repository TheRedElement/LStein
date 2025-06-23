
#%%imports
import LStein
import toml


#%%main
#get dependencies
with open("requirements.txt", "r") as f:
    deps = [l.rstrip("\n") for l in f.readlines()]

#load existing pyproject.toml
data = toml.load("pyproject.toml")

#update pyproject.toml
data["build-system"]["requires"] = ["hatchling >= 1.26"]
data["build-system"]["build-backend"] = "hatchling.build"

data["project"]["name"] = LStein.__modulename__
data["project"]["version"] = LStein.__version__
data["project"]["description"] = "repository of some useful code snippets in various programming languages."
data["project"]["readme"] = "README.md"
data["project"]["requires-python"] = ">=3.13"
data["project"]["classifiers"] = ["Programming Language :: Python :: 3", "Operating System :: OS Independent",]
data["project"]["license"] = "MIT"
data["project"]["license-files"] = ["LICEN[CS]E*",]
data["project"]["dependencies"] = deps
data["project"]["authors"] = [{"name":LStein.__author__, "email":LStein.__author_email__},]
data["project"]["maintaners"] = [{"name":LStein.__maintainer__, "email":LStein.__maintainer_email__},]
data["project"]["urls"] = {"Homepage":LStein.__url__, "Issues":LStein.__url__+"/issues"}

data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] = [LStein.__modulename__]

#save updated version
with open("pyproject.toml", "w") as f:
    toml.dump(data, f)
