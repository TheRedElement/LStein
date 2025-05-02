# RepoTemplate_LuSt

Template for a repository structure.

# Parts

## [code](./code/)

Space for the code of your project.
A naming convention that integrates with my standard workflow is as follows:
* all [bash scripts](./code/bash/) are named starting with a zero-padded 2-digit integer i.e., `00_name.sh`
    * the [bash scripts](./code/bash/) are the main control of the project workflow
    * the number denotes the order of execution
* any script executed from within one of the main [bash scripts](./code/bash/) are named starting with zero-padded 3-digit integers i.e., `000_name.sh`
    * the 3-digit integer denotes the execution-order from within the main [bash scripts](./code/bash/)
        * for `03_example.sh` that would mean
            * the first script executed from within is named `031_routine.py`
            * the second script executed from within is named `032_routine.jl`
            * ...
* scripts that are not used anymore but also can't be deleted yet, follow the same naming scheme, except they get assigned `9` as their first digit
    * for main [bash scripts](./code/bash/) that means: `90_archived.sh`
    * for subscripts that means: `900_archived.jl`, `901_archived.jl`

### [bash](./code/bash/)

A directory for bash-scripts.
A slurm-script template is provided.


### [code_snippets/](./code/code_snippets/)

Submodule containing useful code snippets.

An important example is [install_requirements_python.py](./code/code_snippets/install_requirements_python.py).
This is a script to install all requirements listed in a `requirements.txt` file.
Will ignore any installs that result in errors.
Idea is to prevent errors arising from windows/linux incompitbilities.

### [data](./code/data/)

Store your datasets here.

### [execlogs](./code/execlogs/)

This is a space for any output of scripts that shall be stored for later inspection.
Examples would be slurm script outputs and errors.

### [src_jl](./code/src_jl) and [src_jl_demo](./code/src_jl_demo)

These are two directories one can use to store, demonstrate, and test `julia` modules created during the project.
* [src_jl](./code/src_jl) is hereby designated to the module creation.
* [src_jl_demo](./code/src_jl_demo) is designated for demos.
* [src_jl_tests](./code/src_jl_tests) is designated for unit tests.

### [src_py](./code/src_py/) and [src_py_demo](./code/src_py_demo/)

These are two directories one can use to store, demonstrate, and test `python` modules created during the project.
* [src_py](./code/src_py/) is hereby designated to the module creation.
* [src_py_demo](./code/src_py_demo/) is designated for testing and demos.
* [src_py_tests](./code/src_py_tests) is designated for unit tests.
    * an exemplary template for unit tests can be found in [submodulename_test_template.py](./code/src_py_tests/submodulename_test_template.py)
    * we use [pytest](https://docs.pytest.org/en/stable/) for testing python code

### [_projectbuildingblocks.*](./code/_projectbuildingblocks.*)

These are files designed to host any building blocks that are used across multiple files in the project.
Examples could be data-loading or standard plotting routines.

### [pyproject.toml](./code/pyproject.toml)

Template for a pyproject file.
Defines [src_py](./code/src_py/) as a package.

### [LICENSE](./code/LICENSE)

Template for a MIT Licence.
Only applicable to files in [code](./code/).

## [emails](./emails/)

$\LaTeX$ template for an email (to easily discuss also over equations).

## [report](./report/)

Space for your report.
Will typically end up being a $\LaTeX$ directory tree.
It is advised that graphics generated in  [code](./code/) are directly stored in a `gfx` directory within [report](./report/) to have a contained report.

## [.gitignore](./.gitignore)

A template `.gitignore` file containing ignores for
* `python`
* `julia`
* $\LaTeX$

## [.gitmodules](./.gitmodules)

Pointer to git modules included in the repo.
