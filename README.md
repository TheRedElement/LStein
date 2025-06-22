# LStein: Linking Series to envision information neatly

> [!WARNING]
> Note, that this package is currently under development.
> Most functionalities should work, but changes will be implemented on a running basis and without notice.
> No test have been performed yet.

## Installation
For now, it is easiest if you just clone this repo:

```shell
git clone https://github.com/TheRedElement/LStein.git
```

When using in [Python](https://www.python.org/) simply add it to your python-path:

```python
import sys
sys.path.append("<path/to/cloned/LStein/repo">)
from LStein.LSteinCanvas import LSteinCanvas
```

The package was developed using [Python 3.13.3](https://www.python.org/downloads/release/python-3133/).

## Reference
If you use this code in your work please use this entry in your bibliography (for now):

```latex
@software{PY_Steinwender2020_lstein,
	author    = {{Steinwender}, Lukas},
	title     = {LStein: Linking Series to envision information neatly},
	month     = Jul,
	year      = 2025,
	version   = {latest},
	url       = {https://github.com/TheRedElement/LStein.git}
}
```

## Data For Testing
Data used for [demos](./LStein_demo/LStein_demo.ipynb) and [testing](./LStein_tests/) can be found in [data/](./data/).
Each dataset is a `.csv` file with the following columns:

| Column | Description |
| :- | :- |
$\theta$-values | values to be plotted as azimuthal offset of the panel
$x$-values      | values to be plotted radially
$y$-values      | values to be plotted as an azimuthal offset constraint to a circle-sector
$y$-errors      | errors assigned to $y$-values
`processing context`  | which processing was used

The demo will behave as follows:
1. take the first 3 columns (in order) as $\theta$-, $x$-, $y$-values
2. take the column names as axis-labels
3. plot a scatter for `processing context="raw"`
4. plot a line for `processing context!="raw"`

You can try your own data as well, but make sure to
1. follow the above-mentioned conventions
2. deposit your file in [data/](./data/)
3. add at least one row with `processing context!="raw"`
    1. if you just have raw data, you can always just duplicate the rows and change half of the rows to `processing context!="raw"`

## TODO
* [testing](./LStein_tests/)

## Known Bugs
