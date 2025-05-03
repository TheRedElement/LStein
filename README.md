# LVisP

## Naming
* LViSP: Linked Values in Series Plot
* Muhlstein
* Stoneturner

## Data For Testing
Data used for demos and testing can be found in [data/](./data/).
Each dataset is a `.csv` file with the following columns:

| Column | Description |
| :- | :- |
$\theta$-values | values to be plotted as azimuthal offset of the panel
$x$-values      | values to be plotted radially
$y$-values      | values to be plotted as an azimuthal offset constraint to a circle-sector
$y$-errors      | errors assigned to $y$-values
processing context  | which processing was used

The demo will behave as follows:
1. take the first 3 columns (in order) as $x$-, $y$-, and $\theta$-values
2. take the column names as axis-labels

You can try your own data as well, but make sure to
1. follow the above-mentioned conventions
2. update the file-name in the script

## TODO

