# LStein: Linking Series to envision information neatly

<!-- ## Naming
Alternate names:
* LuStPlot: Linking using Series Transformation
* LukasSt:
    * Learning using knowledge (by) associating sequences 
* Stonetrnr: sequential transformation of n... extration
* LSteinwender: Linking Series to extract informative novelty with ...
* LViSP: Linked Values in Series Plot

Words starting with:
* a: as, associated, association
* e: extraction, elevating, element, ensemble, example, envision
* i: information
* k: keen, keep, key, kind, knowledge
* l: learning, linking
* n: new, novelty, novel, neat, nice, "n", nearby
* o: of, observation, object, objective, on, order
* r: relation, related, round, reveal,
* s: series, separate, sequential, such,
* t: translation, transformation, through, that, to
* u: using, ultimate, unique, useful, ultra, unaesthetic -->

## Data For Testing
Data used for [demos](./LStein_demo/LStein_demo.ipynb) and [testing](./LStein_tests/) can be found in [data/](./data/).
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
* [testing](./LStein_tests/)

## Known Bugs
