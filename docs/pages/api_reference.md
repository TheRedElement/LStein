<!-- NOTE: :recursive: -> show all children as well (to display all methods when summarizing class) -->


# API Reference

```{eval-rst}
.. currentmodule:: lstein.lstein
```

## LSteinCanvas

```{eval-rst}
.. autosummary::
    :signatures: short
    :recursive:
    :toctree: tocLSC/

    LSteinCanvas
    LSteinCanvas.__init__
    LSteinCanvas.add_panel
    LSteinCanvas.compute_thetaaxis
    LSteinCanvas.compute_xaxis
    LSteinCanvas.compute_ylabel
    LSteinCanvas.get_panel
    LSteinCanvas.get_thetas
    LSteinCanvas.plot
    LSteinCanvas.reset
```

## LSteinPanel

```{eval-rst}
.. autosummary::
    :signatures: none
    :recursive:
    :toctree: tocLSP/

    LSteinPanel
    LSteinPanel.__init__
    LSteinPanel.apply_axis_limits
    LSteinPanel.get_rbounds
    LSteinPanel.get_thetabounds
    LSteinPanel.get_yticks
    LSteinPanel.plot
    LSteinPanel.project_xy
    LSteinPanel.project_xy_theta
    LSteinPanel.project_xy_y
```

## Backends

### [matplotlib](https://matplotlib.org/)

```{eval-rst}
.. autosummary::
    :signatures: none
    :recursive:
    :toctree: tocMPL/

    LSteinMPL
    LSteinMPL.__init__
    LSteinMPL.add_thetaaxis
    LSteinMPL.add_xaxis
    LSteinMPL.add_yaxis
    LSteinMPL.add_ylabel
    LSteinMPL.show
```

### [Plotly](https://plotly.com/python/)

```{eval-rst}
.. autosummary::
    :signatures: none
    :recursive:
    :toctree: tocPlotly/

    LSteinPlotly
    LSteinPlotly.__init__
    LSteinPlotly.add_thetaaxis
    LSteinPlotly.add_xaxis
    LSteinPlotly.add_yaxis
    LSteinPlotly.add_ylabel
    LSteinPlotly.show
    LSteinPlotly.translate_kwargs
```

## Utils
```{eval-rst}
.. currentmodule:: lstein
```

```{eval-rst}
.. autosummary::
    :signatures: none
    :recursive:
    :toctree: tocUtils/

    utils.cart2polar
    utils.correct_labelrotation
    utils.get_colors
    utils.minmaxscale
    utils.polar2cart
```