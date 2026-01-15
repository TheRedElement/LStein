API Reference
=============

.. currentmodule:: lstein.lstein

LSteinCanvas
------------

.. autosummary::
    :signatures: none
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


LSteinPanel
-----------

.. autosummary::
    :signatures: none
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

Backends
--------

`matplotlib <https://matplotlib.org/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autosummary::
    :signatures: none
    :recursive:
    :toctree: tocLSP/

    LSteinMPL
    LSteinMPL.__init__
    LSteinMPL.add_thetaaxis
    LSteinMPL.add_xaxis
    LSteinMPL.add_yaxis
    LSteinMPL.add_ylabel
    LSteinMPL.show   

`Plotly <https://plotly.com/python/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
