
WELCOME
=======

.. image:: /_gfx/lstein_logo.svg

.. :hidden: so it's just in navigation
.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:

   pages/*

.. local table of contents
.. contents:: Table of Contents
   :local:
   :depth: 1

.. warning::
   Note, that this package is currently under development.
   Most functionalities should work, but changes will be implemented on a running basis and without notice.
   No test have been performed yet.


Reference
---------
If you find `LStein` useful in your work we would appreciate it if you cite its paper:

.. todo::
   
   paper coming soon, in the meantime you can use this citation once published

.. code-block:: bibtex

   @software{PY_Steinwender2025_lstein,
      author    = {{Steinwender}, Lukas},
      title     = {LStein: Linking Series to envision information neatly},
      month     = Jul,
      year      = 2025,
      version   = {latest},
      url       = {https://github.com/TheRedElement/LStein.git}
   }

Installation
------------

You can easily install the package using `pip <https://pypi.org/project/pip/>`__:

.. code-block:: shell

   pip3 install git+https://github.com/TheRedElement/LStein.git

Quick Start
-----------

Have a look at this notebook for a quick rundown: `Quickstart <https://github.com/TheRedElement/LStein/blob/main/demo/quickstart.ipynb>`__.
More tutorals can be found in `Tutorials <pages/tutorals.rst>`__.

Data For Testing
----------------

Data used for `Tutorials <pages/tutorals.rst>`__ can be found in `data/ <https://github.com/TheRedElement/LStein/blob/main/data/>`__.
There are also a few other datasets so feel free to have a play around.
Each dataset is a `.csv` file with the following columns:

=====================   ===========
Column                  Description
=====================   ===========
:math:`\theta-values`   values to be plotted as azimuthal offset of the panel
:math:`x`-values        values to be plotted radially
:math:`y`-values        values to be plotted as an azimuthal offset constraint to a circle-sector
:math:`y`-errors        errors assigned to :math:`y`-values
`processing context`    which processing was used
=====================   ===========


`Tutorials <pages/tutorals.rst>`__ using the data will behave as follows:

1. take the first 3 columns (in order) as :math:`\theta`-, :math:`x`-, :math:`y`-values
2. take the column names as axis-labels
3. plot a scatter for `processing context="raw"`
4. plot a line for `processing context!="raw"`

You can try your own data as well, but make sure to

1. follow the above-mentioned conventions
2. add at least one row with `processing context!="raw"`
    1. if you just have raw data, you can always just duplicate the rows and change half of the rows to `processing context!="raw"`


Example Plots
-------------

Known Bugs
----------

* `y_projection_method="theta"` goes haywire for huge x-values (for sure :math:`\ge10000`)
    * the reason is the necessity to compute :math:`\tan` and :math:`arc\tan` when converting back and forth between coordinate systems
    * workarounds
        * formulate your series relative to some value so you remain in a reasonable range
        * use `y_projection_method="y"`