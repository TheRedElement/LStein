
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

.. note::
   This page summarizes the most common characteristics, pitfalls etc.
   Please refer to the paper for a more detailed list and `Tutorials <pages/tutorals.rst>`__ for solutions to some known issues/missing features.

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
:math:`\theta`-values   values to be plotted as azimuthal offset of the panel
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

These some exemplary results when using `LStein`.
For more examples take a look at the paper or play around with the tutorials, specifically `Real Data <https://github.com/TheRedElement/LStein/blob/main/demo/real_data.ipynb>`__.

.. raw:: HTML
   
   <div style="display: flex">
      <img src="/_gfx/1189_snia_elasticc.png">
      <img src="/_gfx/2025_tde_elasticc.png">
   </div>
   <div style="display: flex">
      <img src="/_gfx/091_snii_elasticc.png">
      <img src="/_gfx/sin_simulated.png">
   </div>
   <p>
      Examples comparing `LStein` (left panels) to traditional methods (left panels).
      (a)-(c) are simulated lightcurves from `ELAsTICC <https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/>`__ (SNIa, TDE, SNII).
      (d) compares sinusoids of different frequencies.
   </p>

Advantages and Downsides
------------------------

Pros:

+ no overcrowded panels
+ similar :math:`\theta`-values (i.e., passbands) are plotted closer together
+ allows to preserve amplitude-differences across :math:`\theta`-values for same :math:`y`-values
+ allows to depict arbitrary number of :math:`\theta`-values (by means of reducing the angular size of each :math:`\theta`-panel) 
+ works for people with color-blindness due to relational display of information
+ can be applied to variety of data (not only lightcurves)
	+ examples: spectra over time, different machine learning models
+ layout entirely customizable

Cons:

- projection effects close to `xmin`
- does currently *not* support plotting errorbars
	- workaround: plot another line if you want to indicate uncertainties


Known Bugs
----------

* `y_projection_method="theta"` goes haywire for huge x-values (for sure :math:`\ge10000`)
    * the reason is the necessity to compute :math:`\tan` and :math:`arc\tan` when converting back and forth between coordinate systems
    * workarounds
        * formulate your series relative to some value so you remain in a reasonable range
        * use `y_projection_method="y"`