
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

You can easily install the package using `pip <https://pypi.org/project/pip/>`:

.. code-block:: shell

   pip3 install git+https://github.com/TheRedElement/LStein.git

Quick Start
-----------

Have a look at this notebook for a quick rundown: `Quickstart <https://github.com/TheRedElement/LStein/blob/main/demo/quickstart.ipynb>`.
More tutorals can be found in `Tutorials <pages/tutorals.rst>`

Example Plots
-------------

Known Bugs
----------
* `y_projection_method="theta"` goes haywire for huge x-values (for sure $\ge10000$)
    * the reason is the necessity to compute $\tan$ and $arc\tan$ when converting back and forth between coordinate systems
    * workarounds
        * formulate your series relative to some value so you remain in a reasonable range
        * use `y_projection_method="y"`