.. base template for autosummary

.. display name as main title

{{ fullname }}
{{ "=" * fullname|length }}

.. generate toc
.. autosummary::
    :signatures: none
    :toctree: generated/

    {{ fullname }}

.. generate docs
.. automodule:: {{ module }}
   :members: