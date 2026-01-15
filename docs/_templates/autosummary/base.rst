.. base template for autosummary

{{ fullname }}
{{ "=" * fullname|length }}

.. autosummary::
    :signatures: none
    :toctree: generated/

    {{ fullname }}

.. automodule:: {{ module }}
   :members: