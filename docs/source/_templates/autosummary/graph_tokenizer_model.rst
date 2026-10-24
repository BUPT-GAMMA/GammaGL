{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:{% if objname == "GraphBERT" %} forward, manifest{% endif %}
