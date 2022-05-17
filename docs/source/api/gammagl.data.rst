gammagl.data
====================

.. currentmodule:: gammagl.data

.. autosummary::
    :nosignatures:
    {% for cls in gammagl.data.classes %}
      {{ cls }}
    {% endfor %}

.. automodule:: gammagl.data
    :members:
    :exclude-members: Graph, HeteroGraph

    .. autoclass:: Graph
       :special-members: __cat_dim__, __inc__
       :inherited-members:

    .. autoclass:: HeteroGraph
       :special-members: __cat_dim__, __inc__
       :inherited-members:
