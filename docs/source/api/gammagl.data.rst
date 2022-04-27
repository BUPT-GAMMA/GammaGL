gammagl.data
====================


.. currentmodule:: gammal.data
.. autosummary::
   :nosignatures:
   {% for cls in gammal.data.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: gammal.data
    :members:
    :exclude-members: Data, HeteroData

    .. autoclass:: Data
       :special-members: __cat_dim__, __inc__
       :inherited-members:

    .. autoclass:: HeteroData
       :special-members: __cat_dim__, __inc__
       :inherited-members:
