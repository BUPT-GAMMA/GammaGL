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
    :exclude-members: Data, HeteroData

    .. autoclass:: Data
       :special-members: __cat_dim__, __inc__
       :inherited-members:

    .. autoclass:: HeteroData
       :special-members: __cat_dim__, __inc__
       :inherited-members:
