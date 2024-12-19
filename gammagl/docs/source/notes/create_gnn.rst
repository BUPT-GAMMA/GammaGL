Creating Message Passing Networks
=================================

Generalizing the convolution operator to irregular domains is typically expressed as a *neighborhood aggregation* or *message passing* scheme.
With :math:`\mathbf{x}^{(k-1)}_i \in \mathbb{R}^F` denoting node features of node :math:`i` in layer :math:`(k-1)` and :math:`\mathbf{e}_{j,i} \in \mathbb{R}^D` denoting (optional) edge features from node :math:`j` to node :math:`i`, message passing graph neural networks can be described as

.. math::
  \mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right),

where :math:`\square` denotes a differentiable, permutation invariant function, *e.g.*, sum, mean or max, and :math:`\gamma` and :math:`\phi` denote differentiable functions such as MLPs (Multi Layer Perceptrons).

.. contents::
    :local:

The "MessagePassing" Base Class
-------------------------------

GammaGL provides the :class:`~gammagl.layers.conv.message_passing.MessagePassing` base class, which helps in creating such kinds of message passing graph neural networks by automatically taking care of message propagation.
The user only has to define the functions :math:`\phi` , *i.e.* :meth:`~gammagl.layers.conv.message_passing.MessagePassing.message`, and :math:`\gamma` , *i.e.* :meth:`~gammagl.layers.conv.message_passing.MessagePassing.update`.

This is done with the help of the following methods:

* :obj:`MessagePassing.propagate( x, edge_index, edge_weight=None, num_nodes=None, aggr='sum')`:
  The initial call to start propagating messages.
  Takes in the node features, edge indices and all additional data which is needed to construct messages and to update node embeddings.
  The aggregation scheme to use (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`)
  .. Note that :func:`~gammagl.layers.conv.message_passing.MessagePassing.propagate` is not limited to exchanging messages in square adjacency matrices of shape :obj:`[N, N]` only, but can also exchange messages in general sparse assignment matrices, *e.g.*, bipartite graphs, of shape :obj:`[N, M]` by passing :obj:`size=(N, M)` as an additional argument.
  .. If set to :obj:`None`, the assignment matrix is assumed to be a square matrix.
  .. For bipartite graphs with two independent sets of nodes and indices, and each set holding its own information, this split can be marked by passing the information as a tuple, *e.g.* :obj:`x=(x_N, x_M)`.
* :obj:`MessagePassing.message(x, edge_index, edge_weight=None)`: Constructs messages to node :math:`i` in analogy to :math:`\phi` for each edge :math:`(i,j) \in \mathcal{E}`.
  .. Can take any argument which was initially passed to :meth:`propagate`.
  .. In addition, tensors passed to :meth:`propagate` can be mapped to the respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or :obj:`_j` to the variable name, *e.g.* :obj:`x_i` and :obj:`x_j`.
  Note that we generally refer to :math:`i` as the central nodes that aggregates information, and refer to :math:`j` as the neighboring nodes.
* :obj:`MessagePassing.update(x)`: Updates node embeddings in analogy to :math:`\gamma` for each node :math:`i \in \mathcal{V}`.
  Takes in the output of aggregation as first argument and any argument which was initially passed to :func:`~gammagl.layers.conv.message_passing.MessagePassing.propagate`.

Let us verify this by re-implementing two popular GNN variants, the `GCN layer from Kipf and Welling <https://arxiv.org/abs/1609.02907>`_ and the `EdgeConv layer from Wang et al. <https://arxiv.org/abs/1801.07829>`_.

Implementing the GCN Layer
--------------------------

The `GCN layer <https://arxiv.org/abs/1609.02907>`_ is mathematically defined as

.. math::

    \mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}} \cdot \left( \mathbf{\Theta}^{\top} \cdot \mathbf{x}_j^{(k-1)} \right),

where neighboring node features are first transformed by a weight matrix :math:`\mathbf{\Theta}`, normalized by their degree, and finally summed up.
This formula can be divided into the following steps:

1. Add self-loops to the adjacency matrix.
2. Linearly transform node feature matrix.
3. Compute normalization coefficients.
4. Normalize node features in :math:`\phi`.
5. Sum up neighboring node features (:obj:`"add"` aggregation).

Steps 1-3 are typically computed before message passing takes place.
Steps 4-5 can be easily processed using the :class:`~gammagl.layers.conv.message_passing.MessagePassing` base class.
The full layer implementation is shown below:

.. code-block:: python

    import tensorlayerx as tlx
    from gammagl.layers import MessagePassing
    from gammagl.utils import add_self_loops, degree
    from gammagl.mpops import unsorted_segment_sum

    class GCNConv(MessagePassing):
        def __init__(self, in_channels, out_channels, add_bias):
            super().__init__()
            self.lin = tlx.layers.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]

            # Step 1: Add self-loops to the adjacency matrix.
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

            # Step 2: Linearly transform node feature matrix.
            x = self.lin(x)

            # Step 3: Compute edge weight.
            src, dst = edge_index[0], edge_index[1]
            edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))
            deg = degree(dst, num_nodes=x.shape[0])
            deg_inv_sqrt = tlx.pow(deg, -0.5)
            weights = tlx.ops.gather(deg_inv_sqrt, src) * tlx.reshape(edge_weight, (-1,)) * tlx.ops.gather(deg_inv_sqrt, dst)

            # Step 4-5: Start propagating messages.
            return self.propagate(x, edge_index, edge_weight=weights, num_nodes=x.shape[0], aggr_type='sum')
        
        def message(self, x, edge_index, edge_weight):
            msg = tlx.gather(x, edge_index[0, :]) 
            return msg * edge_weight



:class:`~gammagl.layers.conv.GCNConv` inherits from :class:`~gammagl.layers.conv.message_passing.MessagePassing`.
All the logic of the layer takes place in its :meth:`forward` method.
Here, we first add self-loops to our edge indices using the :meth:`gammagl.utils.add_self_loops` function (step 1), as well as linearly transform node features by calling the :class:`torch.nn.Linear` instance (step 2).

The edge weight coefficients are derived by the node degrees :math:`\deg(i)` for each node :math:`i` which gets transformed to :math:`1/(\sqrt{\deg(i)} \cdot \sqrt{\deg(j)})` for each edge :math:`(j,i) \in \mathcal{E}`.
The result is saved in the tensor :obj:`weights` of shape :obj:`[num_edges, ]` (step 3).

We then call :meth:`~gammagl.layers.conv.message_passing.MessagePassing.propagate`, which internally calls :meth:`~gammagl.layers.conv.message_passing.MessagePassing.message`, :meth:`~gammagl.layers.conv.message_passing.MessagePassing.aggregate` and :meth:`~gammagl.layers.conv.message_passing.MessagePassing.update`.
We pass the node embeddings :obj:`x` and the edge weights coefficients :obj:`weights` as additional arguments for message propagation.

In the :meth:`~gammagl.layers.conv.message_passing.MessagePassing.message` function, node features will be send to edge and multiplied with specific edge weights.

That is all that it takes to create a simple message passing layer.
You can use this layer as a building block for deep architectures.
Initializing and calling it is straightforward:

.. code-block:: python

    conv = GCNConv(16, 32)
    x = conv(x, edge_index)

Implementing the Edge Convolution
---------------------------------

The `edge convolutional layer <https://arxiv.org/abs/1801.07829>`_ processes graphs or point clouds and is mathematically defined as

.. math::

    \mathbf{x}_i^{(k)} = \max_{j \in \mathcal{N}(i)} h_{\mathbf{\Theta}} \left( \mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)} - \mathbf{x}_i^{(k-1)} \right),

where :math:`h_{\mathbf{\Theta}}` denotes an MLP.
In analogy to the GCN layer, we can use the :class:`~gammagl.layers.conv.message_passing.MessagePassing` class to implement this layer, this time using the :obj:`"max"` aggregation:

.. code-block:: python

    import tensorlayerx as tlx
    from tensorlayerx.nn import Sequential as Seq, Linear, ReLU
    from gammagl.layers import MessagePassing

    class EdgeConv(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.mlp = Seq(Linear(2 * in_channels, out_channels),
                           ReLU(),
                           Linear(out_channels, out_channels))

        def forward(self, x, edge_index):
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]

            return self.propagate(x=x, edge_index,aggr_type='max')

        def message(self, x_i, x_j):
            # x_i has shape [E, in_channels]
            # x_j has shape [E, in_channels]

            tmp = tlx.concat([x_i, x_j - x_i], axis=1)  # tmp has shape [E, 2 * in_channels]
            return self.mlp(tmp)

Inside the :meth:`~gammagl.layers.conv.message_passing.MessagePassing.message` function, we use :obj:`self.mlp` to transform both the target node features :obj:`x_i` and the relative source node features :obj:`x_j - x_i` for each edge :math:`(j,i) \in \mathcal{E}`.

The edge convolution is actually a dynamic convolution, which recomputes the graph for each layer using nearest neighbors in the feature space.
Luckily, GammaGL comes with a batch-wise k-NN graph generation method named :meth:`gammagl.layers.pool.knn_graph`:

.. code-block:: python

    from gammagl.layers import knn_graph

    class DynamicEdgeConv(EdgeConv):
        def __init__(self, in_channels, out_channels, k=6):
            super().__init__(in_channels, out_channels)
            self.k = k

        def forward(self, x, batch=None):
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            return super().forward(x, edge_index)

Here, :meth:`~gammagl.layers.pool.knn_graph` computes a nearest neighbor graph, which is further used to call the :meth:`forward` method of :class:`~gammagl.layers.conv.EdgeConv`.

This leaves us with a clean interface for initializing and calling this layer:

.. code-block:: python

    conv = DynamicEdgeConv(3, 128, k=6)
    x = conv(x, batch)

Exercises
---------

Imagine we are given the following :obj:`~gammagl.data.Data` object:

.. code-block:: python

    import tensorlayerx as tlx
    from gammagl.data import Graph

    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2],
                                        [1, 0, 2, 1]], dtype=tlx.int64)
    x = tlx.convert_to_tensor([[-1], [0], [1]], dtype=tlx.float32)

    graph = Graph(x=x, edge_index=edge_index)

Try to answer the following questions related to :class:`~gammagl.layers.conv.GCNConv`:

1. What information does :obj:`src` and :obj:`dst` hold?

2. What does :meth:`~gammagl.utils.degree` do?

3. Why do we use :obj:`degree(dst, ...)` rather than :obj:`degree(src, ...)`?

4. What does :obj:`tlx.gather(deg_inv_sqrt, dst)` and :obj:`tlx.gather(deg_inv_sqrt, src)` do?

5. What information does :obj:`msg` hold in the :meth:`~gammagl.layers.conv.MessagePassing.message` function? If :obj:`self.lin` denotes the identity function, what is the exact content of :obj:`msg`?

6. Add an :meth:`~gammagl.layers.conv.MessagePassing.update` function to :class:`~gammagl.layers.conv.GCNConv` that adds transformed central node features to the aggregated output.

Try to answer the following questions related to :class:`~gammagl.layers.conv.EdgeConv`:

1. What is :obj:`x_i` and :obj:`x_j - x_i`?

2. What does :obj:`tlx.concat([x_i, x_j - x_i], axis=1)` do? Why :obj:`axis = 1`?
