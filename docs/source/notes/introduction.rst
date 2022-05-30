Introduction by Example
=======================

Following `PyG <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html>`_ ,
we shortly introduce the fundamental concepts of GammaGL through self-contained examples.
At its core, GammaGL provides the following main features:

.. contents::
    :local:

Data Handling of Graphs
-----------------------

A graph is used to model pairwise relations (edges) between objects (nodes).
A single graph in GammaGL is described by an instance of :class:`gammagl.data.Graph`, which holds the following attributes by default:

- :obj:`Graph.x`: Node feature matrix with shape :obj:`[num_nodes, num_node_features]`
- :obj:`Graph.edge_index`: Graph connectivity in COO format with shape :obj:`[2, num_edges]` and type :obj:`int64`
- :obj:`Graph.edge_attr`: Edge feature matrix with shape :obj:`[num_edges, num_edge_features]`
- :obj:`Graph.y`: Target to train against (may have arbitrary shape), *e.g.*, node-level targets of shape :obj:`[num_nodes, *]` or graph-level targets of shape :obj:`[1, *]`
- :obj:`Graph.pos`: Node position matrix with shape :obj:`[num_nodes, num_dimensions]`

None of these attributes are required.
In fact, the :class:`~gammagl.data.Graph` object is not even restricted to these attributes.
We can, *e.g.*, extend it by :obj:`Graph.face` to save the connectivity of triangles from a 3D mesh in a tensor with shape :obj:`[3, num_faces]` and type :obj:`int64`.

.. Note::
    PyTorch and :obj:`torchvision` define an example as a tuple of an image and a target.
    We omit this notation in GammaGL to allow for various data structures in a clean and understandable way.

We show a simple example of an unweighted and undirected graph with three nodes and four edges.
Each node contains exactly one feature:

.. code-block:: python

    import tensorlayerx as tlx
    from gammagl.data import Graph

    edge_index = tlx.convert_to_tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=tlx.int64)
    x = tlx.convert_to_tensor([[-1], [0], [1]], dtype=tlx.float32)

    graph = Graph(x=x, edge_index=edge_index)
    >>> Graph(edge_index=[2, 4], x=[3, 1])

.. image:: ../_figures/graph.svg
  :align: center
  :width: 300px


Although the graph has only two edges, we need to define four index tuples to account for both directions of a edge.

.. Note::
    You can print out your data object anytime and receive a short information about its attributes and their shapes.

Besides holding a number of node-level, edge-level or graph-level attributes, :class:`~gammagl.data.Graph` provides a number of useful utility functions, *e.g.*:

.. code-block:: python

    print(graph.keys)
    >>> ['x', 'edge_index']

    print(graph['x'])
    >>> tensor([[-1.0],
                [0.0],
                [1.0]])

    for key, item in graph:
        print(f'{key} found in data')
    >>> KeyError: 0
    #  Note that <class 'gammagl.data.graph.Graph'> can not be an iteratable object, or raise error in Paddle.save.

    'edge_attr' in graph
    >>> False

    graph.num_nodes
    >>> 3

    graph.num_edges
    >>> 4

    graph.num_node_features
    >>> 1

    graph.has_isolated_nodes()
    >>> False

    graph.has_self_loops()
    >>> False

    graph.is_directed()
    >>> False


You can find a complete list of all methods at :class:`gammagl.data.Graph`.

Common Benchmark Datasets
-------------------------

Gammagl will contain a large number of common benchmark datasets, *e.g.*, all Planetoid datasets (Cora, Citeseer, Pubmed), all graph classification datasets from `http://graphkernels.cs.tu-dortmund.de <http://graphkernels.cs.tu-dortmund.de/>`_ and their `cleaned versions <https://github.com/nd7141/graph_datasets>`_, the QM7 and QM9 dataset, and a handful of 3D mesh/point cloud datasets like FAUST, ModelNet10/40 and ShapeNet.

Initializing a dataset is straightforward.
An initialization of a dataset will automatically download its raw files and process them to the previously described :class:`~gammagl.data.Data` format.
*E.g.*, to load the ENZYMES dataset (consisting of 600 graphs within 6 classes), type:

.. code-block:: python

    from gammagl.datasets import TUDataset

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    >>> ENZYMES(600)

    len(dataset)
    >>> 600

    dataset.num_classes
    >>> 6

    dataset.num_node_features
    >>> 3

We now have access to all 600 graphs in the dataset:

.. code-block:: python

    graph = dataset[0]
    >>> Graph(edge_index=[2, 168], x=[37, 3], y=[1])

    graph.is_undirected()
    >>> True

We can see that the first graph in the dataset contains 37 nodes, each one having 3 features.
There are 168/2 = 84 undirected edges and the graph is assigned to exactly one class.
In addition, the data object is holding exactly one graph-level target.

We can even use slices, long or bool tensors to split the dataset.
*E.g.*, to create a 90/10 train/test split, type:

.. code-block:: python

    train_dataset = dataset[:540]
    >>> ENZYMES(540)

    test_dataset = dataset[540:]
    >>> ENZYMES(60)

If you are unsure whether the dataset is already shuffled before you split, you can randomly permutate it by running:

.. code-block:: python

    dataset = dataset.shuffle()
    >>> ENZYMES(600)

This is equivalent of doing:

.. code-block:: python

    perm = np.random.permutation(len(dataset))
    dataset = dataset[perm]
    >> ENZYMES(600)

Let's try another one! Let's download Cora, the standard benchmark dataset for semi-supervised graph node classification:

.. code-block:: python

    from gammagl.datasets import Planetoid

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    >>> Cora()

    len(dataset)
    >>> 1

    dataset.num_classes
    >>> 7

    dataset.num_node_features
    >>> 1433

Here, the dataset contains only a single, undirected citation graph:

.. code-block:: python

    graph = dataset[0]
    >>> Graph(edge_index=[2, 10556], x=[2708, 1433], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

    graph.is_undirected()
    >>> True

    tlx.convert_to_numpy(graph.train_mask).sum()
    >>> 140

    tlx.convert_to_numpy(graph.val_mask).sum()
    >>> 500

    tlx.convert_to_numpy(graph.test_mask).sum()
    >>> 1000

This time, the :class:`~gammagl.data.Graph` objects holds a label for each node, and additional node-level attributes: :obj:`train_mask`, :obj:`val_mask` and :obj:`test_mask`, where

- :obj:`train_mask` denotes against which nodes to train (140 nodes),
- :obj:`val_mask` denotes which nodes to use for validation, *e.g.*, to perform early stopping (500 nodes),
- :obj:`test_mask` denotes against which nodes to test (1000 nodes).

Mini-batches
------------

Neural networks are usually trained in a batch-wise fashion.
GammaGL achieves parallelization over a mini-batch by creating sparse block diagonal adjacency matrices (defined by :obj:`edge_index`) and concatenating feature and target matrices in the node dimension.
This composition allows differing number of nodes and edges over examples in one batch:

.. math::

    \mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}

GammaGL contains its own :class:`gammagl.loader.DataLoader`, which already takes care of this concatenation process.
Let's learn about it in an example:

.. code-block:: python

    from gammagl.datasets import TUDataset
    from gammagl.loader.dataloader import DataLoader

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        batch
        >>> GraphBatch(edge_index=[2, 3806], x=[1096, 21], y=[32], batch=[1096], ptr=[33])

        batch.num_graphs
        >>> 32

:class:`gammagl.data.BatchGraph` inherits from :class:`gammagl.data.Graph` and contains an additional attribute called :obj:`batch`.

:obj:`batch` is a column vector which maps each node to its respective graph in the batch:

.. math::

    \mathrm{batch} = {\begin{bmatrix} 0 & \cdots & 0 & 1 & \cdots & n - 2 & n -1 & \cdots & n - 1 \end{bmatrix}}^{\top}

You can use it to, *e.g.*, average node features in the node dimension for each graph individually:

.. code-block:: python

    from gammagl.mpops import unsorted_segment_mean
    from gammagl.datasets import TUDataset
    from gammagl.loader.dataloader import DataLoader

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        data
        >>> DataBatch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

        data.num_graphs
        >>> 32

        x = unsorted_segment_mean(data.x, data.batch)
        x.shape
        >>> TensorShape([32, 21])

You can learn more about the internal batching procedure of GammaGL, *e.g.*, how to modify its behaviour, `here <https://gammagl.readthedocs.io/en/latest/notes/batching.html>`_.

