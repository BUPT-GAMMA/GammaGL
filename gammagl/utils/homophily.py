import tensorlayerx as tlx
import numpy as np
from gammagl.mpops import unsorted_segment_mean
from gammagl.utils import degree

def homophily(edge_index, y, batch=None, method: str = 'edge'): 
    r"""The homophily of a graph characterizes how likely nodes with the same 
    label are near each other in a graph.
    There are many measures of homophily that fits this definition.
    In particular:
    - In the `"Beyond Homophily in Graph Neural Networks: Current Limitations
      and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper, the
      homophily is the fraction of edges in a graph which connects nodes
      that have the same class label:
      .. math::
        \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }
        {|\mathcal{E}|}
      That measure is called the *edge homophily ratio*.
    - In the `"Geom-GCN: Geometric Graph Convolutional Networks"
      <https://arxiv.org/abs/2002.05287>`_ paper, edge homophily is normalized
      across neighborhoods:
      .. math::
        \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w
        \in \mathcal{N}(v) \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }
      That measure is called the *node homophily ratio*.
    - In the `"Large-Scale Learning on Non-Homophilous Graphs: New Benchmarks
      and Strong Simple Methods" <https://arxiv.org/abs/2110.14446>`_ paper,
      edge homophily is modified to be insensitive to the number of classes
      and size of each class:
      .. math::
        \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, h_k - \frac{|\mathcal{C}_k|}
        {|\mathcal{V}|} \right),
      where :math:`C` denotes the number of classes, :math:`|\mathcal{C}_k|`
      denotes the number of nodes of class :math:`k`, and :math:`h_k` denotes
      the edge homophily ratio of nodes of class :math:`k`.
      Thus, that measure is called the *class insensitive edge homophily
      ratio*.
    Args:
        edge_index (Tensor): The graph connectivity.
        y (Tensor): The labels.
        batch (LongTensor, optional): Batch vector\
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns
            each node to a specific example. (default: :obj:`None`)
        method (str, optional): The method used to calculate the homophily,
            either :obj:`"edge"` (first formula), :obj:`"node"` (second
            formula) or :obj:`"edge_insensitive"` (third formula).
            (default: :obj:`"edge"`)
    Examples:
        >>> edge_index = tlx.convert_to_tensor([[0, 1, 2, 3],
        ...                            [1, 2, 0, 4]])
        >>> y = tlx.convert_to_tensor([0, 0, 0, 0, 1])
        >>> # Edge homophily ratio
        >>> homophily(edge_index, y, method='edge')
        0.75
        >>> # Node homophily ratio
        >>> homophily(edge_index, y, method='node')
        0.6000000238418579
        >>> # Class insensitive edge homophily ratio
        >>> homophily(edge_index, y, method='edge_insensitive')
        0.19999998807907104
    """
    assert method in {'edge', 'node', 'edge_insensitive'}
    if tlx.is_tensor(edge_index):
        edge_index = tlx.convert_to_numpy(edge_index)
    if tlx.is_tensor(y):
        y = tlx.convert_to_numpy(y)
    y = np.squeeze(y, axis=-1) if y.ndim > 1 else y

    row, col = edge_index[0], edge_index[1]

    if method == 'edge':
        out = np.zeros(row.shape[0])
        out[y[row] == y[col]] = 1.
        if batch is None:
            return float(out.mean())
        else:
            out = tlx.convert_to_tensor(out)
            col = tlx.convert_to_tensor(col)
            batch = tlx.convert_to_tensor(batch)
            return unsorted_segment_mean(out, tlx.gather(batch, col))

    elif method == 'node':
        out = np.zeros(row.shape[0])
        out[y[row] == y[col]] = 1.
        
        out = unsorted_segment_mean(tlx.convert_to_tensor(out), tlx.convert_to_tensor(col))
        print(out)
        if batch is None:
            return float(tlx.reduce_mean(out))
        else:
            return unsorted_segment_mean(out, batch)

    elif method == 'edge_insensitive':
        assert y.ndim == 1
        num_classes = int(y.max()) + 1
        assert num_classes >= 2
        if batch is None:
            batch = np.zeros_like(y)
        elif tlx.is_tensor(batch):
            batch = tlx.convert_to_numpy(batch)
        num_nodes = tlx.convert_to_numpy(degree(batch))
        num_graphs = num_nodes.size
        batch = num_classes * batch + y

        h = tlx.convert_to_numpy(homophily(edge_index, y, batch, method='edge'))
        h = np.reshape(h, [num_graphs, num_classes])

        counts = np.bincount(batch, minlength=num_classes * num_graphs)
        counts = np.reshape(counts, [num_graphs, num_classes])
        proportions = counts / np.reshape(num_nodes, [-1, 1])

        out = np.clip((h - proportions), a_min=0, a_max=None).sum(axis=-1)
        out /= num_classes - 1
        return out if out.size > 1 else float(out)

    else:
        raise NotImplementedError
