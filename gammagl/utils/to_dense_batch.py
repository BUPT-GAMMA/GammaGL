import tensorlayerx as tlx

def to_dense_batch(x, batch=None, fill_value=0, max_num_nodes=None):
    r"""
        Given a sparse batch of node features
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
        :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
        dense node feature tensor
        :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
        :math:`N_{\max} = \max_i^B N_i`).
        In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
        N_{\max}}` is returned, holding information about the existence of
        fake-nodes in the dense representation.

        Parameters
        ----------
        x: tensor
            Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch: tensor
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value: float, optional
            The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes: int, optional
            The size of the output node dimension.
            (default: :obj:`None`)
        batch_size: int, optional
            The batch size. (default: :obj:`None`)

        Returns
        -------
        (:class:`Tensor`, :class:`BoolTensor`)

    """
    if batch is None and max_num_nodes is None:
        return tlx.expand_dims(x, axis=0), tlx.ones((1, x.shape[0]), dtype=tlx.bool)
    
    if batch is None:
        batch = tlx.zeros((x.shape[0],), dtype=tlx.int64)
    
    batch_size = tlx.reduce_max(batch) + 1
    num_nodes = tlx.unsorted_segment_sum(tlx.ones((x.shape[0],), dtype=batch.dtype), batch, num_segments=batch_size)
    cum_nodes = tlx.concat([tlx.zeros((1,), dtype=batch.dtype), tlx.cumsum(num_nodes)], axis=0)

    if max_num_nodes is None:
        max_num_nodes = tlx.reduce_max(num_nodes)
    
    idx = tlx.arange(0, batch.shape[0], dtype=batch.dtype)
    idx = idx - tlx.gather(cum_nodes, batch) + batch * max_num_nodes

    shape = [batch_size * max_num_nodes] + list(x.shape)[1:]
    ret = tlx.constant(fill_value, shape=shape, dtype=x.dtype)
    ret = tlx.tensor_scatter_nd_update(ret, tlx.expand_dims(idx, axis=1), x)
    ret = tlx.reshape(ret, shape=[batch_size, max_num_nodes] + list(x.shape)[1:])

    mask = tlx.zeros((batch_size * max_num_nodes,), dtype=tlx.bool)
    if tlx.BACKEND == 'paddle':
        mask[idx] = True
    else:
        mask = tlx.scatter_update(mask, idx, tlx.ones_like(idx, dtype=tlx.bool))
    mask = tlx.reshape(mask, (batch_size, max_num_nodes))

    return ret, mask