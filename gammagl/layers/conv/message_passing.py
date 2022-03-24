import tensorlayerx as tlx


class MessagePassing(tlx.nn.Module):
    r"""Base class for creating message passing layers of the form

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    """

    def __init__(self):
        super().__init__()

    def message(self, x, edge_index, edge_weight=None, num_nodes=None):
        msg = tlx.ops.gather(x, edge_index[0, :])
        if edge_weight is not None:
            edge_weight = tlx.expand_dims(edge_weight, -1)
            return msg * edge_weight
        else:
            return msg

    def aggregate(self, msg, edge_index, num_nodes=None, aggr_type='sum'):
        dst_index = edge_index[1, :]
        if aggr_type == 'sum':
            return tlx.ops.unsorted_segment_sum(msg, dst_index, num_nodes)
        if aggr_type == 'mean':
            return tlx.ops.unsorted_segment_mean(msg, dst_index, num_nodes)
        if aggr_type == 'max':
            return tlx.ops.unsorted_segment_max(msg, dst_index, num_nodes)
        if aggr_type == 'min':
            return tlx.ops.unsorted_segment_min(msg, dst_index, num_nodes)

    # def message_aggregate(self, x, index):
    #     x = self.message(x, index)
    #     x = self.aggregate(x, index)
    #     return x

    def update(self, x):
        return x

    def propagate(self, x, edge_index, edge_weight=None, num_nodes=None, aggr='sum'):
        if num_nodes is None:
            num_nodes = x.shape[0]
        msg = self.message(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        x = self.aggregate(msg, edge_index, num_nodes=num_nodes, aggr_type=aggr)
        x = self.update(x)
        return x
