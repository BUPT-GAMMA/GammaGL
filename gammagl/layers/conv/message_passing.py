import tensorlayerx as tlx
from gammagl.mpops import *


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
        """
        Function that construct message from source nodes to destination nodes.
        Args:
            x (Tensor): input node feature
            edge_index (Tensor): edges from src to dst
            edge_weight (Tensor): weight of each edge
            num_nodes (int): number of nodes of the graph
        Returns:

        """
        msg = tlx.gather(x, edge_index[0, :])
        if edge_weight is not None:
            edge_weight = tlx.expand_dims(edge_weight, -1)
            return msg * edge_weight
        else:
            return msg

    def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum'):
        """
        Function that aggregates message from edges to destination nodes.
        Args:
            msg (Tensor): message construct by message function
            edge_index (Tensor): edges from src to dst
            num_nodes (int): number of nodes of the graph
            aggr (str): aggregation type, default = 'sum', optional=['sum', 'mean', 'max']
 
        Returns:

        """
        dst_index = edge_index[1, :]
        if aggr == 'sum':
            return unsorted_segment_sum(msg, dst_index, num_nodes)
        elif aggr == 'mean':
            return unsorted_segment_mean(msg, dst_index, num_nodes)
        elif aggr == 'max':
            return unsorted_segment_max(msg, dst_index, num_nodes)
        else:
            raise NotImplementedError('Not support for this opearator')

    def update(self, x):
        """
        Function defines how to update node embeddings

        Args:
            x: aggregated message
        """
        return x

    def propagate(self, x, edge_index, edge_weight=None, num_nodes=None, aggr='sum'):
        """
        Function that perform message passing. 
        Args:
            x: input node feature
            edge_index: edges from src to dst
            edge_weight:  weight of each edge
            num_nodes: number of nodes of the graph
            aggr: aggregation type, default='sum', optional=['sum', 'mean', 'max']

        Returns:

        """
        if num_nodes is None:
            num_nodes = x.shape[0]
        msg = self.message(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        x = self.aggregate(msg, edge_index, num_nodes=num_nodes, aggr=aggr)
        x = self.update(x)
        return x
