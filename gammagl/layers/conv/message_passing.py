import tensorlayerx as tlx
from gammagl.mpops import *
from gammagl.utils import Inspector


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

    special_args = {
        'edge_index', 'x', 'edge_weight'
    }


    def __init__(self):
        super().__init__()

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.__user_args__ = self.inspector.keys(
            ['message',]).difference(self.special_args)


    def message(self, x, edge_index, edge_weight=None):
        """
        Function that construct message from source nodes to destination nodes.

        Parameters
        ----------
        x: tensor
            input node feature.
        edge_index: tensor
            edges from src to dst.
        edge_weight: tensor, optional
            weight of each edge.

        Returns
        -------
        tensor
            output message

        Returns:
            the message matrix, and the shape is [num_edges, message_dim]
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

        Parameters
        ----------
        msg: tensor
            message construct by message function.
        edge_index: tensor
            edges from src to dst.
        num_nodes: int, optional
            number of nodes of the graph.
        aggr: str, optional
            aggregation type, default = 'sum', optional=['sum', 'mean', 'max'].
 
        Returns
        -------
        tensor
            aggregation outcome.

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

    def message_aggregate(self, x, edge_index, edge_weight=None, aggr='sum'):
        """
        try to fuse message and aggregate to reduce expensed edge information.
        """
        # use_ext is defined in mpops
        if use_ext is not None and use_ext:
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1],
                                         device=x.device, 
                                         dtype=x.dtype)
            out = gspmm(edge_index, edge_weight, x, aggr)
        else:
            msg = self.message(x, edge_index, edge_weight)
            out = self.aggregate(msg, edge_index)
        return out

    def update(self, x):
        """
        Function defines how to update node embeddings.

        Parameters
        ----------
        x: tensor
            aggregated message

        """
        return x

    def propagate(self, x, edge_index, aggr='sum', fuse_kernel=False,  **kwargs):
        """
        Function that perform message passing.

        Parameters
        ----------
        x: tensor
            input node feature.
        edge_index: tensor
            edges from src to dst.
        aggr: str, optional
            aggregation type, default='sum', optional=['sum', 'mean', 'max'].
        fuse_kernel: bool, optional
            use fused kernel function to speed up, default = False.
        kwargs: optional
            other parameters dict.

        """

        if 'num_nodes' not in kwargs.keys() or kwargs['num_nodes'] is None:
            kwargs['num_nodes'] = x.shape[0]

        coll_dict = self.__collect__(x, edge_index, aggr, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        if fuse_kernel:
            x = self.message_aggregate(**msg_kwargs)
        else:
            msg = self.message(**msg_kwargs)
            x = self.aggregate(msg, edge_index, num_nodes=kwargs['num_nodes'], aggr=aggr)
        x = self.update(x)
        return x

    def __collect__(self, x, edge_index, aggr, kwargs):
        out = {}

        for k, v in kwargs.items():
            out[k] = v
        out['x'] = x
        out['edge_index'] = edge_index
        out['aggr'] = aggr

        return out
