# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
from gammagl.layers.conv import APPNPConv
from tensorlayerx.nn import Module, Sequential, ModuleDict, Linear, Tanh


class SemAttAggr(Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()

        self.project = Sequential(
            Linear(in_features=in_size, out_features=hidden_size), # W
            Tanh(),
            Linear(in_features=hidden_size, out_features=1, b_init=None) # q
        )

    def forward(self, z):
        w = tlx.reduce_mean(self.project(z), axis=1)    # (M, 1)
        beta = tlx.softmax(w, axis=0)                   # (M, 1)
        beta = tlx.expand_dims(beta, axis=-1)  # (M, 1, 1) # auto expand
        return tlx.reduce_sum(beta * z, axis=0) # (N, H)


class HPNConv(MessagePassing):
    r"""
        The Heterogeneous Graph Propagation Operator from the
        `"Heterogeneous Graph Propagation Network"
        <https://ieeexplore.ieee.org/abstract/document/9428609>`_ paper.

        .. note::

            For an example of using HPNConv, see `examples/hpn_trainer.py
            <https://github.com/BUPT-GAMMA/GammaGL/tree/main/examples/hpn>`_.

        Parameters
        ----------
        in_channels: int or Dict[str, int]
            Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels: int
            Size of each output sample.
        metadata: Tuple[List[str], List[Tuple[str, str, str]]]
            The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`gammagl.data.HeteroGraph.metadata` for more
            information.
        iter_K: int
            Number of iterations used in APPNPConv.
        alpha: float, optional
            Parameters used in APPNPConv.
            (default: :obj:`0.1`)
        negative_slope: float, optional
            LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout: float, optional
            Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        """
    def __init__(self,
                 in_channels,
                 out_channels,
                 metadata,
                 iter_K,
                 alpha=0.1,
                 negative_slope=0.2,
                 drop_rate=0.5):
        super().__init__()
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.negetive_slop = negative_slope
        self.drop_rate = drop_rate


        self.appnp_dict = ModuleDict({})
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            self.appnp_dict[edge_type] = APPNPConv(in_channels=in_channels[src_type],
                                               out_channels=out_channels,
                                               iter_K=iter_K,
                                               alpha=alpha,
                                               drop_rate=drop_rate
                                               )

        self.sem_att_aggr = SemAttAggr(in_size=out_channels,
                                       hidden_size=out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out_dict = {}
        # Iterate over node types:
        for node_type, x_node in x_dict.items():
            out_dict[node_type] = []

        # node level attention aggregation
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            out = self.appnp_dict[edge_type](x_dict[src_type],
                                           edge_index,
                                           num_nodes = num_nodes_dict[dst_type])
            out = tlx.relu(out)
            out_dict[dst_type].append(out)

        # semantic attention aggregation
        for node_type, outs in out_dict.items():
            outs = tlx.stack(outs)
            out_dict[node_type] = self.sem_att_aggr(outs)

        return out_dict

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')