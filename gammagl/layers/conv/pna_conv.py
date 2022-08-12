import copy
from typing import Dict, List, Optional

import numpy
import torch
import tensorlayerx as tlx
from tensorlayerx import ReLU, convert_to_tensor, convert_to_numpy, log, exp, to_device, reduce_sum
from tensorlayerx.nn import Linear, ModuleList, Sequential
from torch import Tensor
# from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from gammagl.layers.pool.glob import global_sum_pool, global_mean_pool, global_max_pool, global_min_pool

from gammagl.layers.conv.message_passing import MessagePassing
from gammagl.utils.degree import degree
from torch import nn as nn

class PNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote MLPs.

    .. note::

        For an example of using :obj:`PNAConv`, see `examples/pna.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/pna.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers (list of str): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False):

        super(PNAConv, self).__init__()
        print('using self-define conv')
        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        # deg = convert_to_numpy(deg)
        # deg = to_device(deg, device='CPU')
        num_nodes = int(reduce_sum(deg))

        bin_degrees = tlx.arange(start=0, limit=len(deg), dtype=tlx.float32)
        # bin_degrees = tlx.to_device(bin_degrees, device='CPU')
        # bin_degrees = torch.arange(deg.numel())
        self.avg_deg: Dict[str, float] = {
            'lin': float(reduce_sum(bin_degrees * deg)) / num_nodes,
            'log': float(reduce_sum(tlx.log(bin_degrees + 1) * deg)) / num_nodes,
            'exp': float(reduce_sum(tlx.exp(bin_degrees) * deg)) / num_nodes,
        }
        # deg = deg.to(torch.float)
        num_nodes_test = int(deg.sum())
        # bin_degrees = torch.arange(deg.numel())
        self.avg_deg_test: Dict[str, float] = {
            'lin': float((bin_degrees * deg).sum()) / num_nodes,
            'log': float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            'exp': float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(in_features=edge_dim, out_features=self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear(in_features=(3 if edge_dim else 2) * self.F_in, out_features=self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(in_features=self.F_in, out_features=self.F_in)]
            self.pre_nns.append(Sequential(modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_features=in_channels, out_features=self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(in_features=self.F_out, out_features=self.F_out)]
            self.post_nns.append(Sequential(modules))

        self.lin = Linear(in_features=out_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        if self.divide_input:
            x = tlx.reshape(x, (-1, self.towers, self.F_in))
        else:
            # test = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)
            x = tlx.reshape(x, (-1, 1, self.F_in))
            x = x.repeat(1, self.towers, 1)
            # x = convert_to_tensor(numpy.repeat(convert_to_numpy(tlx.reshape(x, (-1, 1, self.F_in))), self.towers, 1))
            # print(tlx.ops.equal(test, x))
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = tlx.concat([x, out], axis=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = tlx.concat(outs, axis=1)
        return self.lin(out)

    def message(self, x, edge_index, edge_attr=None):
        x_j = tlx.gather(x, edge_index[0, :])
        x_i = tlx.gather(x, edge_index[1, :])

        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in).repeat(1, self.towers, 1)
            # edge_attr = tlx.reshape(edge_attr, (-1, 1, self.F_in))
            # edge_attr_list = [tlx.ops.to_device(edge_attr, device='GPU') for i in range(0, self.towers)]
            # edge_attr = tlx.concat(edge_attr_list, axis=1)
            h = tlx.concat([x_i, x_j, edge_attr], axis=-1)
        else:
            h = tlx.concat([x_i, x_j], axis=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return tlx.stack(hs, axis=1)

    def aggregate(self, inputs, index, num_nodes=None, aggr=None):
        outs = []
        dim_size = None
        dst_index = index[1, :]

        for aggregator in self.aggregators:
            if aggregator == 'sum':
                # out = scatter(inputs, dst_index, 0, None, dim_size, reduce='sum')
                out = global_sum_pool(inputs, dst_index)
            elif aggregator == 'mean':
                # out = scatter(inputs, dst_index, 0, None, dim_size, reduce='mean')
                out = global_mean_pool(inputs, dst_index)
            elif aggregator == 'min':
                # out = scatter(inputs, dst_index, 0, None, dim_size, reduce='min')
                out = global_min_pool(inputs, dst_index)
            elif aggregator == 'max':
                # out = scatter(inputs, dst_index, 0, None, dim_size, reduce='max')
                out = global_max_pool(inputs, dst_index)
            elif aggregator == 'var' or aggregator == 'std':
                # mean = scatter(inputs, dst_index, 0, None, dim_size, reduce='mean')
                mean = global_mean_pool(inputs, dst_index)
                # mean_squares = scatter(inputs * inputs, dst_index, 0, None,
                #                        dim_size, reduce='mean')
                mean_squares = global_mean_pool(inputs * inputs, dst_index)
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    # out1 = torch.sqrt(torch.relu(out) + 1e-5)
                    out = tlx.sqrt(tlx.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = tlx.concat(outs, axis=-1)
        deg = degree(dst_index, dim_size, dtype=inputs.dtype)
        deg = tlx.reshape(tlx.where(deg > 1, deg, tlx.ones_like(deg)), (-1, 1, 1))
        # deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (tlx.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / tlx.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return tlx.concat(outs, axis=-1)

