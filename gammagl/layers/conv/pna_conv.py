import tensorlayerx as tlx
import tensorlayerx.nn
from tensorlayerx import reduce_sum
from tensorlayerx.nn import ReLU, Linear, ModuleList, Sequential
from gammagl.layers.pool.glob import global_sum_pool, global_mean_pool, global_max_pool, global_min_pool
from gammagl.layers.conv.message_passing import MessagePassing
from gammagl.utils.degree import degree


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


    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels, out_channels, aggregators, scalers, deg, edge_dim, towers=1,
                 pre_layers=1, post_layers=1, divide_input=False):

        super(PNAConv, self).__init__()
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

        # Mul operation in TensorFlow require two tensor have the same type
        if tlx.BACKEND == 'tensorflow':
            deg = tlx.constant(tlx.convert_to_numpy(deg), dtype=tlx.float32)
        num_nodes = int(reduce_sum(deg))
        bin_degrees = tlx.arange(start=0, limit=len(deg), dtype=tlx.float32)
        self.avg_deg = {
            'lin': float(reduce_sum(bin_degrees * deg)) / num_nodes,
            'log': float(reduce_sum(tlx.log(bin_degrees + 1) * deg)) / num_nodes,
            'exp': float(reduce_sum(tlx.exp(bin_degrees) * deg)) / num_nodes,
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(in_features=edge_dim, out_features=self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = Sequential([Linear(in_features=(3 if edge_dim else 2) * self.F_in, out_features=self.F_in)])
            for _ in range(pre_layers - 1):
                modules.append(ReLU())
                modules.append(Linear(in_features=self.F_in, out_features=self.F_in))
            self.pre_nns.append(modules)

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = Sequential([Linear(in_features=in_channels, out_features=self.F_out)])
            for _ in range(post_layers - 1):
                modules.append(ReLU())
                modules.append(Linear(in_features=self.F_out, out_features=self.F_out))
            self.post_nns.append(modules)

        self.lin = Linear(in_features=out_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        if self.divide_input:
            x = tlx.reshape(x, (-1, self.towers, self.F_in))
        else:
            x = tlx.stack([x for i in range(0, self.towers)], axis=1)

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
            edge_attr = tlx.stack([edge_attr for i in range(0, self.towers)], axis=1)
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
                out = global_sum_pool(inputs, dst_index)
            elif aggregator == 'mean':
                out = global_mean_pool(inputs, dst_index)
            elif aggregator == 'min':
                out = global_min_pool(inputs, dst_index)
            elif aggregator == 'max':
                out = global_max_pool(inputs, dst_index)
            elif aggregator == 'var' or aggregator == 'std':
                mean = global_mean_pool(inputs, dst_index)
                mean_squares = global_mean_pool(inputs * inputs, dst_index)
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = tlx.sqrt(tlx.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = tlx.concat(outs, axis=-1)
        deg = degree(dst_index, dim_size, dtype=inputs.dtype)
        deg = tlx.reshape(tlx.where(deg > 1, deg, tlx.ones_like(deg)), (-1, 1, 1))

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