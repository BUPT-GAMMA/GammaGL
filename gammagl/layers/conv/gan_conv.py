import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import degree
from gammagl.mpops import *
import numpy as np
from gammagl.datasets import Planetoid

class MLP():
    def __init__(self,
                 channel_list=None,
                 norm=None,
                 dropout=0.0,
                 bias=True,
                 ):
        self.layers = tlx.nn.ModuleList()
        for i in range(1, len(channel_list)):
            if bias:
                self.layers.append(tlx.layers.Linear(out_features=channel_list[i], in_features=channel_list[i - 1],
                                                     W_init='xavier_uniform'))
            else:
                self.layers.append(tlx.layers.Linear(out_features=channel_list[i], in_features=channel_list[i - 1],
                                                     W_init='xavier_uniform', b_init=None))
            if i < len(channel_list) - 1:
                if norm and norm == 'batch':
                    self.layers.append(tlx.nn.BatchNorm1d())
                elif norm and norm == 'layer':
                    self.layers.append(tlx.nn.LayerNorm())
                elif norm and norm == 'instance':
                    self.layers.append(tlx.nn.InstanceNormLayer())
                self.layers.append(tlx.nn.ReLU())
                self.layers.append(tlx.nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GANConv(MessagePassing):
    r"""The GENeralized Graph Convolution (GENConv) from the `"DeeperGCN: All
        You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.
        Supports SoftMax & PowerMean aggregation. The message construction is:

        .. math::
            \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_i +
            \mathrm{AGG} \left( \left\{
            \mathrm{ReLU} \left( \mathbf{x}_j + \mathbf{e_{ji}} \right) +\epsilon
            : j \in \mathcal{N}(i) \right\} \right)
            \right)

        .. note::

            For an example of using :obj:`GENConv`see

        Parameters
    ----------
        in_channels: int
            Size of each input sample
        out_channels: int
            Size of each output sample.
        num_layers (int, optional): The number of MLP layers.
            (default: :obj:`2`)
        expansion (int, optional): The expansion factor of hidden channels in
            MLP layers. (default: :obj:`2`)
        eps (float, optional): The epsilon value of the message construction
            function. (default: :obj:`1e-7`)
        p (float, optional): Initial power for power mean aggregation.
            (default: :obj:`1.0`)
        beta (float, optional): Initial inverse temperature for softmax
            aggregation. (default: :obj:`1.0`)
        agg  (str or Aggregation, optional): The aggregation scheme to use.
            (:obj:`"softmax"`, :obj:`"powermean"`, :obj:`"sum"`, :obj:`"mean"`,
            :obj:`max`). (default: :obj:`""`)
    """

    def __init__(self,
                in_channels,
                out_channels,
                agg = 'mean',
                num_layers =2,
                expansion =2,
                eps = 1e-7,
                beta = 1.0,
                beta_learn = True,
                p = 1.0,
                p_learn = True
                ):
        super().__init__()
        if agg not in ['softmax', 'powermean', 'max', 'mean', 'sum']:
            raise ValueError('Invalid agg mode.Must be "softmax" or "powermean" or "max", "mean", "sum".' 'But got "{}".'.format(agg))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._agg = agg
        self.eps = eps
        self.scale = tlx.Variable(initial_value=tlx.convert_to_tensor([1.0]), trainable=True, name='mgs_norm_scalar')
        self.relu = tlx.ReLU()
        self.num_layers = num_layers
        self.expansion = expansion
        if beta_learn:
            self.beta = tlx.Variable(initial_value=tlx.convert_to_tensor([beta]), trainable=True)
        else:
            self.beta = beta

        if p_learn:
            self.p = tlx.Variable(initial_value=tlx.convert_to_tensor([p]), trainable=True)
        else:
            self.p = p

        channels = [self.in_channels]
        for i in range(self.num_layers - 1):
            channels.append(self.out_channels * self.expansion)
        channels.append(self.out_channels)
        self.mlp = MLP(channel_list=channels)


    def message(self, x, edge_index, edge_weight=None):
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = tlx.zeros(shape=(edge_index.shape[1], 1))

        m = tlx.ops.gather(x, dst) + edge_weight
        msg = self.relu(m) + self.eps
        return msg

    def aggregate(self, msg, edge_index, num_nodes=None, aggr='mean' ):
        src_index = edge_index[0, :]
        dst_index = edge_index[1, :]
        deg = degree(src_index, num_nodes=num_nodes, dtype=tlx.float32)
        if aggr == 'sum':
            return unsorted_segment_sum(msg, dst_index, num_nodes)
        elif aggr == 'mean':
            return unsorted_segment_mean(msg, dst_index, num_nodes)
        elif aggr == 'max':
            return unsorted_segment_max(msg, dst_index, num_nodes)
        elif aggr == 'softmax':
            beta_msg = msg * self.beta
            out = np.zeros(shape=(num_nodes, self.in_channels))
            count = 0
            for i in range(len(deg)):
                agg_m = tlx.softmax(beta_msg[count:count + int(deg[i])], axis=0) * msg[count:count + int(deg[i])]
                agg_m = tlx.reduce_sum(agg_m, axis=0)
                out[i] = tlx.convert_to_numpy(agg_m)
            after_agg_msg = tlx.convert_to_tensor(out, dtype=tlx.float32)
            return after_agg_msg
        elif aggr == 'powermean':
            out = np.zeros(shape=(num_nodes, self.in_channels))
            count = 0
            for i in range(len(deg)):
                agg_m = tlx.pow(msg[count:count + int(deg[i])], self.p)
                agg_m = tlx.reduce_sum(agg_m, axis=0) / deg[i]
                out[i] = tlx.convert_to_numpy(agg_m)
            out = tlx.pow(tlx.convert_to_tensor(out), 1 / self.p)
            after_agg_msg = tlx.convert_to_tensor(out)
            after_agg_msg = tlx.cast(after_agg_msg, dtype=tlx.float32)
            return after_agg_msg
        else:
            raise NotImplementedError('Not support for this opearator')

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):
        out = self.propagate(x, edge_index, aggr=self._agg, fuse_kernel=False)

        # msg normalization
        msg = tlx.l2_normalize(out, axis=-1)
        l2_norm_x = tlx.sqrt(tlx.reduce_sum(tlx.square(x), axis=1))
        l2_norm_x = tlx.reshape(l2_norm_x, shape=(tlx.get_tensor_shape(msg)[0], 1))
        x = x + msg * l2_norm_x * self.scale

        x = self.mlp.forward(x)
        return x

