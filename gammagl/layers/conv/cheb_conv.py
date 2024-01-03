from typing import Optional

import tensorlayerx as tlx

from gammagl.layers.conv import MessagePassing
from gammagl.utils.get_laplacian import get_laplacian
from gammagl.utils.loop import add_self_loops, remove_self_loops


class ChebConv(MessagePassing):
    r"""The chebyshev spectral graph convolutional operator from the
       `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
       Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

       .. math::
           \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
           \mathbf{\Theta}^{(k)}

       where :math:`\mathbf{Z}^{(k)}` is computed recursively by

       .. math::
           \mathbf{Z}^{(1)} &= \mathbf{X}

           \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

           \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
           \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

       and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
       :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

       Args:
           in_channels (int): Size of each input sample
           out_channels (int): Size of each output sample.
           K (int): Chebyshev filter size :math:`K`.
           normalization (str, optional): The normalization scheme for the graph
               Laplacian (default: :obj:`"sym"`):

               1. :obj:`None`: No normalization
               :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

               2. :obj:`"sym"`: Symmetric normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}`

               3. :obj:`"rw"`: Random-walk normalization
               :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

               You need to pass :obj:`lambda_max` to the :meth:`forward` method of
               this operator in case the normalization is non-symmetric.
           **kwargs (optional): Additional arguments of
               :class:`gammagl.layers.conv.MessagePassing`.

       Shapes:
           - **input:**
             node features :math:`(|\mathcal{V}|, F_{in})`,
             edge indices :math:`(2, |\mathcal{E}|)`,
             edge weights :math:`(|\mathcal{E}|)` *(optional)*,
             maximum :obj:`lambda` value :math:`(|\mathcal{G}|)` *(optional)*
           - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

       """

    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional = 'sym', **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv, self).__init__()

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = tlx.nn.ModuleList([
            tlx.layers.Linear(in_features=in_channels, out_features=out_channels) for _ in
            range(K)
        ])

    def __normal__(self, edge_index, num_nodes: Optional[int],
                   edge_weight, normalization: Optional[str],
                   lambda_max, batch=None):
        edge_index, edge_weight = remove_self_loops(edge_index,
                                                    edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index=tlx.convert_to_tensor(edge_index), num_nodes=num_nodes,
                                                edge_weight=tlx.convert_to_tensor(edge_weight),
                                                normalization=normalization)
        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_index, edge_weight = add_self_loops(edge_index=tlx.convert_to_tensor(edge_index),
                                                 edge_attr=edge_weight,
                                                 fill_value=-1,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None
        return edge_index, edge_weight

    def forward(self, x, edge_index, num_nodes, edge_weight: Optional = None, lambda_max: Optional = None,
                batch: Optional = None):
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        if lambda_max is None:
            lambda_max = tlx.convert_to_tensor(2.0)
        else:
            lambda_max = tlx.convert_to_tensor(lambda_max)
        assert lambda_max is not None
        edge_index, normal = self.__normal__(edge_index, num_nodes,
                                             edge_weight, self.normalization,
                                             lambda_max, batch=batch)
        Tx_0 = x
        Tx_1 = x
        out = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propagate(x=x, edge_index=edge_index, edge_weight=normal)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(x=Tx_1, edge_index=edge_index, edge_weight=normal)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out
