import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import ieHGCNConv
from tensorlayerx import elu

class ieHGCNModel(tlx.nn.Module):
    r"""ie-HGCN from paper `"Interpretable and Efficient Heterogeneous Graph Convolutional Network"
        <https://arxiv.org/pdf/2005.13183.pdf>`_ paper.


        Parameters
        ----------
        num_layers : int
            number of layers.
        in_channels : int or dict
            input feature dimensions of different input nodes.
        hidden_channels: list
            hidden dimensions of different layers.
        out_channels : int
            number of the target type node.
        attn_channels : int
            the dimension of attention vector.
        metadata : Tuple[List[str], List[Tuple[str, str, str]]]
            The metadata of the heterogeneous graph, *i.e.* its node and edge types given by a list of strings and a list of string triplets, respectively. See :metadata:`gammagl.data.HeteroGraph.metadata` for more information.
        batchnorm : bool, optional
            whether we need batchnorm.
        add_bias : bool, optional
            whether we need bias vector.
        activation : function, optional
            the activation function.
        dropout_rate : float, optional
            the drop out rate.
        name : str, optional
            model name.

    """
    def __init__(self,
                num_layers,
                in_channels,
                hidden_channels,
                out_channels,
                attn_channels,
                metadata,
                batchnorm=False,
                add_bias=False,
                activation=elu,
                dropout_rate=0.0,
                name=None,
                ):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.activation = elu

        if self.num_layers == 1:
            self.conv = nn.ModuleList([ieHGCNConv(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  attn_channels=attn_channels,
                                                  metadata=metadata,
                                                  batchnorm=batchnorm,
                                                  add_bias=add_bias,
                                                  activation=activation,
                                                  dropout_rate=dropout_rate)])
        else:
            self.conv = nn.ModuleList([ieHGCNConv(in_channels=in_channels,
                                                  out_channels=hidden_channels[0],
                                                  attn_channels=attn_channels,
                                                  metadata=metadata,
                                                  batchnorm=batchnorm,
                                                  add_bias=add_bias,
                                                  activation=activation,
                                                  dropout_rate=dropout_rate)])
            for i in range(1, num_layers - 1):
                self.conv.append(ieHGCNConv(in_channels=hidden_channels[i-1],
                                            out_channels=hidden_channels[i],
                                            attn_channels=attn_channels,
                                            metadata=metadata,
                                            batchnorm=batchnorm,
                                            add_bias=add_bias,
                                            activation=activation,
                                            dropout_rate=dropout_rate))
            self.conv.append(ieHGCNConv(in_channels=hidden_channels[-1],
                                        out_channels=out_channels,
                                        attn_channels=attn_channels,
                                        metadata=metadata,
                                        batchnorm=batchnorm,
                                        add_bias=add_bias,
                                        activation=activation,
                                        dropout_rate=dropout_rate))

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        for i in range(self.num_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict, num_nodes_dict)
        return x_dict
