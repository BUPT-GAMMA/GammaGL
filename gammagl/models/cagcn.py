# import os
# os.environ['TL_BACKEND'] = 'torch'

"""
GCN 原样
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import GCNConv

"""

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.models.gcn import GCNModel


class CAGCNModel(tlx.nn.Module):
    r"""
    calibration GCN proposed in `Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration`_.

    .. _Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration:
        https://arxiv.org/pdf/2109.14285.pdf
    
    Parameters
    ----------
        base_model (tlx.nn.Module): the model to calibrate 
        feature_dim (int): input feature dimension
        num_class (int): number of classes, namely output dimension
        drop_rate (float): dropout rate
        hidden_dim (int): hidden dimension
        num_layers (int): number of layers
        norm (str): apply the normalizer
        name (str): model name
    """
    def __init__(self, 
                 base_model,
                 feature_dim,
                 num_class,
                 drop_rate,
                 num_layers = 2,
                 hidden_dim = 64,
                 norm = 'both',
                 name=None):
        super().__init__()
        self.base_model = base_model
        self.cal_model = GCNModel(feature_dim, hidden_dim, num_class, drop_rate, num_layers, norm, name)
        
        # if necessary, reload trainable_weights by pre_trainable_weights
        # self.pre_trainable_weights = self.trainable_weights
        # for w in self.trainable_weights:
        #     self.nontrainable_weights.append(w)
        # self.trainable_weights = None


    """
    Parameters
    ----------
        cal_edge_index:     cal_model's edge index
        cal_edge_weight:    cal_model's edge weight, if None set all weight 1.0
                            default weight setting in gcn_conv
        cal_num_nodes:      cal_model's num nodes
        args && kwargs:     all paras required for base_model
    """
    def forward(self, cal_edge_index, cal_edge_weight, cal_num_nodes, *args, **kwargs):
        # print("extra1:",args)
        # print("extra2:", kwargs)
        logits = self.base_model(*args, **kwargs)
        t = self.cal_model(logits, cal_edge_index, cal_edge_weight, cal_num_nodes)
        t = tlx.nn.Softplus()(t)
        output = logits * t
        return output


# if __name__ == '__main__':
    # net = tlx.nn.Module()
    # x = tlx.nn.Parameter(data=tlx.ones((5,5)), requires_grad=False)
    # net.trainable_weights.append(x)
    # print(net.trainable_weights)

    # net = tlx.nn.Input([8, 100, 1], name='input')
    # conv1d = tlx.nn.Conv1d(out_channels =32, kernel_size=5, stride=2, b_init=None, in_channels=1, name='conv1d_1')
    # print(conv1d)
    # tensor = tlx.nn.Conv1d(out_channels =32, kernel_size=5, stride=2, act=tlx.ReLU, name='conv1d_2')(net)
    # print(tensor)
