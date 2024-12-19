import tensorlayerx as tlx
from gammagl.models.gcn import GCNModel


class CAGCNModel(tlx.nn.Module):
    r"""calibration GCN proposed in `"Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration"
        <https://arxiv.org/pdf/2109.14285.pdf>`_ paper.
    
        Parameters
        ----------
        base_model: nn.Module
            the model to calibrate.
        feature_dim: int
            input feature dimension.
        num_class: int
            number of classes, namely output dimension.
        drop_rate: float
            dropout rate.
        hidden_dim: int
            hidden dimension.
        num_layers: int, optional
            number of layers.
        norm: str, optional
            apply the normalizer.
        name: str, optional
            model name.

    """

    def __init__(self,
                 base_model,
                 feature_dim,
                 num_class,
                 drop_rate,
                 num_layers=2,
                 hidden_dim=64,
                 norm='both',
                 name=None):
        super().__init__()
        self.base_model = base_model
        self.cal_model = GCNModel(feature_dim, hidden_dim, num_class, drop_rate, num_layers, norm, name)

    def forward(self, cal_edge_index, cal_edge_weight, cal_num_nodes, *args, **kwargs):
        """
            The forward function of CAGCN.

            Parameters
            ----------
            cal_edge_index:
                cal_model's edge index
            cal_edge_weight:
                cal_model's edge weight, if None set all weight 1.0
                default weight setting in gcn_conv
            cal_num_nodes:
                cal_model's num nodes
            args && kwargs:
                all paras required for base_model
        """
        logits = self.base_model(*args, **kwargs)
        t = self.cal_model(logits, cal_edge_index, cal_edge_weight, cal_num_nodes)
        t = tlx.nn.Softplus()(t)
        output = logits * t
        return output
