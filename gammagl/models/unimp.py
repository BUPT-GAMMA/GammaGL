import tensorlayerx as tlx
import tlx.nn as nn
from gammagl.layers import MultiHead

class Unimp(tlx.nn.Module):

    r"""The graph attentional operator from the `"Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    Parameters
    ----------
        dataset: 
            num_node_features: int
                Input feature dimension
            num_nodes: int
                Number of nodes
            x: [num_nodes, num_node_features]
                Feature of node
            edge_index: [2, num_edges] 
                Graph connectivity in COO format 
            edge_attr: [num_edges, num_edge_features]
                Edge feature matrix
            y: [1. *]
                Target to train against (may have arbitrary shape)
            pos: [num_nodes, num_dimensions]
                Node position matrix 
    """

    def __init__(self,dataset):
        super(Unimp, self).__init__()

        out_layer1=int(dataset.num_node_features/2)
        self.layer1=MultiHead(dataset.num_node_features+1, out_layer1, 4,dataset[0].num_nodes)
        self.norm1=nn.LayerNorm(out_layer1)
        self.relu1=nn.ReLU()

        self.layer2=MultiHead(out_layer1, dataset.num_classes, 4,dataset[0].num_nodes)
        self.norm2=nn.LayerNorm(dataset.num_classes)
        self.relu2=nn.ReLU()
    def forward(self, x, edge_index):
        out1 = self.layer1(x, edge_index)
        out2=self.norm1(out1)
        out3=self.relu1(out2)
        out4=self.layer2(out3,edge_index)
        out5 = self.norm2(out4)
        out6 = self.relu2(out5)
        return out6