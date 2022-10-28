import tensorlayerx as tlx
import tlx.nn as nn
from gammagl.layers import MultiHead

class Unimp(tlx.nn.Module):
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