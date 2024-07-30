from gammagl.layers.conv import GCNConv
import tensorlayerx.nn as nn
import tensorlayerx as tlx
from gammagl.utils import mask_to_index
import numpy as np
import random


class GCN_encoder(tlx.nn.Module):
    def __init__(self, args):
        super(GCN_encoder, self).__init__()
        self.args = args
        self.conv = GCNConv(args.num_features, args.hidden)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index, edge_weight=tlx.ones((edge_index.shape[1],), dtype=tlx.float32), num_nodes=x.shape[0])

        return h


class MLP_discriminator(tlx.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args    
        self.lin = nn.Linear(in_features=args.hidden, out_features=1)
    
    def forward(self, h):
        h = self.lin(h)

        return tlx.sigmoid(h)


class MLP_classifier(tlx.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args
        self.lin = nn.Linear(in_features=args.hidden, out_features=1)

    def forward(self, h):
        h = self.lin(h)

        return h


class FatraGNNModel(tlx.nn.Module):
    r"""FatraGNN from `"Graph Fairness Learning under Distribution Shifts"
        <https://arxiv.org/abs/2401.16784>`_ paper.
        
        Parameters
        ----------
        in_features: int
            input feature dimension.
        hidden: int
            hidden dimension.
        out_features: int
            number of output feature dimension.
        drop_rate: float
            dropout rate.
    """
    def __init__(self, args):
        super(FatraGNNModel, self).__init__()
        self.classifier = MLP_classifier(args)
        self.graphEdit = Graph_Editer(1, args.num_features, args.device)
        self.encoder = GCN_encoder(args)
        self.discriminator = MLP_discriminator(args)
    def forward(self, x, edge_index, flag):
        if flag==0:
            h = self.encoder(x, edge_index)
            output = self.classifier(h)
            return output

        if flag==1:
            edge_index = edge_index['edge_index']
            h = self.encoder(x, edge_index)
            output = self.discriminator(h)

        elif flag==2:
            h = self.encoder(x, edge_index['edge_index'])
            h = self.classifier(h)
            output = tlx.sigmoid(h)

        elif flag==3:
            h = self.encoder(x, edge_index['edge_index'])
            output = self.discriminator(h)

        elif flag==4:
            x2 = self.graphEdit(x)
            h2 = self.encoder(x2, edge_index['edge_index2'])
            h2 = tlx.l2_normalize(h2, axis=1)
            output = self.classifier(h2)

        elif flag==5:
            x2 = self.graphEdit(x)
            h2 = self.encoder(x2, edge_index['edge_index2'])
            h1 = self.encoder(x, edge_index['edge_index'])
            h2 = tlx.l2_normalize(h2, axis=1)
            h1 = tlx.l2_normalize(h1, axis=1)
            output = {
                'h1': h1,
                'h2': h2,
            }
 
        return output

class Graph_Editer(tlx.nn.Module):
    def __init__(self, n, a, device):
        super(Graph_Editer, self).__init__()
        self.transFeature = nn.Linear(in_features=a, out_features=a)
        self.device = device
        self.seed = 13
 

    def modify_structure1(self, edge_index, A2_edge, sens, nodes_num, drop=0.8, add=0.3):
        random.seed(self.seed)
        src_node, targ_node = edge_index[0], edge_index[1]
        matching = tlx.gather(sens, src_node) == tlx.gather(sens, targ_node)
        
        yipei = mask_to_index(matching == False)
        drop_index = tlx.convert_to_tensor(random.sample(range(yipei.shape[0]), int(yipei.shape[0] * drop)))
        yipei_drop = tlx.gather(yipei, drop_index)
        keep_indices0 = tlx.ones(src_node.shape, dtype=tlx.bool)
        keep_indices = tlx.scatter_update(keep_indices0, yipei_drop, tlx.zeros((yipei_drop.shape), dtype=tlx.bool))
        n_src_node = src_node[keep_indices]
        n_targ_node = targ_node[keep_indices]
        
        src_node2, targ_node2 = A2_edge[0], A2_edge[1]
        matching2 = tlx.gather(sens, src_node2) == tlx.gather(sens, targ_node2)
        matching3 = src_node2 == targ_node2
        tongpei = mask_to_index(tlx.logical_and(matching2 == True, matching3 == False) == True)
        add_index = tlx.convert_to_tensor(random.sample(range(tongpei.shape[0]), int(yipei.shape[0] * drop)))
        tongpei_add = tlx.gather(tongpei, add_index)
        keep_indices0 = tlx.zeros(src_node2.shape, dtype=tlx.bool)
        keep_indices = tlx.scatter_update(keep_indices0, tongpei_add, tlx.ones((tongpei_add.shape), dtype=tlx.bool))
        
        a_src_node = src_node2[keep_indices]
        a_targ_node = targ_node2[keep_indices]

        m_src_node = tlx.concat((a_src_node, n_src_node), axis=0)
        m_targ_node = tlx.concat((a_targ_node, n_targ_node), axis=0)
        n_edge_index = tlx.concat((tlx.expand_dims(m_src_node, axis=1), tlx.expand_dims(m_targ_node, axis=1)), axis=1)
        return n_edge_index
    

    def modify_structure2(self, edge_index, A2_edge, sens, nodes_num, drop=0.6, add=0.3):
        random.seed(self.seed)
        src_node, targ_node = edge_index[0], edge_index[1] 
        matching = tlx.gather(sens, src_node) == tlx.gather(sens, targ_node)
        

        yipei = mask_to_index(matching == False)
        yipei_np = tlx.convert_to_numpy(yipei)
        np.random.shuffle(yipei_np)
        yipei_shuffled = tlx.convert_to_tensor(yipei_np)
        
        drop_index = tlx.convert_to_tensor(random.sample(range(yipei_shuffled.shape[0]), int(yipei_shuffled.shape[0] * drop)))
        yipei_drop = tlx.gather(yipei_shuffled, drop_index)
        keep_indices0 = tlx.ones(src_node.shape, dtype=tlx.bool)
        keep_indices = tlx.scatter_update(keep_indices0, yipei_drop, tlx.zeros((yipei_drop.shape), dtype=tlx.bool))
        n_src_node = src_node[keep_indices]
        n_targ_node = targ_node[keep_indices]
        
        src_node2, targ_node2 = A2_edge[0], A2_edge[1]
        matching2 = tlx.gather(sens, src_node2) != tlx.gather(sens, targ_node2)
        matching3 = src_node2 == targ_node2
        tongpei = mask_to_index(tlx.logical_and(matching2 == True, matching3 == False) == True)
        tongpei_np = tlx.convert_to_numpy(tongpei)
        np.random.shuffle(tongpei_np)
        tongpei_shuffled = tlx.convert_to_tensor(tongpei_np)
        add_index = tlx.convert_to_tensor(random.sample(range(tongpei_shuffled.shape[0]), int(yipei_shuffled.shape[0] * drop)))
        tongpei_add = tlx.gather(tongpei_shuffled, add_index)
        keep_indices0 = tlx.zeros(src_node2.shape, dtype=tlx.bool)
        keep_indices = tlx.scatter_update(keep_indices0, tongpei_add, tlx.ones((tongpei_add.shape), dtype=tlx.bool))
        
        a_src_node = src_node2[keep_indices]
        a_targ_node = targ_node2[keep_indices]

        m_src_node = tlx.concat((a_src_node, n_src_node), axis=0)
        m_targ_node = tlx.concat((a_targ_node, n_targ_node), axis=0)
        n_edge_index = tlx.concat((tlx.expand_dims(m_src_node, axis=1), tlx.expand_dims(m_targ_node, axis=1)), axis=1)

        return n_edge_index

    def forward(self, x):
        x1 = x + 0.1 * self.transFeature(x)

        return x1
    


