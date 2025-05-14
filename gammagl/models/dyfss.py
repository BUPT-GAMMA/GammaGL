r"""Implementation of DyFSS model from the `"Dynamic Feature Selection and Fusion Strategy for Deep Graph Clustering"
<https://arxiv.org/abs/2309.08493>`_ paper.

This model combines multiple self-supervised learning tasks with a dynamic feature selection mechanism for graph clustering.
"""


import numpy as np
from sklearn.cluster import KMeans
import scipy.sparse as sp
from deeprobust.graph.utils import to_tensor, normalize_adj_tensor
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.nn import Module, Linear, Sequential, Dropout

from examples.dyfss.utils.metric import cluster_accuracy
from examples.dyfss.utils.losses import re_loss_func, dis_loss
from examples.dyfss.selfsl import *
from examples.dyfss.selfsl import NaiveGate
from examples.dyfss.utils.layers import GraphConvolution


class EmptyData:
    def __init__(self):
        self.adj = None
        self.features = None
        self.adj_norm = None
        self.features_norm = None
        self.labels = None
        self.adj_label = None
        self.norm = None
        self.pos_weight = None


class MoeSSL(tlx.nn.Module):
    r"""Mixture of Experts SSL model for graph clustering.

    Parameters
    ----------
    data : object
        Input graph data object.
    encoder : VGAE
        Variational Graph Auto-Encoder model.
    modeldis : Discriminator
        Discriminator model for adversarial training.
    set_of_ssl : list
        List of SSL task names.
    args : argparse.Namespace
        Model configuration arguments.
    """
    def __init__(self, data, encoder, modeldis, set_of_ssl, args, **kwargs):
        super(MoeSSL, self).__init__()
        self.args = args
        self.n_tasks = len(set_of_ssl)
        self.set_of_ssl = set_of_ssl
        self.name = "MoeSSL"
        self.encoder = encoder
        self.modeldis = modeldis
        self.weight = tlx.ones((len(set_of_ssl),), dtype=tlx.float32)
        self.gate = NaiveGate(args.hid_dim[1], self.n_tasks, self.args.top_k)
        self.ssl_agent = []
        self.optimizer_dec = None
        self.optimizer_dis = None
        self.data = data
        self.processed_data = EmptyData()
        self.data_process()
        self.params = None
        self.setup_ssl(set_of_ssl)

    def data_process(self):
        features, adj, labels = to_tensor(self.data.features,
                                            self.data.adj, self.data.labels)
        
        adj_norm = normalize_adj_tensor(adj, sparse=True)
        self.processed_data.adj_norm = adj_norm
        self.processed_data.features = features
        self.processed_data.labels = labels
    
        # Create adjacency label matrix (with self-loops)
        adj_label = self.data.adj + sp.eye(self.data.adj.shape[0])
        self.processed_data.adj_label = tlx.convert_to_tensor(adj_label.toarray(), dtype=tlx.float32)
        
        # Calculate positive weight for loss balancing
        self.processed_data.pos_weight = tlx.convert_to_tensor(
            [float(self.data.adj.shape[0] * self.data.adj.shape[0] - self.data.adj.sum()) / self.data.adj.sum()], 
            dtype=tlx.float32)
            
        # Calculate normalization constant
        norm = self.data.adj.shape[0] * self.data.adj.shape[0] / float(
            (self.data.adj.shape[0] * self.data.adj.shape[0] - self.data.adj.sum()) * 2)
        self.processed_data.norm = norm

    def setup_ssl(self, set_of_ssl):
        args = self.args
        self.params = list(self.encoder.trainable_weights)
        
        for ix, ssl in enumerate(set_of_ssl):
            if tlx.BACKEND == 'tensorflow':
                import tensorflow as tf
                with tf.name_scope(f"ssl_agent_{ix}"):
                    agent = eval(ssl)(data=self.data,
                                    processed_data=self.processed_data,
                                    encoder=self.encoder,
                                    nhid1=self.args.hid_dim[0],
                                    nhid2=self.args.hid_dim[1],
                                    dropout=0.0,
                                    args=args)
            else:
                agent = eval(ssl)(data=self.data,
                                processed_data=self.processed_data,
                                encoder=self.encoder,
                                nhid1=self.args.hid_dim[0],
                                nhid2=self.args.hid_dim[1],
                                dropout=0.0,
                                args=args)
            
            self.ssl_agent.append(agent)
            if agent.disc1 is not None:
                if tlx.BACKEND == 'tensorflow':
                    clean_weights = []
                    for weight in agent.disc1.trainable_weights:
                        if hasattr(weight, 'name') and '/' in weight.name:
                            import tensorflow as tf
                            new_name = weight.name.replace('/', '_')
                            new_var = tf.Variable(weight.value(), name=new_name)
                            clean_weights.append(new_var)
                        else:
                            clean_weights.append(weight)
                    self.params = self.params + clean_weights
                else:
                    self.params = self.params + list(agent.disc1.trainable_weights)
    
    def FeatureFusionForward(self, z_embeddings):
        nodes_weight_ori, loss_balan = self.gate(z_embeddings, 0)
        ssl_embeddings_list = []
        
        # Get embeddings from each SSL agent
        for ix, ssl in enumerate(self.ssl_agent):
            ssl.set_train()
            ssl_embeddings = ssl.gcn2_forward(z_embeddings, self.processed_data.adj_norm)
            ssl_embeddings_list.append(ssl_embeddings)
            
        try:
            ssl_emb_tensor = tlx.stack(ssl_embeddings_list)
            ssl_emb_tensor = tlx.transpose(ssl_emb_tensor, [1, 0, 2])
        except Exception as e:
            print(f"Error in stacking or transposing tensors: {e}")
            import numpy as np
            ssl_emb_list_np = [tlx.convert_to_numpy(emb) for emb in ssl_embeddings_list]
            ssl_emb_tensor_np = np.stack(ssl_emb_list_np)
            ssl_emb_tensor_np = np.transpose(ssl_emb_tensor_np, (1, 0, 2))
            ssl_emb_tensor = tlx.convert_to_tensor(ssl_emb_tensor_np)
        
        try:
            nodes_weight = tlx.expand_dims(nodes_weight_ori, 2)
            nodes_weight = tlx.tile(nodes_weight, [1, 1, self.args.hid_dim[1]])
        except Exception as e:
            print(f"Error in expanding dimensions: {e}")
            import numpy as np
            nodes_weight_np = tlx.convert_to_numpy(nodes_weight_ori)
            nodes_weight_np = np.expand_dims(nodes_weight_np, 2)
            nodes_weight_np = np.tile(nodes_weight_np, [1, 1, self.args.hid_dim[1]])
            nodes_weight = tlx.convert_to_tensor(nodes_weight_np)
        
        # Weighted sum to get fusion embeddings
        try:
            fusion_emb = tlx.reduce_sum(ssl_emb_tensor * nodes_weight, axis=1)
        except Exception as e:
            print(f"Error in weighted sum: {e}")
            import numpy as np
            ssl_emb_tensor_np = tlx.convert_to_numpy(ssl_emb_tensor)
            nodes_weight_np = tlx.convert_to_numpy(nodes_weight)
            fusion_emb_np = np.sum(ssl_emb_tensor_np * nodes_weight_np, axis=1)
            fusion_emb = tlx.convert_to_tensor(fusion_emb_np)
        
        return fusion_emb, loss_balan, nodes_weight_ori

    def get_ssl_loss_stage_one(self, x):
        loss = 0
        for ix, ssl in enumerate(self.ssl_agent):
            ssl.set_train()
            loss = loss + ssl.make_loss_stage_one(x)
        return loss

    def get_ssl_loss_stage_two(self, x, adj):
        loss = 0
        for ix, ssl in enumerate(self.ssl_agent):
            ssl.set_train()
            ssl_loss = ssl.make_loss_stage_two(x, adj)
            loss = loss + ssl_loss
        return loss


class VGAE(Module):
    r"""Variational Graph Auto-Encoder implementation.
    
    Parameters
    ----------
    input_feat_dim : int
        Dimension of input features.
    hidden_dim1 : int
        Dimension of first hidden layer.
    hidden_dim2 : int
        Dimension of second hidden layer.
    dropout : float
        Dropout rate.
    cluster_num : int
        Number of clusters.
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, cluster_num):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=tlx.nn.ReLU())
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        backend = tlx.BACKEND
        
        x = tlx.convert_to_tensor(x)
        
        if isinstance(adj, tuple):
            indices, values, shape = adj
            indices = tlx.convert_to_tensor(indices)
            values = tlx.convert_to_tensor(values)
        elif backend == 'tensorflow':
            import tensorflow as tf
            import torch
            if hasattr(adj, 'indices') and hasattr(adj, 'values') and hasattr(adj, 'dense_shape'):
                pass
            elif isinstance(adj, torch.Tensor):
                if adj.is_sparse:
                    adj = adj.to_dense()
                import numpy as np
                adj = tlx.convert_to_tensor(adj.detach().cpu().numpy())
            elif hasattr(adj, 'todense'):
                import numpy as np
                adj = tlx.convert_to_tensor(np.array(adj.todense(), dtype='float32'))
            else:
                adj = tlx.convert_to_tensor(adj)
        elif backend == 'torch':
            import torch
            if isinstance(adj, torch.Tensor) and adj.is_sparse:
                adj = adj.to_dense()
                adj = tlx.convert_to_tensor(adj)
            elif hasattr(adj, 'todense'):
                import numpy as np
                adj = tlx.convert_to_tensor(np.array(adj.todense(), dtype='float32'))
            else:
                adj = tlx.convert_to_tensor(adj)
        else:
            if hasattr(adj, 'todense'):
                import numpy as np
                adj = tlx.convert_to_tensor(np.array(adj.todense(), dtype='float32'))
            else:
                adj = tlx.convert_to_tensor(adj)
        
        hidden1 = self.gc1(x, adj)
        return hidden1, self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.is_train:
            std = tlx.exp(logvar)
            eps = tlx.random_normal(shape=std.shape, dtype=std.dtype)
            return eps * std + mu
        else:
            return mu

    def forward(self, x, adj):
        hidden, mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return hidden, self.dc(z), mu, logvar, z


class Discriminator(Module):
    r"""Discriminator network for adversarial training.
    
    Parameters
    ----------
    hidden_dim1 : int
        Dimension of first hidden layer.
    hidden_dim2 : int
        Dimension of second hidden layer.
    hidden_dim3 : int
        Dimension of third hidden layer.
    """
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Discriminator, self).__init__()
        
        self.dis = tlx.nn.Sequential([
            Linear(out_features=hidden_dim3, in_features=hidden_dim2, 
                  W_init=tlx.initializers.HeNormal()),
            tlx.nn.ReLU(),
            Linear(out_features=hidden_dim1, in_features=hidden_dim3, 
                  W_init=tlx.initializers.HeNormal()),
            tlx.nn.ReLU(),
            Linear(out_features=1, in_features=hidden_dim1, 
                  W_init=tlx.initializers.HeNormal())
        ])

    def forward(self, x):
        tlx.set_seed(1)
        z = self.dis(x)
        return z


class InnerProductDecoder(Module):
    def __init__(self, dropout=0., act=tlx.sigmoid, name=None):
        super(InnerProductDecoder, self).__init__(name=name)
        self.dropout = dropout
        self.act = act
        self.dropout_layer = Dropout(p=dropout)

    def forward(self, z):

        z = self.dropout_layer(z)
        backend = tlx.BACKEND
        
        if backend == 'tensorflow':
            import tensorflow as tf
            adj = tf.matmul(z, tf.transpose(z))
        else:
            adj = tlx.matmul(z, tlx.transpose(z))
        
        return self.act(adj)
