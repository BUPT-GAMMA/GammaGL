import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
import numpy as np
import argparse
import yaml
from yaml import SafeLoader
import tensorlayerx as tlx

from gammagl.datasets import Planetoid, Flickr, BlogCatalog
from gammagl.layers.conv import GCNConv
from gammagl.models import Grace_Spco_Encoder, Grace_Spco_Model
from gammagl.transforms import DropEdge
from gammagl.data import Graph
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss

from eval import label_classification
import scipy.sparse as ssp

class train_loss(WithLoss):
    def __init__(self, model):
        super(train_loss, self).__init__(backbone=model, loss_fn=None)

    def forward(self, x, edge_index, edge_attr, ori_index, ori_attr,
                drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2):
        g = Graph(edge_index = edge_index, edge_attr = edge_attr)
        ori_g = Graph(edge_index = ori_index, edge_attr = ori_attr)
        g = DropEdge(p=drop_edge_rate_1)(g)
        ori_g = DropEdge(p=drop_edge_rate_2)(ori_g)
        edge_index_1 = g.edge_index
        edge_attr_1 = g.edge_attr
        edge_index_2 = ori_g.edge_index
        edge_attr_2 = ori_g.edge_attr
        x_1 = drop_feature(x, drop_feature_rate_1)
        x_2 = drop_feature(x, drop_feature_rate_2)
        z1 = self.backbone_network(x_1, edge_index_1, edge_attr_1)
        z2 = self.backbone_network(x_2, edge_index_2, edge_attr_2)

        loss = self.backbone_network.loss(z1, z2, batch_size=0)

        return loss

def drop_feature(x, drop_prob):
    drop_mask = tlx.random_uniform(shape=(tlx.get_tensor_shape(x)[1], )) < drop_prob
    drop_mask = tlx.reshape(drop_mask, (1, -1))
    drop_mask = tlx.tile(drop_mask, [tlx.get_tensor_shape(x)[0], 1])

    x_dropped = tlx.where(drop_mask, tlx.zeros_like(x), x)

    return x_dropped

def test(model, x, edge_index, edge_attr, y, idx_train, idx_val, idx_test, final=False):
    model.set_eval()
    z = model(x, edge_index, edge_attr)
    # z = model(x, edge_index)
    label_classification(z, y, idx_train, idx_val, idx_test)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ssp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return tlx.convert_to_tensor(features, dtype=tlx.float32)

def normalize_adj(adj, self_loop=False):
    """Symmetrically normalize adjacency matrix."""
    adj = ssp.coo_matrix(adj)
    rowsum = np.array(np.abs(adj.A).sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ssp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def sinkhorn(K, dist, sin_iter):
    # make the matrix sum to 1
    u = np.ones([len(dist), 1]) / len(dist)
    K_ = ssp.diags(1. / dist) * K
    dist = dist.reshape(-1, 1)
    ll = 0
    for it in range(sin_iter):        
        u = 1. / K_.dot(dist / (K.T.dot(u)))
    v = dist / (K.T.dot(u))
    delta = np.diag(u.reshape(-1)).dot(K).dot(np.diag(v.reshape(-1)))
    return delta    

def plug(theta, num_node, laplace, delta_add, delta_dele, epsilon, dist, sin_iter, c_flag=False):
    C = (1 - theta) * laplace.A
    if c_flag:
        C = laplace.A
    K_add = np.exp(2 * (C * delta_add).sum() * C / epsilon)
    K_dele = np.exp(-2 * (C * delta_dele).sum() * C / epsilon)

    delta_add = sinkhorn(K_add, dist, sin_iter)

    delta_dele = sinkhorn(K_dele, dist, sin_iter)
    return delta_add, delta_dele

def update(theta, epoch, total):
    theta = theta - theta*(epoch/total)
    return theta

def main(args):
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': tlx.relu, 'prelu': tlx.nn.activation.PRelu()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = args.num_epochs
    weight_decay = config['weight_decay']

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root="", name=args.dataset)
    elif args.dataset == 'flickr':
        dataset = Flickr(root="")
    elif args.dataset == 'blog':
        dataset = BlogCatalog(root="")

    graph = dataset[0]

    train_idx = mask_to_index(graph.train_mask)
    val_idx = mask_to_index(graph.val_mask)
    test_idx = mask_to_index(graph.test_mask)

    num_class = dataset.num_classes
    feat = preprocess_features(tlx.convert_to_numpy(tlx.to_device(graph.x, "cpu")))
    edge_index = graph.edge_index
    scope = add_self_loops(edge_index)[0]
    num_nodes = graph.num_nodes
    num_features = tlx.get_tensor_shape(feat)[-1]
    adj = ssp.coo_matrix((np.ones(tlx.get_tensor_shape(edge_index)[1]),
                          (tlx.to_device(edge_index[0, :], "cpu"), tlx.to_device(edge_index[1, :], "cpu"))),
                          shape=[num_nodes, num_nodes])
    laplace = ssp.eye(adj.shape[0]) - normalize_adj(adj)
    scope_matrix = ssp.coo_matrix((np.ones(tlx.get_tensor_shape(scope)[1]),
                                   (tlx.to_device(scope[0, :], "cpu"), tlx.to_device(scope[1, :], "cpu"))),
                                    shape = [num_nodes, num_nodes]).A
    dist = adj.A.sum(-1) / adj.A.sum()

    encoder = Grace_Spco_Encoder(num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers)
    model = Grace_Spco_Model(encoder, num_hidden, num_proj_hidden, tau)

    optimizer = tlx.optimizers.Adam(lr=config["learning_rate"], weight_decay=config["weight_decay"])
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = train_loss(model)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    theta = 1
    delta = np.ones(adj.shape) * args.delta_origin
    delta_add = delta
    delta_dele = delta

    new_adj = adj.tocsc()
    ori_index = edge_index
    ori_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]
    edge_index = ori_index
    edge_attr = ori_attr
    k = 0

    for epoch in range(num_epochs):
        model.set_train()
        loss = train_one_step(feat, edge_index, edge_attr, ori_index, ori_attr,
                     drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2)

        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % args.turn == 0:
            print("================================================")
            if args.dataset in ["cora", "citeseer"] and epoch != 0:
                delta_add, delta_dele = plug(theta, num_nodes, laplace, delta_add, delta_dele, args.epsilon, dist, args.sin_iter, True)
            else:
                delta_add, delta_dele = plug(theta, num_nodes, laplace, delta_add, delta_dele, args.epsilon, dist, args.sin_iter)
            delta = normalize_adj((delta_add - delta_dele) * scope_matrix)
            new_adj = adj + args.lam * delta

            edge = new_adj.nonzero()
            edge = np.vstack([edge[0], edge[1]])
            edge_index = tlx.convert_to_tensor(edge, dtype=tlx.int64)
            edge_attr = tlx.convert_to_tensor(new_adj[new_adj.nonzero()], dtype=tlx.float32)[0]
            theta = update(1, epoch, num_epochs+1)

    print("=== Final ===")
    test(model, feat, graph.edge_index, ori_attr, graph.y, train_idx, val_idx, test_idx, final=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="cora") 
    parser.add_argument('--gpu', type=int, default=0) 
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=0) 

    #####################################
    ## new parameters
    parser.add_argument('--delta_origin', type=float, default=0.5)
    parser.add_argument('--sin_iter', type=int, default=3) 

    parser.add_argument('--lam', type=float, default=1.0) 
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--turn', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--scope_flag', type=int, default=1)
    #####################################

    args, _ = parser.parse_known_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
