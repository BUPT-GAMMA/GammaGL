
import os
from tensorlayerx.model import WithLoss, TrainOneStep
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import argparse
from utils import init_params, seed_everything, split
from evaluation_test import node_evaluation
from model import GCN,EigenMLP, SpaSpeNode, Encoder, Basic, SAN
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse.linalg
import tensorlayerx as tlx
from gammagl.utils import to_scipy_sparse_matrix
import networkx as nx
from facebook_data import FacebookPagePage

def connected_components(sparse_adj):
    G = nx.from_scipy_sparse_array(sparse_adj)
    cc = nx.connected_components(G)

    components = []
    lens = []

    for c in cc:
        c = list(c)
        components.append(c)
        lens.append(len(c))

    return lens, components
def compute_laplacian(data):

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    row, col = edge_index
    data_adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    degree = np.array(data_adj.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(degree)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    I = csr_matrix(np.eye(num_nodes))
    D_inv_sqrt = csr_matrix((deg_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))))
    L = I - D_inv_sqrt.dot(data_adj).dot(D_inv_sqrt)
    e, u = scipy.sparse.linalg.eigsh(L, k=100, which='SM', tol=1e-3)
    print("e:",e)
    print("u:",u)
    adj = to_scipy_sparse_matrix(data.edge_index)
    lens, components = connected_components(adj)
    print("adj:",adj)
    print("lens:",lens)
    data.e = tlx.convert_to_tensor(e, dtype=tlx.float32)
    data.u = tlx.convert_to_tensor(u, dtype=tlx.float32)

    return data, lens, components


class ContrastiveLoss(WithLoss):
    def __init__(self, model, temp=1.0):
        super(ContrastiveLoss, self).__init__(backbone=model, loss_fn=None)
        self.temp = temp

    def forward(self, data, label):
        h_node_spa, h_node_spe = self.backbone_network(data['x'], data['edge_index'], data['e'], data['u'])
        h1 = tlx.l2_normalize(h_node_spa, axis=-1, eps=1e-12)
        h2 = tlx.l2_normalize(h_node_spe, axis=-1, eps=1e-12)
        logits = tlx.matmul(h1, h2.transpose(-2, -1)) / self.temp
        labels = tlx.arange(start=0, limit=h1.shape[0], delta=1, dtype=tlx.int64)
        loss = 0.5 * tlx.losses.softmax_cross_entropy_with_logits(logits, labels) + 0.5 * tlx.losses.softmax_cross_entropy_with_logits(logits.transpose(-2, -1), labels)
        return loss
def main(args):
    global edge, e, u, test_idx
    seed_everything(args.seed)
    print(args.dataset)
    if args.dataset in ['pubmed-3', 'flickr', 'arxiv', 'wiki', 'facebook']:
        dataset = FacebookPagePage(root='data/facebook')
        data = dataset[0]
        data, lens, components = compute_laplacian(data)

        x = tlx.convert_to_tensor(data.x, dtype=tlx.float32)
        edge = tlx.convert_to_tensor(data.edge_index, dtype=tlx.int64)
        e = tlx.convert_to_tensor(data.e[:args.spe_dim], dtype=tlx.float32)
        u = tlx.convert_to_tensor(data.u[:, :args.spe_dim], dtype=tlx.float32)
        y = tlx.convert_to_tensor(data.y)
        print(y.min().item(), y.max().item())


        if 'train_mask' in data.keys:
            if len(data.train_mask.size()) > 1:
                train_idx = tlx.where(data.train_mask[:, args.seed])[0]
                val_idx = tlx.where(data.val_mask[:, args.seed])[0]
                test_idx = tlx.where(data.test_mask)[0]
            else:
                train_idx = tlx.where(data.train_mask)[0]
                val_idx = tlx.where(data.val_mask)[0]
                test_idx = tlx.where(data.test_mask)[0]
        else:
            train_idx, val_idx, test_idx = split(y)

    else:
        pass

    print('test_idx:',len(test_idx))

    # 初始化空间编码器和谱编码器
    # spa_encoder = GCN(x.size(1), args.hidden_dim, args.hidden_dim)
    spa_encoder = Encoder(x.size(1), args.hidden_dim, args.hidden_dim)
    spe_encoder = EigenMLP(args.spe_dim, args.hidden_dim, args.hidden_dim, args.period)
    #spe_encoder = Basic(args.spe_dim, args.hidden_dim, args.hidden_dim).to(device)
    #spe_encoder = SAN(args.spe_dim, args.hidden_dim, args.hidden_dim).to(device)

    model = SpaSpeNode(spa_encoder, spe_encoder, hidden_dim=args.hidden_dim, t=args.t)
    model.apply(init_params)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    train_weights = model.trainable_weights
    loss_func = ContrastiveLoss(model, temp=args.t)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data_all = {
        'x': data.x,
        'edge_index': data.edge_index,
        'e': data.e,
        'u': data.u,
    }

    t1 = time.time()
    for i in range(1000):
        model.set_eval()
        spe_emb = model.spe_encoder(e, u).detach()
    t2 = time.time()
    print("t2-t1:",t2 - t1)

    for idx in range(100):
        model.set_train()
        loss = train_one_step(data=data_all, label=data.y)
        if (idx+1) % 10 == 0:
            model.set_eval()
            spa_emb = model.spa_encoder(x, edge).detach()
            spe_emb = model.spe_encoder(e, u).detach()
            acc, pred = node_evaluation((spa_emb + spe_emb)/2, y, train_idx, val_idx, test_idx)
            #acc, pred = node_evaluation(torch.cat((spa_emb, spe_emb), dim=-1), y, train_idx, val_idx, test_idx)
            #acc, pred = node_evaluation(spe_emb, y, train_idx, val_idx, test_idx)

            print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--dataset', default='wiki')
    parser.add_argument('--spe_dim', type=int, default=100)
    parser.add_argument('--period', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
