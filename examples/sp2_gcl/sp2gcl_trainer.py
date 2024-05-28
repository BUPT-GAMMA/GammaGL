import os
# os.environ['TL_BACKEND'] = 'torch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from gammagl.utils import mask_to_index
from tensorlayerx.model import WithLoss, TrainOneStep
import argparse
from evaluation_test import node_evaluation
from gammagl.models import EigenMLP, SpaSpeNode, Encoder
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse.linalg
import tensorlayerx as tlx
from gammagl.datasets import FacebookPagePage,WikiCS,Planetoid



def split(node_labels):

    y = node_labels
    train_ratio = 0.1
    val_ratio = 0.1
    test_ratio = 0.8

    N = len(y)
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = tlx.convert_to_tensor(train_idx)
    val_idx = tlx.convert_to_tensor(val_idx)
    test_idx = tlx.convert_to_tensor(test_idx)

    return train_idx, val_idx, test_idx


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
    data.e = tlx.convert_to_tensor(e, dtype=tlx.float32)
    data.u = tlx.convert_to_tensor(u, dtype=tlx.float32)

    return data


class ContrastiveLoss(WithLoss):
    def __init__(self, model, temp=1.0):
        super(ContrastiveLoss, self).__init__(backbone=model, loss_fn=None)
        self.temp = temp

    def forward(self, data, label):
        h_node_spa, h_node_spe = self.backbone_network(data['x'], data['edge_index'], data['e'], data['u'])
        h1 = tlx.l2_normalize(h_node_spa, axis=-1, eps=1e-12)
        h2 = tlx.l2_normalize(h_node_spe, axis=-1, eps=1e-12)
        logits = tlx.matmul(h1, tlx.transpose(h2, perm=(1, 0))) / self.temp
        labels = tlx.arange(start=0, limit=h1.shape[0], delta=1, dtype=tlx.int64)
        loss = 0.5 * tlx.losses.softmax_cross_entropy_with_logits(logits, labels) + 0.5 * tlx.losses.softmax_cross_entropy_with_logits(logits.transpose(-2, -1), labels)
        return loss

def main(args):
    if args.dataset in ['pubmed', 'flickr', 'arxiv', 'wikics', 'facebook']:
        if args.dataset == 'facebook':
            dataset = FacebookPagePage(root='data/facebook')
        elif args.dataset == 'wikics':
            dataset = WikiCS(root='data/wikics')
        elif args.dataset == 'pubmed':
            dataset = dataset = Planetoid(root='', name='pubmed')
        data = dataset[0]
        data = compute_laplacian(data)
        x = tlx.convert_to_tensor(data.x, dtype=tlx.float32)
        edge = tlx.convert_to_tensor(data.edge_index, dtype=tlx.int64)
        e = tlx.convert_to_tensor(data.e[:args.spe_dim], dtype=tlx.float32)
        u = tlx.convert_to_tensor(data.u[:, :args.spe_dim], dtype=tlx.float32)
        y = tlx.convert_to_tensor(data.y)
        if 'train_mask' in data.keys:
            if len(data.train_mask.size()) > 1:
                train_idx = mask_to_index(data.train_mask)
                test_idx = mask_to_index(data.test_mask)
                val_idx = mask_to_index(data.val_mask)
            else:
                train_idx = tlx.where(data.train_mask)[0]
                val_idx = tlx.where(data.val_mask)[0]
                test_idx = tlx.where(data.test_mask)[0]
        else:
            train_idx, val_idx, test_idx = split(y)

    else:
        pass

    spa_encoder = Encoder(x.shape[1], args.hidden_dim, args.hidden_dim)
    spe_encoder = EigenMLP(args.spe_dim, args.hidden_dim, args.hidden_dim, args.period)
    model = SpaSpeNode(spa_encoder, spe_encoder, hidden_dim=args.hidden_dim, t=args.t)
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

    for epoch in range(args.num_epochs):
        model.set_train()
        loss = train_one_step(data=data_all, label=data.y)
        if (epoch + 1) % 10 == 0:
            model.set_eval()
            spa_emb = tlx.detach(model.spa_encoder(x, edge))
            spe_emb = tlx.detach(model.spe_encoder(e, u))
            acc, pred = node_evaluation((spa_emb + spe_emb)/2, y, train_idx, val_idx, test_idx)
            print(f'Epoch {epoch+1}/{args.num_epochs}, Accuracy: {acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--dataset', default='facebook')
    parser.add_argument('--spe_dim', type=int, default=100)
    parser.add_argument('--period', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    main(args)
