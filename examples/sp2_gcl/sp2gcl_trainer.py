import os
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from gammagl.utils import mask_to_index
from tensorlayerx.model import WithLoss, TrainOneStep
import argparse
from evaluation_test import node_evaluation
from gammagl.models import EigenMLP, SpaSpeNode, Encoder
from scipy.sparse import csr_matrix
import scipy.sparse.linalg
from gammagl.datasets import FacebookPagePage,WikiCS,Planetoid
import tensorlayerx as tlx
import numpy as np
from sklearn.model_selection import train_test_split


def get_train_val_test_split(graph, train_ratio, val_ratio):
    """
    Split the dataset into train, validation, and test sets.

    Parameters
    ----------
    graph :
        The graph to split.
    train_ratio : float
        The proportion of the dataset to include in the train split.
    val_ratio : float
        The proportion of the dataset to include in the validation split.

    Returns
    -------
    :class:`tuple` of :class:`tensor`
    """

    random_state = np.random.RandomState(0)
    num_samples = graph.num_nodes
    all_indices = np.arange(num_samples)

    # split into train and (val + test)
    train_indices, val_test_indices = train_test_split(
        all_indices, train_size=train_ratio, random_state=random_state
    )

    # calculate the ratio of validation and test splits in the remaining data
    test_ratio = 1.0 - train_ratio - val_ratio
    val_size_ratio = val_ratio / (val_ratio + test_ratio)

    # split val + test into validation and test sets
    val_indices, test_indices = train_test_split(
        val_test_indices, train_size=val_size_ratio, random_state=random_state
    )

    return generate_masks(num_samples, train_indices, val_indices, test_indices)


def generate_masks(num_nodes, train_indices, val_indices, test_indices):
    np_train_mask = np.zeros(num_nodes, dtype=bool)
    np_train_mask[train_indices] = 1
    np_val_mask = np.zeros(num_nodes, dtype=bool)
    np_val_mask[val_indices] = 1
    np_test_mask = np.zeros(num_nodes, dtype=bool)
    np_test_mask[test_indices] = 1

    train_mask = tlx.ops.convert_to_tensor(np_train_mask, dtype=tlx.bool)
    val_mask = tlx.ops.convert_to_tensor(np_val_mask, dtype=tlx.bool)
    test_mask = tlx.ops.convert_to_tensor(np_test_mask, dtype=tlx.bool)

    return train_mask, val_mask, test_mask


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
    if args.dataset in ['pubmed',  'wikics', 'facebook']:
        if args.dataset == 'facebook':
            dataset = FacebookPagePage(root=args.dataset_path)
        elif args.dataset == 'wikics':
            dataset = WikiCS(root=args.dataset_path)
        elif args.dataset == 'pubmed':
            dataset = Planetoid(root=args.dataset_path, name=args.dataset)
        data = dataset[0]
        data = compute_laplacian(data)
        e = tlx.convert_to_tensor(data.e[:args.spe_dim], dtype=tlx.float32)
        u = tlx.convert_to_tensor(data.u[:, :args.spe_dim], dtype=tlx.float32)
        if 'train_mask' in data.keys:
            train_mask = data.train_mask
            test_mask = data.test_mask
            val_mask = data.val_mask
        else:
            train_mask, val_mask, test_mask = get_train_val_test_split(data, 0.1, 0.1)
        train_idx = mask_to_index(train_mask)
        test_idx = mask_to_index(test_mask)
        val_idx = mask_to_index(val_mask)

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    spa_encoder = Encoder(data.x.shape[1], args.hidden_dim, args.hidden_dim)
    spe_encoder = EigenMLP(args.spe_dim, args.hidden_dim, args.hidden_dim, args.period)
    model = SpaSpeNode(spa_encoder, spe_encoder, hidden_dim=args.hidden_dim, t=args.t)
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    train_weights = model.trainable_weights
    loss_func = ContrastiveLoss(model, temp=args.t)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data_all = {
        'x': data.x,
        'edge_index': data.edge_index,
        'train_idx': train_idx,
        'valid_idx': val_idx,
        'test_idx': test_idx,
        'e': data.e,
        'u': data.u,
    }

    for epoch in range(args.n_epochs):
        model.set_train()
        loss = train_one_step(data=data_all, label=data.y)
        if (epoch + 1) % 10 == 0:
            model.set_eval()
            spa_emb = tlx.detach(model.spa_encoder(data.x, data.edge_index))
            spe_emb = tlx.detach(model.spe_encoder(e, u))
            # acc, pred = node_evaluation((spa_emb + spe_emb)/2, y, train_idx, val_idx, test_idx)
            acc, pred = node_evaluation(tlx.concat((spa_emb, spe_emb), axis=-1), data.y, train_idx, val_idx, test_idx)
            print(f'Epoch {epoch+1}/{args.n_epochs}, Accuracy: {acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='facebook')
    parser.add_argument('--dataset_path', type=str, default=r'./', help="path to save dataset")
    parser.add_argument('--spe_dim', type=int, default=100)
    parser.add_argument('--period', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--t', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    if args.gpu >=0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")
    main(args)
