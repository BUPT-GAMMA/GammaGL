import os
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorlayerx as tlx
import scipy.sparse.linalg
from evaluation_test import node_evaluation
from tensorlayerx.model import WithLoss, TrainOneStep
from gammagl.models import SpaSpeNode
from gammagl.datasets import FacebookPagePage, WikiCS, Planetoid
from gammagl.utils import get_laplacian, to_scipy_sparse_matrix, get_train_val_test_split, mask_to_index
import numpy as np


def compute_laplacian(data, args):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    index, attr = get_laplacian(edge_index, num_nodes, normalization="sym")
    L = to_scipy_sparse_matrix(index, attr)
    e, u = scipy.sparse.linalg.eigsh(L, k=args.spe_dim, which='SM', tol=1e-3)

    data.e = tlx.convert_to_tensor(e, dtype=tlx.float32)
    data.u = tlx.convert_to_tensor(u, dtype=tlx.float32)

    return data


class ContrastiveLoss(WithLoss):
    def __init__(self, model, loss_fn):
        super(ContrastiveLoss, self).__init__(backbone=model, loss_fn=loss_fn)

    def forward(self, data, label):
        h1, h2 = self.backbone_network(data['x'], data['edge_index'], data['e'], data['u'])

        logits = tlx.matmul(h1, tlx.transpose(h2, perm=(1, 0)))

        exp_logits = tlx.exp(logits)
        diag = tlx.convert_to_tensor(np.diagonal(tlx.convert_to_numpy(exp_logits)))

        sum_rows_a = tlx.reduce_sum(exp_logits, axis=1)
        sum_rows_b = tlx.reduce_sum(exp_logits, axis=0)

        log_prob1 = tlx.log(diag / (sum_rows_a - diag))
        log_prob2 = tlx.log(diag / (sum_rows_b - diag))
        loss = -0.5 * (log_prob1 + log_prob2).mean()

        return loss

def main(args):
    if args.dataset in ['pubmed',  'wikics', 'facebook']:
        if args.dataset == 'facebook':
            dataset = FacebookPagePage()
        elif args.dataset == 'wikics':
            dataset = WikiCS()
        elif args.dataset == 'pubmed':
            dataset = Planetoid(name=args.dataset)

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    data = dataset[0]
    data = compute_laplacian(data, args)
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

    model = SpaSpeNode(input_dim=data.x.shape[1],
                       spe_dim=args.spe_dim,
                       hidden_dim=args.hidden_dim,
                       output_dim=args.output_dim,
                       period=args.period,
                       name="sp2gcl")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    train_weights = model.trainable_weights
    loss_func = ContrastiveLoss(model, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data_all = {
        'x': data.x,
        'edge_index': data.edge_index,
        'train_idx': train_idx,
        'valid_idx': val_idx,
        'test_idx': test_idx,
        'e': data.e,
        'u': data.u
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data=data_all, label=data.y)
        model.set_eval()
        spa_emb = tlx.detach(model.spa_encoder(data.x, data.edge_index))
        spe_emb = tlx.detach(model.spe_encoder(e, u))
        val_acc = node_evaluation(tlx.concat((spa_emb, spe_emb), axis=-1), data.y, train_idx, val_idx)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))
        
    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    model.set_eval()
    spa_emb = tlx.detach(model.spa_encoder(data.x, data.edge_index))
    spe_emb = tlx.detach(model.spe_encoder(e, u))
    test_acc = node_evaluation(tlx.concat((spa_emb, spe_emb), axis=-1), data.y, train_idx, test_idx)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='wikics', help='dataset')
    parser.add_argument('--dataset_path', type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument('--spe_dim', type=int, default=100)
    parser.add_argument('--period', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512, help="dimention of hidden layers")
    parser.add_argument('--output_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01, help="learnin rate")
    parser.add_argument('--l2_coef', type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--n_epoch', type=int, default=3, help="number of epoch")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu >=0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
