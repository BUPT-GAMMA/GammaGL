import os
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorlayerx as tlx
import scipy.sparse.linalg
from evaluation_test import node_evaluation
from tensorlayerx.model import WithLoss, TrainOneStep
from gammagl.models import EigenMLP, SpaSpeNode, Encoder
from gammagl.datasets import FacebookPagePage, WikiCS, Planetoid
from gammagl.utils import get_laplacian, to_scipy_sparse_matrix, get_train_val_test_split, mask_to_index


def compute_laplacian(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    index, attr = get_laplacian(edge_index, num_nodes, normalization="sym")
    L = to_scipy_sparse_matrix(index, attr)
    e, u = scipy.sparse.linalg.eigsh(L, k=100, which='SM', tol=1e-3)

    data.e = tlx.convert_to_tensor(e, dtype=tlx.float32)
    data.u = tlx.convert_to_tensor(u, dtype=tlx.float32)

    return data


class ContrastiveLoss(WithLoss):
    def __init__(self, model, loss_fn):
        super(ContrastiveLoss, self).__init__(backbone=model, loss_fn=loss_fn)

    def forward(self, data, label):
        h_node_spa, h_node_spe = self.backbone_network(data['x'], data['edge_index'], data['e'], data['u'])
        h1 = tlx.l2_normalize(h_node_spa, axis=-1, eps=1e-12)
        h2 = tlx.l2_normalize(h_node_spe, axis=-1, eps=1e-12)

        logits = tlx.matmul(h1, tlx.transpose(h2, perm=(1, 0))) / data['t']
        labels = tlx.arange(start=0, limit=h1.shape[0], delta=1, dtype=tlx.int64)

        loss = 0.5 * self._loss_fn(logits, labels) + 0.5 * self._loss_fn(logits.transpose(-2, -1), labels)
        return loss

def main(args):
    if args.dataset in ['pubmed',  'wikics', 'facebook']:
        if args.dataset == 'facebook':
            dataset = FacebookPagePage()
        elif args.dataset == 'wikics':
            dataset = WikiCS()
        elif args.dataset == 'pubmed':
            dataset = Planetoid(name=args.dataset)
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

    spa_encoder = Encoder(data.x.shape[1], args.hidden_dim, args.output_dim)
    spe_encoder = EigenMLP(args.spe_dim, args.hidden_dim, args.output_dim, args.period)
    model = SpaSpeNode(spa_encoder, spe_encoder, hidden_dim=args.hidden_dim, t=args.t, name="sp2gcl")
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
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
        'u': data.u,
        't': args.t
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data=data_all, label=data.y)
        if (epoch + 1) % 10 == 0:
            model.set_eval()
            spa_emb = tlx.detach(model.spa_encoder(data.x, data.edge_index))
            spe_emb = tlx.detach(model.spe_encoder(e, u))
            # acc, pred = node_evaluation((spa_emb + spe_emb)/2, y, train_idx, val_idx, test_idx)
            val_acc = node_evaluation(tlx.concat((spa_emb, spe_emb), axis=-1), data.y, train_idx, val_idx)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

            print(f'Epoch {epoch+1}/{args.n_epoch}, Accuracy: {val_acc}')

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item()))
        
    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    model.set_eval()
    spa_emb = tlx.detach(model.spa_encoder(data.x, data.edge_index))
    spe_emb = tlx.detach(model.spe_encoder(e, u))
    # acc, pred = node_evaluation((spa_emb + spe_emb)/2, y, train_idx, val_idx, test_idx)
    test_acc = node_evaluation(tlx.concat((spa_emb, spe_emb), axis=-1), data.y, train_idx, test_idx)
    print("Test acc:  {:.4f}".format(test_acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pubmed')
    parser.add_argument('--dataset_path', type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument('--spe_dim', type=int, default=100)
    parser.add_argument('--period', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=512)
    parser.add_argument('--t', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu >=0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")
    main(args)
