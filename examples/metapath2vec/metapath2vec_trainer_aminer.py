import argparse
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# os.environ['TL_BACKEND'] = 'tensorflow'

import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.

import os.path as osp
import tensorlayerx as tlx
from gammagl.datasets import AMiner
from gammagl.models import MetaPath2Vec
from tensorlayerx.model import WithLoss, TrainOneStep
from sklearn.linear_model import LogisticRegression
import numpy as np

if tlx.BACKEND == 'torch':  # when the backend is torch and you want to use GPU
    try:
        tlx.set_device(device='GPU', id=4)
    except:
        print("GPU is not available")


class Unsupervised_Loss(WithLoss):
    def __init__(self, net, loss_fn):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data["pos_rw"], data["neg_rw"])
        loss = self._loss_fn(logits, label)

        return loss


def calculate_acc(train_z, train_y, test_z, test_y, solver='lbfgs', multi_class='auto', max_iter=300):
    train_z = tlx.convert_to_numpy(train_z)
    train_y = tlx.convert_to_numpy(train_y)
    test_z = tlx.convert_to_numpy(test_z)
    test_y = tlx.convert_to_numpy(test_y)

    clf = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter).fit(train_z, train_y)
    return clf.score(test_z, test_y)


def main(args, log_steps=10):
    # load datasets
    if str.lower(args.dataset) not in ['aminer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Aminer')
    dataset = AMiner(path)
    graph = dataset[0]

    if tlx.BACKEND == "paddle":
        graph['author'].y_index = tlx.convert_to_tensor(graph['author'].y_index)
    else:
        graph['author'].y_index = tlx.convert_to_tensor(tlx.convert_to_numpy(graph['author'].y_index), dtype=tlx.int64)

    metapath = [
        ('author', 'writes', 'paper'),
        ('paper', 'published_in', 'venue'),
        ('venue', 'publishes', 'paper'),
        ('paper', 'written_by', 'author'),
    ]

    model = MetaPath2Vec(graph.edge_index_dict,
                         embedding_dim=args.embedding_dim,
                         metapath=metapath,
                         walk_length=args.walk_length,
                         context_size=args.window_size,
                         walks_per_node=args.num_walks,
                         num_negative_samples=args.num_negative_samples,
                         name="MetaPath2Vec")
    loader = model.loader(batch_size=args.batch_size, shuffle=True)

    optimizer = tlx.optimizers.Adam(lr=args.lr)
    train_weights = model.trainable_weights
    loss_func = Unsupervised_Loss(net=model, loss_fn=tlx.losses.absolute_difference_error)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            data = {
                "pos_rw": pos_rw,
                "neg_rw": neg_rw,
            }
            model.set_train()
            train_loss = train_one_step(data, tlx.convert_to_tensor(0, dtype=tlx.float32))
            total_loss += train_loss.item()
            # print(i)
            if (i + 1) % log_steps == 0:
                model.set_eval()
                z = model.campute('author', batch=graph['author'].y_index)
                y = graph['author'].y
                if tlx.BACKEND == "paddle":
                    z = tlx.convert_to_tensor(z)
                    y = tlx.convert_to_tensor(y)
                perm = np.random.permutation(z.shape[0])
                # train_perm = perm[:int(z.size(0) * args.train_ratio)]
                train_perm = perm[:int(z.shape[0] * args.train_ratio)]
                # test_perm = perm[int(z.size(0) * args.train_ratio):]
                test_perm = perm[int(z.shape[0] * args.train_ratio):]
                # train_perm = tlx.convert_to_tensor(train_perm)
                # test_perm = tlx.convert_to_tensor(test_perm)
                # y = np.array(y)
                # val_acc = calculate_acc(z[train_perm], y[train_perm], z[test_perm], y[test_perm], max_iter=300)
                # val_acc = calculate_acc(tlx.gather(z, train_perm), tlx.gather(y, train_perm), tlx.gather(z, test_perm), tlx.gather(y, test_perm), max_iter=300)
                if tlx.BACKEND == "paddle":
                    val_acc = calculate_acc(tlx.gather(z, tlx.convert_to_tensor(train_perm)),
                                            tlx.gather(y, tlx.convert_to_tensor(train_perm)),
                                            tlx.gather(z, tlx.convert_to_tensor(test_perm)),
                                            tlx.gather(y, tlx.convert_to_tensor(test_perm)), max_iter=300)
                else:
                    val_acc = calculate_acc(tlx.gather(z, train_perm), tlx.gather(y, train_perm),
                                            tlx.gather(z, test_perm), tlx.gather(y, test_perm), max_iter=300)

                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                       f'Loss: {total_loss / log_steps:.4f}, '
                       f'Acc: {val_acc:.4f}'))
                total_loss = 0

                # save best model on evaluation set
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model.save_weights(
                        args.best_model_path + tlx.BACKEND + "_" + args.dataset + "_" + model.name + ".npz",
                        format='npz_dict')

    model.load_weights(args.best_model_path + tlx.BACKEND + "_" + args.dataset + "_" + model.name + ".npz",
                       format='npz_dict')
    model.set_eval()
    z = model.campute('author', batch=graph['author'].y_index)
    y = graph['author'].y
    if tlx.BACKEND == "paddle":
        z = tlx.convert_to_tensor(z)
        y = tlx.convert_to_tensor(y)
    perm = np.random.permutation(z.shape[0])
    # train_perm = perm[:int(z.size(0) * args.train_ratio)]
    train_perm = perm[:int(z.shape[0] * args.train_ratio)]
    # test_perm = perm[int(z.size(0) * args.train_ratio):]
    test_perm = perm[int(z.shape[0] * args.train_ratio):]
    # test_acc = calculate_acc(z[train_perm], y[train_perm], z[test_perm], y[test_perm], max_iter=300)
    if tlx.BACKEND == "paddle":
        test_acc = calculate_acc(tlx.gather(z, tlx.convert_to_tensor(train_perm)),
                                 tlx.gather(y, tlx.convert_to_tensor(train_perm)),
                                 tlx.gather(z, tlx.convert_to_tensor(test_perm)),
                                 tlx.gather(y, tlx.convert_to_tensor(test_perm)), max_iter=300)
    else:
        test_acc = calculate_acc(tlx.gather(z, train_perm), tlx.gather(y, train_perm),
                                 tlx.gather(z, test_perm), tlx.gather(y, test_perm), max_iter=300)

    print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aminer', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=5, help="number of epoch")
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--walk_length", type=int, default=60)
    parser.add_argument("--num_walks", type=int, default=600)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--num_negative_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    main(args)
