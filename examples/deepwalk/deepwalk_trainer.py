import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'tensorflow'

import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import DeepWalkModel
from gammagl.utils import calc_gcn_norm, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
from sklearn.linear_model import LogisticRegression


class Unsupervised_Loss(WithLoss):
    def __init__(self, net, loss_fn):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data["edge_index"])
        loss = self._loss_fn(logits, label)

        return loss


def calculate_acc(train_z, train_y, test_z, test_y, solver='lbfgs', multi_class='auto', max_iter=150):
    train_z = tlx.convert_to_numpy(train_z)
    train_y = tlx.convert_to_numpy(train_y)
    test_z = tlx.convert_to_numpy(test_z)
    test_y = tlx.convert_to_numpy(test_y)

    clf = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter).fit(train_z, train_y)
    return clf.score(test_z, test_y)


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index = graph.edge_index
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes, edge_weight=graph.edge_weight))

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    model = DeepWalkModel(edge_index=edge_index,
                          edge_weight=edge_weight,
                          embedding_dim=args.embedding_dim,
                          walk_length=args.walk_length,
                          num_walks=args.num_walks,
                          window_size=args.window_size,
                          name="DeepWalk")

    optimizer = tlx.optimizers.Adam(lr=args.lr)
    train_weights = model.trainable_weights

    loss_func = Unsupervised_Loss(net=model, loss_fn=tlx.losses.absolute_difference_error)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data, tlx.convert_to_tensor(0, dtype=tlx.float32))
        model.set_eval()
        z = model.campute()

        val_acc = calculate_acc(tlx.gather(z[0], data['train_idx']), tlx.gather(graph.y, data['train_idx']),
                                tlx.gather(z[0], data['val_idx']), tlx.gather(graph.y, data['val_idx']),
                                max_iter=150)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

    model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        model.to(data['x'].device)
    model.set_eval()
    z = model.campute()
    test_acc = calculate_acc(tlx.gather(z[0], data['train_idx']), tlx.gather(graph.y, data['train_idx']),
                             tlx.gather(z[0], data['test_idx']), tlx.gather(graph.y, data['test_idx']),
                             max_iter=150)
    print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=10)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=5)

    args = parser.parse_args()

    main(args)
