# import os
# os.environ['TL_BACKEND'] = 'torch'
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss

from gammagl.datasets.flickr import Flickr
from gammagl.models.sign import SignModel

import gammagl.transforms as T


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['xs'])
        # CrossEntropyLoss equal nll_loss(log_sigmod(h), target)
        loss = self._loss_fn(logits[data['train_mask']], label[data['train_mask']])
        return loss


def evaluate(net, xs, label, mask, metrics):
    net.set_eval()
    logits = net(xs)
    metrics.update(logits[mask], label[mask])
    acc = metrics.result()
    metrics.reset()
    return acc


def main(args):
    transform = T.Compose([T.NormalizeFeatures(), T.SIGN(args.K)])
    dataset = Flickr(args.dataset_path, transform=transform)
    graph = dataset[0]
    graph = graph.tensor()  # transform may modify some data into Numpy.ndarray.

    xs = [graph.x]
    xs += [graph[f'x{i}']for i in range(1, args.K + 1)]

    net = SignModel(K=args.K, in_feat=dataset.num_node_features,
                    hid_feat=args.hidden_dim, num_classes=dataset.num_classes,
                    drop=1 - args.keep_rate)

    optimizer = tlx.optimizers.Adam(args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc = 0
    data = {"xs": xs,
            "train_mask": graph.train_mask}
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        val_acc = evaluate(net, xs, graph.y, graph.val_mask, metrics)
        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))
        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + "SIGN.npz")
    net.load_weights(args.best_model_path + "SIGN.npz")
    test_acc = evaluate(net, xs, graph.y, graph.test_mask, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--dataset_path", type=str, default=r'../Flickr', help="path to save dataset")
    parser.add_argument('--dataset', type=str, default='Flickr', help='Only flickr')
    parser.add_argument("--hidden_dim", type=int, default=1024, help="dimention of hidden layers")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--l2_coef", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--keep_rate", type=float, default=0.5, help="keep_rate = 1 - drop_rate")
    args = parser.parse_args()
    main(args)
