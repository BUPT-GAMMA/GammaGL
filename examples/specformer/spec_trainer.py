"""
@File    :   spec_trainer.py
@Time    :   2023/8/10
@Author  :   ZZH
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from gammagl.datasets import WikipediaNetwork
from tensorlayerx.model import TrainOneStep
import argparse
import gammagl.transforms as T
from gammagl.models.specformer import *

try:
    import cPickle as pickle
except ImportError:
    import pickle

if tlx.BACKEND == 'torch':  # when the backend is torch and you want to use GPU
    try:
        tlx.set_device(device='GPU', id=0)
    except:
        print("GPU is not available")



def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'])

        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss





def main(args):
    dataset = WikipediaNetwork(root="./", name=args.dataset, transform=T.NormalizeFeatures())
    graph = dataset[0]
    edge_index = graph.edge_index

    train_idx,val_idx,test_idx = get_split(y=graph.y , nclass = dataset.num_classes)
    net = Specformer(nclass=dataset.num_classes, nfeat=graph.num_node_features, nlayer=args.nlayer, hidden_dim=args.hidden_dim)
    optimizer = tlx.optimizers.Adam(lr=args.lr,weight_decay=0.0005)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights
    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }
    best_val_acc = 0.0
    for now_epoch in range(args.n_epoch):

        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        logits = net(data['x'], data['edge_index'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)
        #
        print("Epoch [{:0>3d}] ".format(now_epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights("./"+ net.name +".npz", format='npz_dict')

    net.load_weights("./"+ net.name +".npz", format='npz_dict')

    net.set_eval()
    logits = net(data['x'], data['edge_index'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=2000, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=32, help="dimension of hidden layers")
    parser.add_argument("--nlayer", type=int, default=2, help="number of Speclayers")
    parser.add_argument('--dataset', type=str, default='chameleon', help='dataset')

    args = parser.parse_args()
    main(args)
