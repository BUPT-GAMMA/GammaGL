import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models import DNAModel
from sklearn.model_selection import StratifiedKFold
import numpy as np


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


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

def gen_uniform_20_20_60_split(data):
    skf = StratifiedKFold(5, shuffle=True, random_state=55)
    data.y = tlx.convert_to_numpy(data.y)
    idx = [tlx.convert_to_tensor(i) for _, i in skf.split(data.y, data.y)]
    data.train_idx = tlx.convert_to_tensor(idx[0], dtype=tlx.int64)
    data.val_idx = tlx.convert_to_tensor(idx[1], dtype=tlx.int64)
    data.test_idx = tlx.convert_to_tensor(tlx.concat(idx[2:], axis=0), dtype=tlx.int64)
    data.y = tlx.convert_to_tensor(data.y)
    return data


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    graph = gen_uniform_20_20_60_split(graph)

    net = DNAModel(in_channels=dataset.num_node_features,
                   hidden_channels=args.hidden_dim,
                   out_channels=dataset.num_classes,
                   num_layers=args.num_layers,
                   drop_rate_conv=args.drop_rate_conv,
                   drop_rate_model=args.drop_rate_model,
                   heads=args.heads,
                   groups=args.groups,
                   name="DNA")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        # "edge_weight": edge_weight,
        "train_idx": graph.train_idx,
        "test_idx": graph.test_idx,
        "val_idx": graph.val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        logits = net(data['x'], data['edge_index'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path+net.name+".npz", format='npz_dict')

    net.load_weights(args.best_model_path+net.name+".npz", format='npz_dict')
    net.set_eval()
    logits = net(data['x'], data['edge_index'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimention of hidden layers")
    parser.add_argument("--drop_rate_conv", type=float, default=0.8, help="drop_rate_conv")
    parser.add_argument("--drop_rate_model", type=float, default=0.8, help="drop_rate_model")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--heads", type=int, default=8, help="number of heads for stablization")
    parser.add_argument("--groups", type=int, default=16, help="number of groups")
    parser.add_argument("--l2_coef", type=float, default=5e-5, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--gpu", type=int, default=6)
    
    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    # main(args)

    test_accs = []
    for i in range(5):
        print(f"Run {i+1}:")
        test_acc = main(args)
        test_accs.append(test_acc)

    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    print(f"Mean Test Accuracy: {mean_test_acc:.4f}")
    print(f"Standard Deviation of Test Accuracy: {std_test_acc:.4f}")
