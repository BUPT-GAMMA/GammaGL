import os
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
from gammagl.models.graphsage import GraphSAGE_Full_Model

import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid

from tensorlayerx.model import TrainOneStep, WithLoss

from gammagl.utils import mask_to_index


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





def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    net = GraphSAGE_Full_Model(in_feats=dataset.num_features,
                               n_hidden=args.hidden_dim,
                               n_classes=dataset.num_classes,
                               n_layers=args.n_layers,
                               activation=tlx.ReLU(),
                               dropout=args.drop_rate,
                               aggregator_type=args.aggregator_type)
    optimizer = tlx.optimizers.Adam(args.lr)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "edge_index": graph.edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
        "y": graph.y
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
        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)
    net.set_eval()

    logits = net(data['x'], data['edge_index'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--aggregator_type", type=str, default="gcn", help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()

    main(args)
