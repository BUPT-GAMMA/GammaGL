import os
import sys
import time
import argparse
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['TL_BACKEND'] = 'paddle'

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import tensorlayerx as tlx
tlx.set_device('GPU', 0)
from gammagl.layers.conv import GCNConv
from gammagl.datasets import Planetoid
from gammagl.models import GEstimationN
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
from sklearn.metrics.pairwise import cosine_similarity as cos


class GCN(tlx.nn.Module):
    def __init__(self, num_feature, num_class, hidden_size, dropout=0.5, activation="relu"):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_class)

        self.dropout = tlx.layers.Dropout(dropout)
        # assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = tlx.ReLU()

    def forward(self, feature, adj):
        x1 = self.activation(self.conv1(feature, adj))
        x1 = self.dropout(x1)
        x2 = self.conv2(x1, adj)
        return x1, tlx.nn.LogSoftmax(dim=1)(x2)


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        _, logits = self.backbone_network(data['x'], data['edge_index'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def prob_to_adj(mx, threshold):
    mx = np.triu(mx, 1)
    mx += mx.T
    edge_index = np.where(mx > threshold)
    return tlx.convert_to_tensor(edge_index)


def knn(num_node, feature, k):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(feature)
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(num_node).repeat(k + 1), col] = 1
    return adj


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def main(args):
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    estimator = GEstimationN(dataset)
    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }
    net = GCN(num_feature=dataset.num_node_features,
              num_class=dataset.num_classes,
              hidden_size=args.hidden)
    train_weights = net.trainable_weights
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    metrics = tlx.metrics.Accuracy()
    # Train Model
    t_total = time.time()
    global_best_acc_val = 0
    global_hidden = None
    global_output = None
    for iter in range(args.iter):
        start = time.time()
        best_val_acc = 0
        for epoch in range(args.epoch):
            net.set_train()
            train_loss = train_one_step(data, graph.y)

            net.set_eval()
            hidden_output, logits = net(data['x'], data['edge_index'])
            val_logits = tlx.gather(logits, data['val_idx'])
            val_y = tlx.gather(data['y'], data['val_idx'])
            val_acc = calculate_acc(val_logits, val_y, metrics)

            print("Epoch [{:0>3d}] ".format(epoch + 1) \
                  + "  train loss: {:.4f}".format(train_loss.item()) \
                  + "  val acc: {:.4f}".format(val_acc))

            # save best model on evaluation set
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if val_acc > global_best_acc_val:
                    global_best_acc_val = val_acc
                    global_hidden = hidden_output
                    global_output = logits
                    best_edge_index = data['edge_index']
                    net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

        estimator.reset_obs()
        estimator.update_obs(knn(graph.num_nodes, tlx.convert_to_numpy(graph.x), args.k))
        estimator.update_obs(knn(graph.num_nodes, tlx.convert_to_numpy(global_hidden), args.k))
        estimator.update_obs(knn(graph.num_nodes, tlx.convert_to_numpy(global_output), args.k))

        # self.iter += 1
        alpha, beta, O, Q, iterations = estimator.EM(tlx.argmax(global_output, 1), args.tolerance)
        print(iterations)
        data['edge_index'] = prob_to_adj(Q, args.threshold)


    print("***********************************************************************************************")
    print("Optimization Finished!")
    print("Total time:{:.4f}s".format(time.time() - t_total),
          "Best validation accuracy:{:.4f}".format(global_best_acc_val),
          "EM iterations:{:04d}\n".format(iterations))

    """Evaluate the performance on testset.
    """
    print("=== Testing ===")
    print("Picking the best model according to validation performance")
    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.set_eval()
    hidden_output, logits = net(data['x'], best_edge_index)

    val_logits = tlx.gather(logits, data['val_idx'])
    val_y = tlx.gather(data['y'], data['val_idx'])
    val_acc = calculate_acc(val_logits, val_y, metrics)

    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    loss_test = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_y)
    print("Val acc:  {:.4f}".format(val_acc),
          "Test loss:  {:.4f}".format(loss_test),
          "Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='gcn', choices=['gcn', 'sgc', 'gat', 'appnp', 'sage'])
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=16, help='hidden size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'elu'])
    parser.add_argument('--dataset', type=str, default='citeseer',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to"
                                                                         " save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train the base model')
    parser.add_argument('--iter', type=int, default=30, help='number of iterations to train the GEN')
    parser.add_argument('--k', type=int, default=9, help='k of knn graph')
    parser.add_argument('--threshold', type=float, default=.5, help='threshold for adjacency matrix')
    parser.add_argument('--tolerance', type=float, default=.01, help='tolerance to stop EM algorithm')
    args = parser.parse_args()

    main(args)
