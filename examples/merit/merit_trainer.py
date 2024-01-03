from cProfile import label
import numpy as np
import scipy.sparse as sp
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
import sys
sys.path.append(os.getcwd())
import tensorlayerx as tlx
from gammagl.models.merit import MERIT
from process import compute_diff
from gammagl.datasets import Planetoid
from tensorlayerx.model import TrainOneStep, WithLoss
from eval import evaluation
from tqdm import tqdm
from gammagl.utils.corrupt_graph import dfde_norm_g
from gammagl.utils.norm import calc_gcn_norm

class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        loss = self._backbone(data['feat1'], data['edge1'], data['weight1'], data['num_node1'],
                              data['feat2'], data['edge2'], data['weight2'], data['num_node2'])
        return loss


def main(args):
    # load dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    # dataset=Amazon(root='./Amazon/',name='photo')
    graph = dataset[0]
    num_node = graph.num_nodes
    edge_index = graph.edge_index
    weight = tlx.ops.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))
    graph.edge_weight = weight
    row, col = edge_index[0], edge_index[1]
    # origin_graph = Graph(x=graph.x, edge_index=tlx.convert_to_tensor([row, col], dtype=tlx.int64),
    #                    num_nodes=graph.num_nodes, y=graph.y)
    # origin_graph.edge_weight = tlx.convert_to_tensor(weight)
    features = graph.x
    labels = graph.y
    train_mask = tlx.convert_to_tensor(graph.train_mask)
    val_mask = tlx.convert_to_tensor(graph.val_mask)
    test_mask = tlx.convert_to_tensor(graph.test_mask)

    weight = np.ones(edge_index.shape[1])
    sparse_adj = sp.coo_matrix((weight, (edge_index[0], edge_index[1])), shape=(num_node, num_node))
    diff_edge_index, diff_weight = compute_diff(sparse_adj, alpha=args.alpha, eps=0.0001)
    # define model
    net = MERIT(
        feat_size=dataset.num_node_features,
        projection_size=args.proj_size,
        projection_hidden_size=args.proj_hid,
        prediction_size=args.pred_size,
        prediction_hidden_size=args.pred_hid,
        moving_average_decay=args.momentum, beta=args.beta)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    train_weights = net.online_encoder.trainable_weights + net.online_predictor.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    best = 0
    # cnt_wait = 0
    patience_count = 0
    results = []
    result_over_runs = []
    features = tlx.convert_to_tensor(np.array(features), dtype='float32')
    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        for _ in range(args.batch_size):
            graph1 = dfde_norm_g(graph.edge_index, features, args.drop_feat_rate_1,
                                 args.drop_edge_rate_1)
            graph2 = dfde_norm_g(graph.edge_index, features, args.drop_feat_rate_2,
                                 args.drop_edge_rate_2)
            data = {"feat1": graph1.x, "edge1": graph1.edge_index, "weight1": graph1.edge_weight,
                    "num_node1": graph1.num_nodes, \
                    "feat2": graph2.x, "edge2": graph2.edge_index, "weight2": graph2.edge_weight,
                    "num_node2": graph2.num_nodes,
                    }
            loss = train_one_step(data, label=tlx.convert_to_tensor([0]))
            net.update_ma()

        if epoch % args.eval_every_epoch == 0:
            net.set_eval()
            acc = evaluation(graph.edge_index, graph.edge_weight, diff_edge_index, diff_weight, \
                             features, net.online_encoder.gnn, train_mask, test_mask, labels, tlx.nn.PRelu(args.gnn_dim))
            if acc > best:
                best = acc
                patience_count = 0
            else:
                patience_count += 1
            results.append(acc)
            print('\t epoch {:03d} | loss {:.4f} | acc {:.4f}'.format(epoch, loss.item(), acc))
            if patience_count >= args.patience:
                print('Early Stopping.')
                break

    result_over_runs.append(max(results))
    print('\t best acc {:.5f}'.format(max(results)))


if __name__ == '__main__':
    # parameters setting

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='cora', help='dataset,cora/pubmed/citeseer')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--eval_every_epoch', type=int, default=1)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--gnn_dim', type=int, default=512)
    parser.add_argument('--proj_size', type=int, default=512)
    parser.add_argument('--proj_hid', type=int, default=4096)
    parser.add_argument('--pred_size', type=int, default=512)
    parser.add_argument('--pred_hid', type=int, default=4096)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.2)
    parser.add_argument('--drop_feat_rate_1', type=float, default=0.5)
    parser.add_argument('--drop_feat_rate_2', type=float, default=0.5)
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()
    main(args)
