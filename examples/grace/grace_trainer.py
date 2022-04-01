from tqdm import tqdm
import tensorflow as tf
from gammagl.data import Graph
from gammagl.datasets import Planetoid
import tensorlayerx as tlx
import argparse
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from gammagl.models.grace import grace, calc
from tensorlayerx.model import TrainOneStep, WithLoss

from gammagl.utils.corrupt_graph import dfde_norm_g
from gammagl.utils.use_sklearn import label_classification


class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label):
        # edge_index_2, feat_2, edge_weight_2, num_node_2
        loss = self._backbone(data["edge_index_1"], data["feat_1"], data["edge_weight_1"],
                              data["edge_index_2"], data["feat_2"], data["edge_weight_2"],
                              data["num_node"])
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in
                            self._backbone.trainable_weights]) * args.l2  # only support for tensorflow backend
        return loss + l2_loss



def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    row, col, weight = calc(graph.edge_index, graph.num_nodes)
    # orgin_graph = Graph(x=graph.x, edge_index=tlx.convert_to_tensor([row, col], dtype=tlx.int64),
    #                     num_nodes=graph.num_nodes, y=graph.y)
    orgin_graph_edge_weight = tlx.convert_to_tensor(weight)
    orgin_graph_feat = graph.x
    orgin_graph_edge_index = tlx.convert_to_tensor([row, col], dtype=tlx.int64)
    x = graph.x

    # build model
    net = grace(in_feat=x.shape[1], hid_feat=args.hidden_dim,
                out_feat=args.hidden_dim,
                num_layers=args.num_layers,
                activation=tlx.nn.PRelu(args.hidden_dim) if args.dataset is "citeseer" else tlx.ReLU(),
                temp=args.temp)
    optimizer = tlx.optimizers.Adam(args.lr)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        edge_index_1, feat_1, edge_weight_1, num_node_1 = dfde_norm_g(graph.edge_index, x, args.drop_feature_rate_1,
                                                                      args.drop_edge_rate_1)
        edge_index_2, feat_2, edge_weight_2, num_node_2 = dfde_norm_g(graph.edge_index, x, args.drop_feature_rate_2,
                                                                      args.drop_edge_rate_2)
        data = {"edge_index_1": edge_index_1, "feat_1": feat_1, "edge_weight_1": edge_weight_1,
                "edge_index_2": edge_index_2, "feat_2": feat_2, "edge_weight_2": edge_weight_2,
                "num_node": num_node_1}
        loss = train_one_step(data, label=None)
        print("loss :{:4f}".format(loss))
    print("=== Final ===")
    net.set_eval()
    embeds = net.get_embeding(orgin_graph_feat, orgin_graph_edge_index, orgin_graph_edge_weight, graph.num_nodes)
    '''Evaluation Embeddings  '''
    print(label_classification(embeds, tlx.argmax(graph.y, 1), graph.train_mask, graph.test_mask, args.split))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--l2", type=float, default=0., help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset,cora/pubmed/citeseer')
    parser.add_argument('--split', type=str, default='random', help='random or public')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()
    main(args)
