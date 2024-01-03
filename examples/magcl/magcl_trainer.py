import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
import argparse
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.data import Graph
from gammagl.datasets import Planetoid
from gammagl.utils import add_self_loops, calc_gcn_norm
from gammagl.utils.corrupt_graph import dfde_norm_g
from gammagl.models.magcl import NewGrace
from magcl_eval import evaluate

class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=NewGrace.get_loss)

    def forward(self, data, label):
        loss = self._backbone(data['feat1'], data['edge1'], data['weight1'], data['num_node1'],
                              data['feat2'], data['edge2'], data['weight2'], data['num_node2'])
        return loss


def main(args):
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    # original graph
    original_graph = Graph(x=graph.x, edge_index=edge_index,
                           num_nodes=graph.num_nodes, y=graph.y)
    original_graph.edge_weight = tlx.convert_to_tensor(edge_weight)

    # build model
    net = NewGrace(in_feat=dataset.num_node_features,
                     hid_feat=args.hid_dim,
                     out_feat=args.hid_dim,
                     num_layers=args.num_layers,
                     activation=tlx.nn.PRelu() if args.dataset == "citeseer" else tlx.relu,
                     temp=args.temp)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in range(1, args.n_epoch + 1):
        '''train'''
        net.set_train()

        graph1 = dfde_norm_g(graph.edge_index, graph.x, args.drop_feature_rate_1,
                             args.drop_edge_rate_1)
        graph2 = dfde_norm_g(graph.edge_index, graph.x, args.drop_feature_rate_2,
                             args.drop_edge_rate_2)

        data = {"feat1": graph1.x, "edge1": graph1.edge_index,
                "weight1": graph1.edge_weight, "num_node1": graph1.num_nodes,
                "feat2": graph2.x, "edge2": graph2.edge_index,
                "weight2": graph2.edge_weight, "num_node2": graph2.num_nodes}

        loss = train_one_step(data, label=tlx.convert_to_tensor([1]))

        print(f'(T) | Epoch={epoch:03d}, loss={loss.item():.4f}')

        '''eval'''
        if epoch % 100 == 0:
            net.set_eval()

            embeds = net.get_embeding(original_graph.x, original_graph.edge_index, original_graph.edge_weight,
                                      graph.num_nodes)

            '''Evaluation Embeddings  '''
            best_acc = 0.0
            for _ in range(5):
                acc = evaluate(tlx.convert_to_numpy(embeds), graph.y, tlx.convert_to_numpy(graph.train_mask),
                               tlx.convert_to_numpy(graph.test_mask), args.split)
                if best_acc < acc:
                    best_acc = acc
            print(f'(E) | Epoch={epoch:03d}, acc={best_acc}')


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epoch")
    parser.add_argument("--hid_dim", type=int, default=128, help="dimention of hidden layers")
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.4)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.3)
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.4)
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--l2", type=float, default=0.00001, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset,cora/pubmed/citeseer')
    parser.add_argument('--split', type=str, default='random', help='random or public')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    args = parser.parse_args()
    main(args)
