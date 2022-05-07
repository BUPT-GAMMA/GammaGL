import os

os.environ['TL_BACKEND'] = 'tensorflow'
# note, now can only support tensorflow, due to tlx dont support update operation
# set your backend here, default 'tensorflow', you can choose 'paddle'、'tensorflow'、'torch'

from tqdm import tqdm
from gammagl.data import Graph
from gammagl.datasets import Planetoid
import tensorlayerx as tlx
import argparse
from gammagl.models.grace import grace, calc
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils.corrupt_graph import dfde_norm_g
from eval import label_classification


class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label=None):
        loss = self._backbone(data['graph1'], data['graph2'])
        return loss


def main(args):
    # load cora dataset
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    dataset.process()
    graph = dataset[0]
    row, col, weight = calc(graph.edge_index, graph.num_nodes)
    # orign graph
    orgin_graph = Graph(x=graph.x, edge_index=tlx.convert_to_tensor([row, col], dtype=tlx.int64),
                        num_nodes=graph.num_nodes, y=graph.y)
    orgin_graph.edge_weight = tlx.convert_to_tensor(weight)

    x = graph.x

    # build model
    net = grace(in_feat=x.shape[1], hid_feat=args.hid_dim,
                out_feat=args.hid_dim,
                num_layers=args.num_layers,
                activation=tlx.nn.PRelu() if args.dataset == "citeseer" else tlx.ReLU(),
                temp=args.temp)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    best = 1e9
    cnt_wait = 0
    for epoch in tqdm(range(args.n_epoch)):
        net.set_train()
        graph1 = dfde_norm_g(graph.edge_index, x, args.drop_feature_rate_1,
                             args.drop_edge_rate_1)
        graph2 = dfde_norm_g(graph.edge_index, x, args.drop_feature_rate_2,
                             args.drop_edge_rate_2)
        data = {"graph1": graph1, "graph2": graph2}
        loss = train_one_step(data, label=None)
        if loss < best:
            best = loss
            cnt_wait = 0
            net.save_weights(args.best_model_path + "Grace.npz")
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        print("loss :{:4f}".format(loss.item()))
    print("=== Final ===")
    net.load_weights(args.best_model_path + "Grace.npz")
    net.set_eval()
    embeds = net.get_embeding(orgin_graph.x, orgin_graph.edge_index, orgin_graph.edge_weight, graph.num_nodes)

    '''Evaluation Embeddings  '''
    label_classification(embeds, tlx.argmax(graph.y, 1), graph.train_mask, graph.test_mask, args.split)



if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epoch")
    parser.add_argument("--hid_dim", type=int, default=512, help="dimention of hidden layers")
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--l2", type=float, default=1e-5, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='citeseer', help='dataset,cora/pubmed/citeseer')
    parser.add_argument('--split', type=str, default='random', help='random or public')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()
    main(args)
