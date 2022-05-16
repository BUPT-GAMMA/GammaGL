import numpy as np
import scipy.sparse as sp
import argparse
import os
os.environ['TL_BACKEND'] = 'tensorflow'# set your backend here, default `tensorflow`, you can choose 'paddle'、'tensorflow'、'torch'
import tensorlayerx as tlx
from gammagl.models.merit import MERIT,calc,gdc
from gammagl.datasets import Planetoid
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.layers.conv import GCNConv
from examples.merit.eval import evaluation
from gammagl.data import Graph
from tqdm import tqdm
from gammagl.utils.corrupt_graph import dfde_norm_g
from gammagl.datasets.amazon import Amazon
class Unsupervised_Loss(WithLoss):
    def __init__(self, net):
        super(Unsupervised_Loss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, data, label=None):
        #loss = self._backbone(data['adj_1'], data['adj_2'], data['aug_feat1'], data['aug_feat2'], data['sparse'])
        loss = self._backbone(data['graph1'], data['graph2'])
        return loss
        #return loss


def main(args):
    # load cora dataset
    if str.lower(args.data) not in ['cora', 'pubmed', 'citeseer']:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(r'../', args.data)
    dataset.process()
    #dataset=Amazon(root='./Amazon/',name='photo')
    #dataset.process()
    graph = dataset[0]
    num_node =graph.num_nodes

    edge=graph.edge_index
    row, col, weight = calc(graph.edge_index, graph.num_nodes)
    orgin_graph = Graph(x=graph.x, edge_index=tlx.convert_to_tensor([row, col], dtype=tlx.int64),
                        num_nodes=graph.num_nodes, y=graph.y)
    orgin_graph.edge_weight = tlx.convert_to_tensor(weight)

    features=tlx.convert_to_tensor(graph.x)
    #labels = tlx.convert_to_tensor(graph.y)
    labels = tlx.argmax(graph.y, axis=1)
    train_mask=tlx.convert_to_tensor(graph.train_mask)
    val_mask=tlx.convert_to_tensor(graph.val_mask)
    test_mask=tlx.convert_to_tensor(graph.test_mask)
    weight = np.ones(edge.shape[1])
    sparse_adj = sp.coo_matrix((weight, (edge[0], edge[1])), shape=(num_node, num_node))
    A = (sparse_adj + sp.eye(num_node)).tocoo()#原始图的邻接矩阵
    diff = gdc(A, alpha=args.alpha, eps=0.0001)

    norm_diff = sp.csr_matrix(diff).tocoo()
    diff_row, diff_col,diff_weight = norm_diff.row,norm_diff.col,norm_diff.data
    diff_edge_index =tlx.convert_to_tensor([diff_col,diff_row],dtype='int64')
    diff_weight=tlx.convert_to_tensor(diff_weight,dtype='float32')

    '''
    if args.sparse:
        eval_adj = tlx.convert_to_tensor(norm_adj.todense())
        eval_diff = tlx.convert_to_tensor(norm_diff.todense())
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = tlx.convert_to_tensor(eval_adj[np.newaxis])
        eval_diff = tlx.convert_to_tensor(eval_diff[np.newaxis])
    '''
    #define model
    model = GCNConv(args.input_dim, args.gnn_dim)
    net = MERIT(
        gnn=model,
        feat_size=args.input_dim,
        num_layers=args.num_layers,
        projection_size=args.proj_size,
        projection_hidden_size=args.proj_hid,
        prediction_size=args.pred_size,
        activation=tlx.nn.PRelu(args.proj_hid,a_init=tlx.initializers.constant(0.25)),
        prediction_hidden_size=args.pred_hid,
        moving_average_decay=args.momentum, beta=args.beta)
    #adj=tlx.convert_to_tensor(adj.todense())

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    train_weights = net.trainable_weights
    loss_func = Unsupervised_Loss(net)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    best = 0
    #cnt_wait = 0
    patience_count = 0
    results=[]
    result_over_runs=[]
    features = tlx.convert_to_tensor(np.array(features),dtype='float32')
    for epoch in tqdm(range(args.epochs)):
        net.set_train()
        for _ in range(args.batch_size):
            graph1 = dfde_norm_g(graph.edge_index, features, args.drop_feat_rate_1,
                                args.drop_edge_rate_1)
            graph2 = dfde_norm_g(graph.edge_index, features, args.drop_feat_rate_2,
                                args.drop_edge_rate_2)
            data = {"graph1": graph1, "graph2": graph2}
            #data = {"adj_1": adj_1, "adj_2": adj_2, "aug_feat1":aug_features1, "aug_feat2":aug_features2, "sparse":args.sparse}
            loss = train_one_step(data,label=None)
            #print("loss:",loss.item())
            net.update_ma()
            
            
            
        if epoch % args.eval_every_epoch == 0:
            net.set_eval()
            acc, micro = evaluation(orgin_graph.edge_index, orgin_graph.edge_weight, diff_edge_index, diff_weight,\
                 features, model, train_mask, test_mask, args.sparse,labels,tlx.nn.PRelu(args.gnn_dim, tlx.nn.initializers.constant(0.25)))
            if acc > best:
                best = acc
                patience_count = 0
                #net.save_weights(args.best_model_path + "Merit.npz")

            else:
                patience_count += 1
            results.append(acc)
            print('\t epoch {:03d} | loss {:.5f} | test acc {:.5f} | F1 micro {:.5f}'.format(epoch, loss.item(),acc,micro))
            if patience_count >= args.patience:
                print('Early Stopping.')
                break
            
    result_over_runs.append(max(results))
    print('\t best acc {:.5f}'.format(max(results)))
   


    
if __name__ == '__main__':
    # parameters setting
    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_layers',type=int, default=2)
    parser.add_argument('--data', type=str, default='cora',help='dataset,cora/pubmed/citeseer')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--eval_every_epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--sparse', type=str_to_bool, default=True)

    parser.add_argument('--input_dim', type=int, default=1433)#feature.shape[1]
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