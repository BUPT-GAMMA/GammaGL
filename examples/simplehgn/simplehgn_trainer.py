import argparse
import os
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
sys.path.insert(0, os.path.abspath('./')) # adds path2gammagl to execute in command line.
import tensorlayerx as tlx
from gammagl.datasets import HGBDataset
from gammagl.models import SimpleHGNModel
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils import add_self_loops, mask_to_index

def TODO():
    return 

class SemiSpvzLoss(WithLoss):
    def __init__(self, model, loss_fn):
        super(SemiSpvzLoss,self).__init__(backbone = model, loss_fn = loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data['x'])
        train_logits = tlx.gather(logits, data["train_idx"])
        train_y = tlx.gather(data["y"], data["train_idx"])
        loss = self._loss_fn(train_logits, train_y)
        return loss

def main(args):
    TODO()
    if(str.lower(args.dataset) not in ['dblp',]):
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    Unknownname = {
        'dblp':['author', 'paper', 'term', 'venue'],
    }

    dataset = HGBDataset(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph._edge_index, n_loops=1, num_nodes=graph._num_nodes)
    val_ratio = 0.2 
    train_idx = mask_to_index(graph._train_mask)
    split = int(train_idx.shape[0]*val_ratio)
    train_idx = train_idx[split: ]
    val_idx = train_idx[ :split]
    test_idx = mask_to_index(graph._test_mask)
    y = graph._y
    num_nodes = graph._num_nodes
    x = [ graph[node_type].x for node_type in Unknownname[str.lower(args.dataset)]]
    feature_dims = [graph[node_type].shape[0] for node_type in Unknownname[str.lower(args.dataset)]]
    heads_list = [args.heads] * arg.num_layers + [1]
    #TODO：数据集中添加num_etypes
    num_etypes = graph._num_etypes
    #TODO:数据集中添加num_classes
    num_classes = graph._num_classes
    activation = tlx.nn.activation.ELU()

    data = {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }


    for _ in range(args.repeat):
        model = SimpleHGNModel(feature_dims=feature_dims, 
                          hidden_dim=args.hidden_dim, 
                          edge_dim=args.edge_dim, 
                          heads_list=heads_list, 
                          num_etypes=num_etypes, 
                          num_classes=num_classes, 
                          num_layers=args.num_layers, 
                          activation=activation, 
                          feat_drop=args.drop_rate, 
                          attn_drop=args.drop_rate, 
                          negative_slope=args.slope, 
                          residual=True, 
                          beta=0.05)

        test = model()
        model(data['x'],data['edge_index'])
        return 
        loss = tlx.losses.softmax_cross_entropy_with_logits
        optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
        #metrics暂时不支持
        #metrics = tlx.metrics.
        train_weights = model.trainable_weights

        loss_func = SemiSpvzLoss(model, loss)
        train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

        best_val_acc = 0
        for epoch in range(args.n_epoch):
            TODO()
















if __name == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                         '4 - only term features (id vec for others);' + 
                         '5 - only term features (zero vec for others).')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    parser.add_argument('--n_epoch', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='Patience.')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--dataset_path", type = str, default = r"../')

    args = parser.parse_args()
    main()