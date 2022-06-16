import argparse
import os
import sys
os.environ['TL_BACKEND'] = 'tensorflow'
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
sys.path.insert(0, os.path.abspath('./')) # adds path2gammagl to execute in command line.
import tensorlayerx as tlx
from sklearn.metrics import f1_score
from gammagl.datasets import HGBDataset
from gammagl.models import SimpleHGNModel
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils import add_self_loops, mask_to_index


def calculate_f1_score(val_logits, val_y):
    val_logits = tlx.ops.argmax(val_logits, axis=-1)
    return f1_score(val_y, val_logits, average='micro'), f1_score(val_y, val_logits, average='macro')


class SemiSpvzLoss(WithLoss):
    def __init__(self, model, loss_fn):
        super(SemiSpvzLoss,self).__init__(backbone=model, loss_fn = loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data['x'], data['edge_index'], data['e_feat'])
        train_logits = tlx.gather(logits, data["train_idx"])
        train_y = tlx.gather(data["y"], data["train_idx"])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def main(args):
    if(str.lower(args.dataset) not in ['dblp',]):
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    targetType = {
        'dblp': 'author',
    }

    dataset = HGBDataset(args.dataset_path, args.dataset)
    heterograph = dataset[0]
    homograph = heterograph.to_homogeneous()

    edge2feat = {}
    edge_index_numpy = tlx.ops.convert_to_numpy(homograph.edge_index)
    for i in range(edge_index_numpy.shape[-1]):
        edge2feat[(edge_index_numpy[0, i], edge_index_numpy[1, i])] = homograph.edge_type[i]

    edge_index, _ = add_self_loops(homograph.edge_index, n_loops=1, num_nodes=homograph.num_nodes)
    y = heterograph[targetType[str.lower(args.dataset)]].y
    num_nodes = heterograph.num_nodes
    x = [heterograph[node_type].x for node_type in heterograph.node_types ]
    feature_dims = [heterograph.num_node_features[node_type] for node_type in heterograph.node_types]
    heads_list = [args.heads] * args.num_layers + [1]
    num_etypes = tlx.ops.argmax(homograph.edge_type) + 1
    num_classes = tlx.ops.argmax(y) + 1

    e_feat = []
    edge_index_numpy = tlx.ops.convert_to_numpy(edge_index)
    for i in range(edge_index_numpy.shape[-1]):
        if edge_index_numpy[0, i] == edge_index_numpy[1, i]:
            e_feat.append(num_etypes)
        else:
            e_feat.append(edge2feat[(edge_index_numpy[0,i], edge_index_numpy[1,i])])
    e_feat = tlx.ops.convert_to_tensor(e_feat)

    activation = tlx.nn.activation.ELU()

    val_ratio = 0.2
    train_idx = mask_to_index(heterograph[targetType[str.lower(args.dataset)]].train_mask)
    split = int(train_idx.shape[0]*val_ratio)
    train_idx = train_idx[split:]
    val_idx = train_idx[ :split]
    test_idx = mask_to_index(heterograph[targetType[str.lower(args.dataset)]].test_mask)

    data = {
        'x': x,
        'e_feat': e_feat,
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
                          num_etypes=num_etypes + 1,
                          num_classes=num_classes,
                          num_layers=args.num_layers,
                          activation=activation,
                          feat_drop=args.drop_rate,
                          attn_drop=args.drop_rate,
                          negative_slope=args.slope,
                          residual=True,
                          beta=0.05)

        loss = tlx.losses.softmax_cross_entropy_with_logits
        optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)

        train_weights = model.trainable_weights
        loss_func = SemiSpvzLoss(model, loss)

        train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

        best_val_loss = float('inf')
        early_stop_count = 0
        for epoch in range(args.n_epoch):
            model.set_train()
            train_loss = train_one_step(data, y)
            model.set_eval()
            logits = model(data['x'], data['edge_index'], data['e_feat'])
            val_logits = tlx.gather(logits, data['val_idx'])
            val_y = tlx.gather(data['y'], data['val_idx'])
            val_loss = loss(val_logits, val_y)
            val_micro_f1, val_macro_f1 = calculate_f1_score(val_logits, val_y)
            print("Epoch [{:0>3d}]  ".format(epoch + 1),
               "   train loss: {:.4f}".format(train_loss.item()),
               "   val loss: {:.4f}".format(val_loss.item()),
               "   val micro: {:.4f}".format(val_micro_f1),
               "   val macro: {:.4f}".format(val_macro_f1),)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
                model.save_weights(args.best_model_path+model.name+'.npz', format='npz_dict')
            else:
                early_stop_count += 1
            if early_stop_count >= args.patience:
                break

        model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
        if tlx.BACKEND == 'torch':
            model.to(data["x"][0].device)
        model.set_eval()
        logits = model(data['x'], data['edge_index'], data['e_feat'])
        test_logits = tlx.gather(logits, data['test_idx'])
        test_y = tlx.gather(data['y'], data['test_idx'])
        test_micro_f1, test_macro_f1 = calculate_f1_score(test_logits, test_y)
        print("Test micro:  {:.4f}, Test macro: {:.4f}".format(test_micro_f1, test_macro_f1))


if __name__ == '__main__':
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
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--dataset_path', type = str, default = r"../")
    parser.add_argument("--best_model_path", type = str, default = r"./")

    args = parser.parse_args()
    main(args)
