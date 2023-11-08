import argparse
import os.path as osp
import os
os.environ['TL_BACKEND'] = 'torch'
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import numpy as np
import pickle
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
from tensorlayerx.dataflow import random_split

from gammagl.utils import set_device
from gammagl.layers.conv import GCNConv
from gammagl.datasets import Planetoid, Coauthor, Amazon
import gammagl.transforms as T

from gammagl.models.grace_pot import Encoder, Model
from eval_gracepot import log_regression, MulticlassEvaluator

A_upper_1 = None
A_upper_2 = None
A_lower_1 = None
A_lower_2 = None

class train_loss(WithLoss):
    def __init__(self, model, drop_edge_rate_1, drop_edge_rate_2, use_pot=False, pot_batch=-1, kappa=0.5):
        super(train_loss, self).__init__(backbone=model, loss_fn=None)
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.use_pot = use_pot
        self.pot_batch = pot_batch
        self.kappa = kappa

    def forward(self, model, x, edge_index, epoch, data=None):
        edge_index_1 = dropout_adj(edge_index, p=self.drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(edge_index, p=self.drop_edge_rate_2)[0]
        x_1, x_2 = x, x
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        node_list = np.arange(z1.shape[0])
        np.random.shuffle(node_list)
        
        batch_size = 4096 if args.dataset in ["PubMed", "Computers", "WikiCS"] else None

        if batch_size is not None:
            node_list_batch = get_batch(node_list, batch_size, epoch)

        # nce loss
        if batch_size is not None:
            z11 = z1[node_list_batch]
            z22 = z2[node_list_batch]
            nce_loss = model.loss(z11, z22)
        else:
            nce_loss = model.loss(z1, z2)

        # pot loss
        if self.use_pot:
            # get node_list_tmp, the nodes to calculate pot_loss
            if self.pot_batch != -1:
                if batch_size is None:
                    node_list_tmp = get_batch(node_list, self.pot_batch, epoch)
                else:
                    node_list_tmp = get_batch(node_list_batch, self.pot_batch, epoch)
            else:
                # full pot batch
                if batch_size is None:
                    node_list_tmp = node_list
                else:
                    node_list_tmp = node_list_batch
                    
            z11 = tlx.gather(z1, tlx.convert_to_tensor(node_list_tmp))
            z22 = tlx.gather(z2, tlx.convert_to_tensor(node_list_tmp))
            # z11 = z1[tlx.convert_to_tensor(node_list_tmp)]
            # z22 = z2[tlx.convert_to_tensor(node_list_tmp)]

            global A_upper_1, A_upper_2, A_lower_1, A_lower_2
            if A_upper_1 is None or A_upper_2 is None:
                A_upper_1, A_lower_1 = get_A_bounds(args.dataset, self.drop_edge_rate_1)
                A_upper_2, A_lower_2 = get_A_bounds(args.dataset, self.drop_edge_rate_2)
            ###x index???
            pot_loss_1 = model.pot_loss(z11, z22, data.x, data.edge_index, edge_index_1, local_changes=self.drop_edge_rate_1, 
                                          node_list=node_list_tmp, A_upper=A_upper_1, A_lower=A_lower_1)
            pot_loss_2 = model.pot_loss(z22, z11, data.x, data.edge_index, edge_index_2, local_changes=self.drop_edge_rate_2, 
                                          node_list=node_list_tmp, A_upper=A_upper_2, A_lower=A_lower_2)
            pot_loss = (pot_loss_1 + pot_loss_2) / 2
            loss = (1 - self.kappa) * nce_loss + self.kappa * pot_loss
        else:
            loss = nce_loss

        return loss


def test(model, data, dataset, split):
    model.set_eval()
    z = model(data.x, data.edge_index)
    if args.dataset == 'ogbn-arxiv':
        y_pred = z.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': data.y[split['train']],
            'y_pred': y_pred[split['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split['valid']],
            'y_pred': y_pred[split['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split['test']],
            'y_pred': y_pred[split['test']],
        })['acc']
        return {
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "test_acc": test_acc
        }
    else:
        evaluator = MulticlassEvaluator()
        res = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=split)
    return res

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Computers', 'Photo', 'ogbn-arxiv', 'ogbg-code', 'BlogCatalog', 'Flickr', 'sbm']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'Computers':
        return Amazon(root=root_path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Photo':
        return Amazon(root=root_path, name='photo', transform=T.NormalizeFeatures())
    # if name in ['BlogCatalog']:
    #     return AttributedGraphDataset(root=root_path, name=name, transform=T.NormalizeFeatures())
    # if name in ['Flickr']:
    #     return AttributedGraphDataset(root=root_path, name=name, transform=NormalizeFeaturesSparse())
    # if name.startswith('ogbn'):
    #     return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures()) # public split

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(tlx.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = tlx.zeros((num_samples,)).to(tlx.bool)
    test_mask = tlx.zeros((num_samples,)).to(tlx.bool)
    val_mask = tlx.zeros((num_samples,)).to(tlx.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

def get_batch(node_list, batch_size, epoch):
    num_nodes = len(node_list)
    num_batches = (num_nodes - 1) // batch_size + 1
    i = epoch % num_batches
    if (i + 1) * batch_size >= len(node_list):
        node_list_batch = node_list[i * batch_size:]
    else:
        node_list_batch = node_list[i * batch_size:(i + 1) * batch_size]
    return node_list_batch
def get_A_bounds(dataset, drop_rate):
    upper_lower_file = osp.join(osp.expanduser('~/datasets'),f"bounds/{dataset}_{drop_rate}_upper_lower.pkl")
    if osp.exists(upper_lower_file):
        with open(upper_lower_file, 'rb') as file:       
            A_upper, A_lower=pickle.load(file)
    else:
        A_upper, A_lower = None, None
    return A_upper, A_lower

from typing import Optional, Tuple

def filter_adj(row, col, edge_attr,
               mask):
    mask = tlx.convert_to_tensor(mask, dtype=tlx.bool)
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]

def dropout_adj(
    edge_index,
    edge_attr= None,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[int] = None,
    training: bool = True,
):

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        return edge_index, edge_attr

    # row, col = edge_index
    row = edge_index[0]
    col = edge_index[1]

    mask = np.random.random(tlx.get_tensor_shape(row)) >= p

    if force_undirected:
        mask[row > col] = False

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = tlx.stack(
            [tlx.concat([row, col], 0),
             tlx.concat([col, row], 0)], dim=0)
        if edge_attr is not None:
            edge_attr = tlx.concat([edge_attr, edge_attr], 0)
    else:
        edge_index = tlx.stack([row, col])

    return edge_index, edge_attr


def main(args):
    if args.gpu_id >= 0:
        tlx.set_device(device='GPU', id=args.gpu_id)
    else:
        tlx.set_device(device='CPU')

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    # for hyperparameter tuning
    if args.drop_1 != -1:
        config['drop_edge_rate_1'] = args.drop_1
    if args.drop_2 != -1:
        config['drop_edge_rate_2'] = args.drop_2
    if args.tau != -1:
        config['tau'] = args.tau
    if args.num_epochs != -1:
        config['num_epochs'] = args.num_epochs
    print(args)
    print(config)

    seed_everything(args.seed)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': tlx.nn.ReLU, 'prelu': tlx.nn.PRelu()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    use_pot = args.use_pot
    kappa = args.kappa
    pot_batch = args.pot_batch

    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    
    # generate split
    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        split = data.train_mask, data.val_mask, data.test_mask
        print("Public Split")
    else:
        split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
        print("Random Split")

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers)
    model = Model(encoder, num_hidden, num_proj_hidden, tau, dataset=args.dataset)
    train_weights = model.trainable_weights
    #Adam少参数
    optimizer = tlx.optimizers.Adam(lr=learning_rate, weight_decay=weight_decay)
    loss_func = train_loss(model, drop_edge_rate_1, drop_edge_rate_2, use_pot, pot_batch, kappa)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    #timing        
    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        model.set_train()
        loss=train_one_step(model, data.x, data.edge_index, epoch ,data)
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        if epoch % 100 == 0:
            res = test(model, data, dataset, split)
            print(res)
        prev = now

    print("=== Final ===")
    res = test(model, data, dataset, split)
    print(res)

    # res_file = f"res/{args.dataset}_pot_temp.csv" if use_pot else f"res/{args.dataset}_base_temp.csv"
    # if args.save_file == '.':
    #     f = open(res_file,"a+")
    # else:
    #     f = open(args.save_file, "a+")
    # res_str = f'{res["F1Mi"]:.4f}, {res["F1Ma"]:.4f}' 
    # if use_pot:
    #     f.write(f'{config["drop_edge_rate_1"]}, {config["drop_edge_rate_2"]}, {config["tau"]}, {kappa}, '
    #             f'{res_str}\n')
    # else:
    #     f.write(f'{config["drop_edge_rate_1"]}, {config["drop_edge_rate_2"]}, {config["tau"]}, '
    #             f'{res_str}\n')
    # f.close()


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=5)
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--use_pot', default=False, action="store_true") # whether to use pot in loss
    parser.add_argument('--kappa', type=float, default=0.5)
    parser.add_argument('--pot_batch', type=int, default=-1)
    parser.add_argument('--drop_1', type=float, default=0.4)
    parser.add_argument('--drop_2', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.9) # temperature of nce loss
    parser.add_argument('--num_epochs',type=int,default=-1)
    parser.add_argument('--save_file', type=str, default=".")
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    main(args)

    