# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pna_trainer.py
@Time    :   2022/5/26 20:07:55
@Author  :   huang le
"""
import os
os.environ['TL_BACKEND'] = 'tensorflow'  # set your backend here, default `tensorflow`
import os.path as osp
import argparse
import sys
sys.path.insert(0, osp.abspath('../../'))  # adds path2gammagl to execute in command line.
import numpy as np
import tensorlayerx as tlx
from tensorlayerx import convert_to_tensor, convert_to_numpy
from gammagl.datasets import ZINC
from gammagl.loader import DataLoader
from gammagl.models import PNAModel
from gammagl.utils.degree import degree
from tensorlayerx.model import TrainOneStep, WithLoss
from tensorlayerx.losses import absolute_difference_error
from tensorlayerx.optimizers.lr import ReduceOnPlateau

tlx.set_device(device='GPU', id=0)  # set your device here, default `GPU`


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        # type conversion
        x = convert_to_tensor(convert_to_numpy(data.x), dtype=tlx.int64)
        edge_attr = convert_to_tensor(convert_to_numpy(data.edge_attr), dtype=tlx.int64)
        label = convert_to_tensor(convert_to_numpy(label), dtype=tlx.float32)

        logits = self.backbone_network(x, data.edge_index, edge_attr, data.batch)
        loss = self._loss_fn(tlx.squeeze(logits, axis=1), label)
        return loss


def evaluate(model, loader):
    model.set_eval()
    for i in range(0, 4):
        model.batch_norms[i].is_train = False
    total_loss = 0
    for data in loader:
        # type conversion
        x = convert_to_tensor(convert_to_numpy(data.x), dtype=tlx.int64)
        edge_attr = convert_to_tensor(convert_to_numpy(data.edge_attr), dtype=tlx.int64)
        label = convert_to_tensor(convert_to_numpy(data.y), dtype=tlx.float32)

        out = model(x, data.edge_index, edge_attr, data.batch)
        loss = absolute_difference_error(tlx.squeeze(x=out, axis=1), label)
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)


def main(args):
    # load datasets
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ZINC')
    train_dataset = ZINC(path, subset=True, split='train')
    val_dataset = ZINC(path, subset=True, split='val')
    test_dataset = ZINC(path, subset=True, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Compute the maximum in-degree in the training data.
    # max_degree = -1
    # for data in train_dataset:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=tlx.int64)
    #     max_degree = max(max_degree, int(tlx.reduce_max(d)))

    # Compute the in-degree histogram tensor
    # deg = tlx.zeros((max_degree + 1,), dtype=tlx.int64)
    # for data in train_dataset:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=tlx.int64)
    #     deg_i = numpy.bincount(convert_to_numpy(d), minlength=len(deg))
    #     deg += convert_to_tensor(deg_i)

    deg = tlx.convert_to_tensor(np.array([0, 41130, 117278, 70152, 3104]), dtype=tlx.int64)

    model = PNAModel(in_channels=args.in_channels,
                     out_channels=args.out_channels,
                     aggregators=args.aggregators,
                     scalers=args.scalers,
                     deg=deg,
                     edge_dim=args.edge_dim,
                     towers=args.towers,
                     pre_layers=args.pre_layers,
                     post_layers=args.post_layers,
                     divide_input=args.divide_input)

    scheduler = ReduceOnPlateau(learning_rate=args.lr, mode='min', factor=0.5, patience=20,
                                min_lr=0.00001)
    optimizer = tlx.optimizers.Adam(lr=scheduler)
    train_weights = model.trainable_weights
    loss_func = SemiSpvzLoss(model, absolute_difference_error)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_test_mae = 1
    for epoch in range(1, args.n_epoch+1):
        model.set_train()
        for i in range(0, 4):
            model.batch_norms[i].is_train = True
        all_loss = 0
        for data in train_loader:
            loss = train_one_step(data, data.y)
            all_loss += loss.item() * data.num_graphs
        all_loss = all_loss / len(train_loader.dataset)
        val_mae = evaluate(model, val_loader)
        test_mae = evaluate(model, test_loader)
        scheduler.step(val_mae)

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

        print(f'Epoch: {epoch:02d}, Loss: {all_loss:.4f}, val_mae: {val_mae:.4f}, test_mae: {test_mae:.4f}')

    model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    test_mae = evaluate(model, test_loader)
    print("Test mae:  {:.4f}".format(test_mae))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=400, help="number of epoch")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument('--in_channels', type=int, default=75, help='Size of each input sample in PNAConv layer')
    parser.add_argument('--out_channels', type=int, default=75, help='Size of each output sample in PNAConv layer')
    parser.add_argument('--aggregators', type=str, default='mean min max std', help='Aggregators to use')
    parser.add_argument('--scalers', type=str, default='identity amplification attenuation', help='Scalers to use')
    parser.add_argument('--edge_dim', type=int, default=50, help='Edge feature dimensionality')
    parser.add_argument('--towers', type=int, default=5, help='Number of towers in PNA layers')
    parser.add_argument('--pre_layers', type=int, default=1, help='Number of MLP layers before aggregation')
    parser.add_argument('--post_layers', type=int, default=1, help='Number of MLP layers after aggregation')
    parser.add_argument('--divide_input', type=bool, default=False, help='Whether the input features shouldbe split '
                                                                         'between towers or not')
    args = parser.parse_args()
    main(args)
