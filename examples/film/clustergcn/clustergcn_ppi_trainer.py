# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   clustergcn_ppi_trainer.py
@Time    :   2022/12/10 22:05:55
@Author  :   chen weijie
"""

import os
#os.environ['TL_BACKEND'] = 'torch'
os.environ['CUDA_VISIBLE_DEVICES']='4'

from sklearn.metrics import f1_score
from gammagl.loader import ClusterData, ClusterLoader, DataLoader
from gammagl.models import ClusterGCNModel_ppi as ClusterGCNModel
import tensorlayerx as tlx
from tensorlayerx.model import WithLoss, TrainOneStep
from gammagl.datasets import PPI
from gammagl.data import BatchGraph
import argparse

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        if tlx.BACKEND =='torch':
            logits = self.backbone_network(data['x'].to('cuda'), data['edge_index'].to('cuda'))
            loss = self._loss_fn(logits, tlx.convert_to_tensor(y, dtype=float).to('cuda'))
        else:
            logits = self.backbone_network(data['x'], data['edge_index'])
            loss = self._loss_fn(logits, tlx.convert_to_tensor(tlx.convert_to_numpy(y), dtype=float))
        return loss

def calculate_f1_score(ys, preds):
    """
    Args:
        ys: labels
        preds: predictions

    Returns:
        f1 score
    """

    y, pred = tlx.convert_to_numpy(tlx.concat(ys, 0)), tlx.convert_to_numpy(tlx.concat(preds, 0))

    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

def main(args):
    path = args.dataset_path
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

    train_data = BatchGraph.from_data_list(train_dataset)
    cluster_data = ClusterData(train_data, num_parts=args.num_parts, recursive=False,
                               save_dir=train_dataset.processed_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
                                 num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = 'cuda'
    if tlx.BACKEND == 'torch':
        net = ClusterGCNModel(in_channels=train_dataset.num_features, hidden_channels=args.hidden_channels,
                              out_channels=train_dataset.num_classes, num_layers=args.num_layers).to(device)
    else:
        net = ClusterGCNModel(in_channels=train_dataset.num_features, hidden_channels=args.hidden_channels,
                out_channels=train_dataset.num_classes, num_layers=args.num_layers)

    optimizer = tlx.optimizers.Adam(args.lr)

    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.sigmoid_cross_entropy)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in range(1, args.n_epoch+1):
        total_loss = total_nodes = 0
        for batch in train_loader:
            net.set_train()
            data = {"x": batch.x,
                     "edge_index": tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(batch.edge_index),
                                                            dtype=tlx.int64),
                    "y": tlx.convert_to_tensor(batch.y.numpy(), dtype=tlx.float32)
                    }

            train_loss = train_one_step(data, data['y'])

            nodes = len(batch.x)
            total_loss += train_loss.item() * nodes
            total_nodes += nodes

        loss = total_loss / total_nodes

        print('do testing')
        val_f1_score = measure(net,val_loader,device)
        test_f1_score = measure(net,test_loader,device)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_f1_score:.4f}, '
              f'Test: {test_f1_score:.4f}')


if tlx.BACKEND == 'torch':
    import torch
    @torch.no_grad()
    def measure(net,loader,device):
        net.set_eval()
        ys, preds = [], []
        for data in loader:
            ys.append(tlx.convert_to_tensor(data.y.numpy(), dtype=tlx.float32))

            out = net(data.x.to(device), data.edge_index.to(device))
            preds.append((out > 0).float().cpu())

        val_f1_score = calculate_f1_score(ys, preds)
        return val_f1_score
else:
    def measure(net,loader,device):
        ys, preds = [], []
        for data in loader:
            ys.append(tlx.convert_to_tensor(data.y.numpy(), dtype=tlx.float32))

            out = net(data.x, data.edge_index)
            preds.append(tlx.convert_to_tensor((out > 0).numpy(), dtype=float))
        val_f1_score = calculate_f1_score(ys, preds)
        return val_f1_score

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--num_parts", type=int, default=50, help="number of partition parts")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_channels", type=int, default=1024, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.2, help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--dataset_path", type=str, default=r'./ppi', help="path to save dataset")
    args = parser.parse_args()

    main(args)
