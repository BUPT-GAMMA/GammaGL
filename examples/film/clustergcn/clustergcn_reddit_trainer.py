# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   clustergcn_reddit_trainer.py
@Time    :   2022/12/10 22:05:55
@Author  :   chen weijie
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#os.environ['TL_BACKEND'] = 'torch'

from gammagl.datasets import Reddit
from gammagl.loader import ClusterData, ClusterLoader, Neighbor_Sampler
from gammagl.models import ClusterGCNModel_reddit as ClusterGCNModel
import tensorlayerx as tlx
from tensorlayerx.model import WithLoss, TrainOneStep
import numpy as np
import argparse

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'])
        loss = self._loss_fn(logits[data['mask']], y[data['mask']])
        return loss

def main(args):
    path = args.dataset_path
    dataset = Reddit(path)
    graph = dataset[0]
    if tlx.BACKEND == 'torch':
        graph.edge_index = graph.edge_index[[1, 0], :]
    else:
        graph.edge_index = tlx.convert_to_tensor(tlx.convert_to_numpy(graph.edge_index)[[1, 0], :])

    cluster_data = ClusterData(graph, num_parts=args.num_parts, recursive=False,
                               save_dir=dataset.processed_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True,
                                 num_workers=0)
    subgraph_loader = Neighbor_Sampler(edge_index=tlx.convert_to_numpy(graph.edge_index),
                                       dst_nodes=np.arange(graph.num_nodes),
                                       sample_lists=[-1], batch_size=1024, shuffle=False, num_workers=0)

    x = tlx.convert_to_tensor(graph.x)
    device = 'cuda'
    if tlx.BACKEND == 'torch':
        net = ClusterGCNModel(dataset.num_features, args.hidden_dim,dataset.num_classes).to(device)
    else:
        net = ClusterGCNModel(dataset.num_features,args.hidden_dim, dataset.num_classes)

    optimizer = tlx.optimizers.Adam(args.lr)
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in range(1, args.n_epoch+1):

        total_loss = total_nodes = 0
        for batch in train_loader:
            net.set_train()
            if tlx.BACKEND == 'torch':
                data = {"x": batch.x.to(device),
                        "edge_index": tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(batch.edge_index),
                                                                dtype=tlx.int64).to(device),
                        "mask": batch.train_mask.to(device),
                        "y": batch.y.to(device)
                        }
            else:
                data = {"x": batch.x,
                        "edge_index": tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(batch.edge_index),
                                                                dtype=tlx.int64),
                        "mask": batch.train_mask,
                        "y": batch.y
                        }
            train_loss = train_one_step(data, data['y'])

            print("Epoch [{:0>3d}] ".format(epoch) + "  train loss: {:.4f}".format(train_loss.item()))

            nodes = tlx.convert_to_numpy(batch.train_mask).sum().item()
            total_loss += train_loss.item() * nodes
            total_nodes += nodes

        loss = total_loss / total_nodes
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if epoch == 1:
            train_acc,val_acc,test_acc = measure(net,x,subgraph_loader,graph,device)
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, test: {test_acc:.4f}')

if tlx.BACKEND == 'torch':
    import torch
    @torch.no_grad()
    def measure(net,x,subgraph_loader,graph,device):
        net.set_eval()
        logits = net.inference(x, subgraph_loader,device)
        y_pred = tlx.argmax(logits, -1)
        accs = []
        for mask in [graph.train_mask, graph.val_mask, graph.test_mask]:
            correct = tlx.convert_to_numpy(graph.y[mask] == y_pred[mask]).sum()
            accs.append(correct / tlx.convert_to_numpy(mask).sum().item())
        train_acc = accs[0]
        val_acc = accs[1]
        test_acc = accs[2]
        return train_acc,val_acc,test_acc
else:
    def measure(net,x,subgraph_loader,graph,device):
        logits = net.inference(x, subgraph_loader,device)
        y_pred = tlx.argmax(logits, -1)
        accs = []
        for mask in [graph.train_mask, graph.val_mask, graph.test_mask]:
            correct = tlx.convert_to_numpy(graph.y[mask] == y_pred[mask]).sum()
            accs.append(correct / tlx.convert_to_numpy(mask).sum().item())
        train_acc = accs[0]
        val_acc = accs[1]
        test_acc = accs[2]
        return train_acc,val_acc,test_acc

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--num_parts", type=int, default=1500, help="number of partition parts")
    parser.add_argument("--n_epoch", type=int, default=30, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--dataset_path", type=str, default=r'./reddit', help="path to save dataset")
    args = parser.parse_args()

    main(args)
