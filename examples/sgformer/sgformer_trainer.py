# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   sgformer_trainer.py
@Time    :   2024/12/30 12:57:55
@Author  :   Cui Shanyuan
"""

import os
import argparse
import tensorlayerx as tlx
import numpy as np
from gammagl.datasets import Planetoid, WikipediaNetwork,Actor,DeezerEurope
from gammagl.models import SGFormerModel
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'], None, data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst

def main(args):

    if str.lower(args.dataset) in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(args.dataset_path, args.dataset)
        graph = dataset[0]
        train_idx = mask_to_index(graph.train_mask)
        test_idx = mask_to_index(graph.test_mask)
        val_idx = mask_to_index(graph.val_mask)
    elif str.lower(args.dataset) in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(args.dataset_path, args.dataset)
        graph = dataset[0]

        split_idx = 0
        current_dir = os.path.dirname(os.path.abspath(__file__))
        split_path = os.path.join(current_dir, args.dataset, 'geom_gcn', 'raw', 
                                 f'{args.dataset}_split_0.6_0.2_{split_idx}.npz')
        print(f"Looking for split file at: {split_path}")
        splits_file = np.load(split_path)
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
    elif str.lower(args.dataset) == 'actor':
        dataset = Actor(args.dataset_path)
        graph = dataset[0]

        split_idx = args.split_idx 
        train_idx = mask_to_index(graph.train_mask[:, split_idx])
        val_idx = mask_to_index(graph.val_mask[:, split_idx])
        test_idx = mask_to_index(graph.test_mask[:, split_idx])
    elif str.lower(args.dataset) == 'deezer':
        dataset = DeezerEurope(args.dataset_path)
        graph = dataset[0]

        num_nodes = graph.num_nodes
        train_ratio = 0.6
        val_ratio = 0.2
        

        indices = np.random.permutation(num_nodes)
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
        
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)


    net = SGFormerModel(feature_dim=dataset.num_node_features,
                       hidden_dim=args.hidden_dim,
                       num_class=dataset.num_classes,
                       trans_num_layers=args.trans_num_layers,
                       trans_num_heads=args.trans_num_heads,
                       trans_dropout=args.trans_dropout,
                       gnn_num_layers=args.gnn_num_layers,
                       gnn_dropout=args.gnn_dropout,
                       graph_weight=args.graph_weight,
                       name="SGFormer")

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)


    data = {
        "x": tlx.convert_to_tensor(graph.x),
        "y": tlx.convert_to_tensor(graph.y),
        "edge_index": tlx.convert_to_tensor(edge_index),
        "train_idx": tlx.convert_to_tensor(train_idx),
        "test_idx": tlx.convert_to_tensor(test_idx),
        "val_idx": tlx.convert_to_tensor(val_idx),
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path+net.name+".npz", format='npz_dict')

    net.load_weights(args.best_model_path+net.name+".npz", format='npz_dict')
    net.set_eval()
    logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--trans_num_layers', type=int, default=1, help='number of transformer layers')
    parser.add_argument('--trans_num_heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--trans_dropout', type=float, default=0.5, help='transformer dropout rate')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--gnn_dropout', type=float, default=0.5, help='GNN dropout rate')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='weight for GNN branch')
    parser.add_argument('--l2_coef', type=float, default=5e-4, help='l2 loss coefficient')
    parser.add_argument('--dataset', type=str, default='cora', 
                       choices=['cora', 'pubmed', 'citeseer', 'chameleon', 
                               'squirrel', 'actor', 'deezer'],
                       help='dataset name')
    parser.add_argument('--dataset_path', type=str, default=r'', help='path to save dataset')
    parser.add_argument('--best_model_path', type=str, default=r'./', help='path to save best model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--split_idx', type=int, default=0,
                       help='split index for actor dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    

    np.random.seed(args.seed)
    
    if args.gpu >= 0:
        tlx.set_device("cuda", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args) 