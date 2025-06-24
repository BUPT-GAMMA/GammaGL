# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
import argparse
from gammagl.datasets import Planetoid
from gammagl.models import GCNModel
from gammagl.utils import add_self_loops, mask_to_index
import torch
import torch.nn.functional as F
import tensorlayerx as tlx

def main(args):
    if args.device >= 0:
        tlx.set_device("GPU", args.device)
    else:
        tlx.set_device("CPU")
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)

    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    model = GCNModel(feature_dim=dataset.num_node_features,
                   hidden_dim=args.hidden_dim,
                   num_class=dataset.num_classes,
                   drop_rate=args.drop_rate,
                   num_layers=args.num_layers,
                   norm = args.norm,
                   name="GCN")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        # "edge_weight": edge_weight,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        logits = model(data['x'], data['edge_index'], None, data['num_nodes'])
        train_logits = logits[data['train_idx']]
        train_y = data['y'][data['train_idx']]
        loss = F.cross_entropy(train_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.set_eval()
        logits = model(data['x'], data['edge_index'], None, data['num_nodes'])
        val_logits = logits[data['val_idx']]
        val_y = data['y'][data['val_idx']]
        pred = torch.argmax(val_logits, dim=1)
        val_acc = (float((val_y==pred).sum()) / float(len(data['val_idx'])))

        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(loss.item())\
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')

    model.eval()
    logits = model(data['x'], data['edge_index'], None, data['num_nodes'])
    test_logits = logits[data['test_idx']]
    test_y = data['y'][data['test_idx']]
    pred = torch.argmax(test_logits, dim=1)
    test_acc = (float((test_y==pred).sum()) / float(len(data['test_idx'])))
    print("Test acc:  {:.4f}".format(test_acc))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--norm", type=str, default='both', help="how to apply the normalizer.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--device", type=int, default=0, help="the training device, -1 means cpu")
    args = parser.parse_args()

    main(args)
