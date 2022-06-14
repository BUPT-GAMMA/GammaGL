# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   agnn_trainer.py
@Time    :   
@Author  :   shanyuxuan
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['TL_BACKEND'] = 'torch'
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
sys.path.insert(0, os.path.abspath('./')) # adds path2gammagl to execute in command line.
import time
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import AGNNModel
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils import add_self_loops, mask_to_index


class SemiSpvzLoss(WithLoss):
    def __init__(self, model, loss_fn):
        super(SemiSpvzLoss,self).__init__(backbone = model, loss_fn = loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data['x'], data["edge_index"], data["num_nodes"])
        train_logits = tlx.gather(logits, data["train_idx"])
        train_y = tlx.gather(data["y"], data["train_idx"])
        loss = self._loss_fn(train_logits, train_y)
        #Use l2_loss when there is no weight_decay in optimizer
        #l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._backbone.trainable_weights]) * args.l2_coef
        return loss #+ l2_loss


def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """

    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def main(args):
    #load dataset
    if(str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']):
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, n_loops=1, num_nodes=graph.num_nodes)
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)
    x = tlx.l2_normalize(graph.x,axis=-1)
    y = graph.y
    num_nodes = graph.num_nodes

    model = AGNNModel(feature_dim = dataset.num_node_features,
                      hidden_dim = args.hidden_dim,
                      num_class = dataset.num_classes,
                      n_att_layers = args.n_att_layers,
                      dropout_rate = args.drop_rate,
                      is_cora = (args.dataset == 'cora'),       #is_cora: Whether the dateset is cora. Cora dataset contains two agnn_conv layers and others contain four.
                      name = "AGNN")
    
    loss = tlx.losses.softmax_cross_entropy_with_logits
    optimizer = tlx.optimizers.Adam(lr = args.lr,weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(model, loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes":num_nodes
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data, y)
        model.set_eval()
        logits = model(data["x"], data["edge_index"], data["num_nodes"])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1),
               "   train loss: {:.4f}".format(train_loss.item()),
               "   val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        model.to(data["x"].device)
    model.set_eval()
    logits = model(data["x"], data["edge_index"], data["num_nodes"])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))
   


if __name__ == "__main__":
    # paramenters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 0.005)
    parser.add_argument("--n_epoch", type = int, default = 1000)
    parser.add_argument("--hidden_dim", type=float, default = 16)
    parser.add_argument("--drop_rate", type = float, default = 0.5)
    parser.add_argument("--n_att_layers", type = int, default = 2)
    parser.add_argument("--l2_coef", type = float, default = 5e-4)
    parser.add_argument("--dataset", type = str, default = "cora")
    parser.add_argument("--dataset_path", type = str, default = r"../")
    parser.add_argument("--best_model_path", type = str, default = r"./")
    
    args = parser.parse_args()

    main(args)


