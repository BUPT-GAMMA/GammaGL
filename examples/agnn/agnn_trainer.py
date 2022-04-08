# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   agnn_trainer.py
@Time    :   
@Author  :   shanyuxuan
"""

import os
os.environ['TL_BACKEND'] = 'tensorflow' # set your backend here, default `tensorflow`
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
sys.path.insert(0, os.path.abspath('./')) # adds path2gammagl to execute in command line.
from adam import AdamWeightDecayOptimizer
import time
import argparse
import tensorflow as tf
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import AGNNModel
from tensorlayerx.model import TrainOneStep, WithLoss
import tensorflow_addons as tfa
import copy


#
class SemiSpvzLoss(WithLoss):
    def __init__(self, model, loss_fn):
        super(SemiSpvzLoss,self).__init__(backbone = model, loss_fn = loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['x'])
        train_logits = logits[data['train_mask']]
        train_label = label[data['train_mask']]
        loss = self._loss_fn(train_logits, train_label)
        #TODO:weight_decay在AdamW优化器中使用
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._backbone.trainable_weights]) * args.weight_decay
        return loss + l2_loss



def evaluate(model, data, y, mask, metrics):
    model.set_eval()
    logits = model(data['x'])
    _logits = logits[mask]
    _label = y[mask]
    metrics.update(_logits, _label)
    acc = metrics.result()  # [0]
    metrics.reset()
    return acc




def main():
    #load dataset
    if(str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']):
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    #TODO:这里增加自环
    graph.add_self_loop(n_loops=1)
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    x = graph.x
    x = tlx.ops.l2_normalize(x,axis=-1)
    y = tlx.argmax(graph.y,1)

    model = AGNNModel(feature_dim = x.shape[1],
                      hidden_dim = args.hidden_dim,
                      num_class = graph.y.shape[1],
                      n_att_layers = args.n_att_layers,
                      dropout_rate = args.dropout_rate,
                      edge_index = edge_index,
                      num_nodes = num_nodes,
                      is_cora = (args.dataset == 'cora'),
                      name = "AGNN")
    
    loss = tlx.losses.softmax_cross_entropy_with_logits
    #optimizer = tlx.optimizers.Adam(learning_rate = args.lr,weight_decay=args.weight_decay)
    optimizer = tlx.optimizers.Adam(learning_rate = args.lr)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(model, loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    '''
    x: 输入特征向量,shape = (num_nodes, feature_dims)
    edge_index: 边索引, shape = (2, num_edges)??打印出来的边和数据有点对不上
    xxx_mask: list,元素为true or false, 用于划分数据集
    num_nodes: 点的个数
    '''
    #edge_index, num_nodes直接传入模型中
    data = {
        "x": x,
        #"edge_index": edge_index,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
        #"num_nodes": graph.num_nodes,
    }
    
    val_acc = best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data, y)
        train_acc = evaluate(model, data, y, data['train_mask'], metrics)
        val_acc = evaluate(model, data, y, data['val_mask'], metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1),
               "   train loss: {:.4f}".format(train_loss),
               "   train_acc:{:.4f}".format(train_acc),
               "   val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    test_acc = evaluate(model, data, y, data['test_mask'], metrics)
    print('{},{},{:.4f},{:.4f}'.format(args.lr, args.weight_decay, args.dropout_rate, test_acc))
    #print("Test acc:  {:.4f}".format(test_acc))
   


if __name__ == "__main__":
    # paramenters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 0.005)
    parser.add_argument("--n_epoch", type = int, default = 1000)
    parser.add_argument("--hidden_dim", type=float, default = 16)
    parser.add_argument("--dropout_rate", type = float, default = 0.5)
    parser.add_argument("--n_att_layers", type = int, default = 2)
    parser.add_argument("--weight_decay", type = float, default = 5e-4)
    parser.add_argument("--dataset", type = str, default = "cora")
    parser.add_argument("--dataset_path", type = str, default = r"../")
    parser.add_argument("--best_model_path", type = str, default = r"./")
    
    args = parser.parse_args()

    main()


