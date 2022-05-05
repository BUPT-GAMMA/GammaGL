# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   agnn_trainer.py
@Time    :   
@Author  :   shanyuxuan
"""

import os
os.environ['TL_BACKEND'] = 'torch' # set your backend here, default `tensorflow`
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
sys.path.insert(0, os.path.abspath('./')) # adds path2gammagl to execute in command line.
import time
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import AGNNModel
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils.loop import add_self_loops
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
        #Use l2_loss when there is no weight_decay in optimizer
        #l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._backbone.trainable_weights]) * args.weight_decay
        return loss #+ l2_loss



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
    dataset.process()  # suggest to execute explicitly so far

    graph = dataset[0]
    graph.tensor()
    edge_index, _ = add_self_loops(graph.edge_index, n_loops=1)
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
    optimizer = tlx.optimizers.Adam(learning_rate = args.lr,weight_decay=args.weight_decay)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(model, loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    '''
    x(2-D tensor): Shape = (num_nodes, feature_dims)
    edge_index(2-D tensor): Shape:(2, num_edges). A element(integer) of dim-1 expresses a node of graph and
        edge_index[0,i] points to edge_index[1,i].
    xxx_mask(1-D tensor): Shape = (num_edges). Spilt x into train, validation and test by mask.
    num_nodes: Number of nodes on the graph.
    '''
    data = {
        "x": x,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
    }

    val_acc = best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data, y)
        train_acc = evaluate(model, data, y, data['train_mask'], metrics)
        val_acc = evaluate(model, data, y, data['val_mask'], metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1),
               "   train loss: {:.4f}".format(train_loss.item()),
               "   train_acc:{:.4f}".format(train_acc),
               "   val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    test_acc = evaluate(model, data, y, data['test_mask'], metrics)
    print("Test acc:  {:.4f}".format(test_acc))
   


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


