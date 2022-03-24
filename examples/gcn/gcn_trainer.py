# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gcn_trainer.py
@Time    :   2021/11/02 22:05:55
@Author  :   hanhui
"""

import os
os.environ['TL_BACKEND'] = 'tensorflow' # set your backend here, default `tensorflow`
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import time
import argparse
import numpy as np
import tensorflow as tf
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import GCNModel
from gammagl.utils.config import Config
from tensorlayerx.model import TrainOneStep, WithLoss


# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
parser.add_argument("--keep_rate", type=float, default=0.5, help="keep_rate = 1 - drop_rate")
parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
args = parser.parse_args()


best_model_path = r'./best_models/'
if not os.path.exists(best_model_path): os.makedirs(best_model_path)
# physical_gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_gpus) > 0:
#     # dynamic allocate gpu memory
#     tf.config.experimental.set_memory_growth(physical_gpus[0], True)


# load cora dataset
dataset = Planetoid("../", "pubmed")
dataset.process() # suggest to execute explicitly so far
graph = dataset[0]
graph.add_self_loop(n_loops=1) # self-loop trick
edge_weight = tlx.ops.convert_to_tensor(GCNModel.calc_gcn_norm(graph.edge_index, graph.num_nodes))
node_feat = graph.x
edge_index = graph.edge_index
node_label = tlx.argmax(graph.y,1)

# configurate
cfg = Config(feature_dim=node_feat.shape[1],
             hidden_dim=args.hidden_dim,
             num_class=graph.y.shape[1],
             keep_rate=args.keep_rate,)


# build model
model = GCNModel(cfg, name="GCN")
train_weights = model.trainable_weights
optimizer = tlx.optimizers.Adam(args.lr)
best_val_acc = 0.

# fit the training set
print('training starts')
start_time = time.time()

for epoch in range(args.n_epoch):
    # forward and optimize
    model.set_train()
    with tf.GradientTape() as tape:
        logits = model(node_feat, edge_index, edge_weight, graph.num_nodes)
        train_logits = logits[graph.train_mask]
        train_labels = node_label[graph.train_mask]
        train_loss = tlx.losses.softmax_cross_entropy_with_logits(train_logits, train_labels)
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tape.watched_variables()]) * args.l2_coef
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tape.watched_variables()[0]]) * args.l2_coef  # Only perform normalization on first convolution.
        loss = train_loss + l2_loss
    grad = tape.gradient(loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))
    train_acc = np.mean(np.equal(np.argmax(train_logits, 1), train_labels))
    log_info = "epoch [{:0>3d}]  train loss: {:.4f}, l2_loss: {:.4f}, total_loss: {:.4f}, train acc: {:.4f}".format(epoch+1, train_loss, l2_loss, loss, train_acc)

    # evaluate
    model.set_eval()  # disable dropout
    logits = model(node_feat, edge_index, edge_weight, graph.num_nodes)
    val_logits = logits[graph.val_mask]
    val_labels = node_label[graph.val_mask]
    val_loss = tlx.losses.softmax_cross_entropy_with_logits(val_logits, val_labels)
    val_acc = np.mean(np.equal(np.argmax(val_logits, 1), val_labels))
    log_info += ",  eval loss: {:.4f}, eval acc: {:.4f}".format(val_loss, val_acc)

    # save best model on evaluation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_weights(best_model_path+model.name+".npz")

    # # test when training
    # logits = model(x, sparse_adj)
    # test_logits = tlx.gather(logits, idx_test)
    # test_labels = tlx.gather(node_label, idx_test)
    # test_loss = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_labels)
    # test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_labels))
    # log_info += ",  test loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc)

    print(log_info)

end_time = time.time()
print("training ends in {t}s".format(t=int(end_time-start_time)))

# test performance
model.load_weights(best_model_path+model.name+".npz")
model.set_eval()
logits = model(node_feat, edge_index, edge_weight, graph.num_nodes)
test_logits = logits[graph.test_mask]
test_labels = node_label[graph.test_mask]
test_loss = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_labels)
test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_labels))
print("\ntest loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc))





"""
NOTE: The following codes are framework agnostic. 
However the can not applied l2 loss.
"""

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['node_feat'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        train_logits = logits[data['train_mask']]
        train_label = label[data['train_mask']]
        loss = self._loss_fn(train_logits, train_label)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self._backbone.trainable_weights[0]]) * args.l2_coef # only support for tensorflow backend
        return loss+l2_loss

loss = tlx.losses.softmax_cross_entropy_with_logits
optimizer = tlx.optimizers.Adam(args.lr)
metrics = tlx.metrics.Accuracy()
net = GCNModel(cfg, name="GCN")
train_weights = net.trainable_weights

loss_func = SemiSpvzLoss(net, loss)
train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

data = {
    "node_feat": node_feat,
    "edge_index": edge_index,
    "edge_weight": edge_weight,
    "train_mask": graph.train_mask,
    "test_mask": graph.test_mask,
    "val_mask": graph.val_mask,
    "num_nodes": graph.num_nodes,
}

best_val_acc = 0
for epoch in range(args.n_epoch):
    start_time = time.time()
    net.set_train()
    # train one step
    train_loss = train_one_step(data, node_label)
    # evaluate
    net.set_eval()
    _logits = net(data['node_feat'], data['edge_index'], data['edge_weight'], data['num_nodes'])

    train_logits = _logits[data['train_mask']]
    train_label = node_label[data['train_mask']]
    metrics.update(train_logits, train_label)
    train_acc = metrics.result()# [0] # depend on tl
    metrics.reset()

    val_logits = _logits[data['val_mask']]
    val_label = node_label[data['val_mask']]
    metrics.update(val_logits, val_label)
    val_acc = metrics.result()# [0]
    metrics.reset()

    print("Epoch [{:0>3d}]  ".format(epoch + 1)\
          + "   train loss: {:.4f}".format(train_loss)\
          + "   train acc: {:.4f}".format(train_acc)\
          + "   val acc: {:.4f}".format(val_acc))

    # save best model on evaluation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        net.save_weights(best_model_path+net.name+".npz", format='npz_dict')

net.load_weights(best_model_path+net.name+".npz", format='npz_dict')
net.set_eval()
logits = net(data['node_feat'], data['edge_index'], data['edge_weight'], data['num_nodes'])
test_logits = logits[data['test_mask']]
test_label = node_label[data['test_mask']]
metrics.update(test_logits, test_label)
test_acc = metrics.result()# [0]
metrics.reset()
print("Test acc:  {:.4f}".format(test_acc))

