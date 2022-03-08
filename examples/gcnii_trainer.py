# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gcnii_trainer.py
@Time    :   2021/12/22 16:12:02
@Author  :   Han Hui
'''

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# sys.path.insert(0, os.path.abspath('../')) # adds path2gammagl to execute in command line.
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from gammagl.datasets import Cora
from gammagl.models import GCNIIModel
from gammagl.utils.config import Config
from tensorlayer.models import TrainOneStep, WithLoss
from gammagl.utils import calc_A_norm_hat
from gammagl.sparse.sparse_adj import SparseAdj

# parameters setting
n_epoch = 1000
learning_rate = 0.01
hidden_dim = 64
keep_rate = 0.6
l2_norm = 5e-4
alpha = 0.1
beta = 0.5
lambd = 0.5
num_layers = 64
variant = True

cora_path = r'../gammagl/datasets/raw_file/cora/'
best_model_path = r'./best_models/'

# physical_gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_gpus) > 0:
#     # dynamic allocate gpu memory
#     tf.config.experimental.set_memory_growth(physical_gpus[0], True)


class NetWithLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(NetWithLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = tl.gather(
            self._backbone(data['x'], data['sparse_adj']),
            data['idx_train']
        )
        label = tl.gather(label, data['idx_train'])
        loss = self._loss_fn(logits, label)
        return loss


class GCNTrainOneStep(TrainOneStep):
    def __init__(self, net_with_loss, optimizer, train_weights):
        super().__init__(net_with_loss, optimizer, train_weights)
        self.train_weights = train_weights

    def __call__(self, data, label):
        loss = self.net_with_train(data, label)
        return loss


# load dataset
dataset = Cora(cora_path)
sparse_adj = SparseAdj.from_sparse(calc_A_norm_hat(dataset.edges))
x = tl.ops.convert_to_tensor(dataset.features)
node_label = tl.ops.convert_to_tensor(dataset.labels)
idx_train = tl.ops.convert_to_tensor(dataset.idx_train)
idx_test = tl.ops.convert_to_tensor(dataset.idx_test)
idx_val = tl.ops.convert_to_tensor(dataset.idx_val)


# configurate and build model
cfg = Config(feature_dim=dataset.feature_dim,
             hidden_dim=hidden_dim,
             num_class=dataset.num_class,
             alpha = alpha,
             beta = beta,
             lambd = lambd,
             variant = variant,
             num_layers = num_layers,
             keep_rate=keep_rate,)

loss = tl.cost.softmax_cross_entropy_with_logits
# metrics = tl.metric.Accuracy() # zhen sb
net = GCNIIModel(cfg, name="GCNII")
train_weights = net.trainable_weights
optimizer = tl.optimizers.Adam(learning_rate)
best_val_acc = 0.

net_with_loss = NetWithLoss(net, loss)
train_one_step = GCNTrainOneStep(net_with_loss, optimizer, train_weights)

data = {
    "x": x,
    "sparse_adj": sparse_adj,
    "idx_train": idx_train,
    "idx_test": idx_test,
    "idx_val": idx_val
}

# fit the training set
print('training starts')
for epoch in range(n_epoch):
    start_time = time.time()
    net.set_train()
    # train one step
    train_loss = train_one_step(data, node_label)
    # eval
    _logits = net(data['x'], data['sparse_adj'])

    train_logits = tl.gather(_logits, data['idx_train'])
    train_label = tl.gather(node_label, data['idx_train'])
    # metrics.update(train_logits, train_label)
    # train_acc = metrics.result()
    # metrics.reset()
    train_acc = np.mean(np.equal(np.argmax(train_logits, 1), train_label))

    val_logits = tl.gather(_logits, data['idx_val'])
    val_label = tl.gather(node_label, data['idx_val'])
    # metrics.update(val_logits, val_label)
    # val_acc = metrics.result()
    # metrics.reset()
    val_acc = np.mean(np.equal(np.argmax(val_logits, 1), val_label))

    print("Epoch {} ".format(epoch + 1)\
          + "   train loss: {:.4f}".format(train_loss)\
          + "   train acc:  {:.4f}".format(train_acc)\
          + "   val acc:  {:.4f}".format(val_acc))

    # save best model on evaluation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        net.save_weights(best_model_path+net.name+".npz", format='npz_dict')

net.load_weights(best_model_path+net.name+".npz", format='npz_dict')
test_logits = tl.gather(net(data['x'], data['sparse_adj']), data['idx_test'])
test_label = tl.gather(node_label, data['idx_test'])
# metrics.update(test_logits, label)
# test_acc = metrics.result()
# metrics.reset()
test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_label))
print("Test acc:  {:.4f}".format(test_acc))

