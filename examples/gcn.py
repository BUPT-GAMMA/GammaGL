# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   gcn.py
@Time    :   2021/11/02 22:05:55
@Author  :   hanhui
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# sys.path.insert(0, os.path.abspath('../')) # adds path2gammagraphlibrary to execute in command line.
import sys
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from gammagraphlibrary.datasets import Cora
from gammagraphlibrary.layers.conv import GCNConv


# super parameters
n_epoch = 500           # 200
learning_rate = 0.01    # 0.01
hidden_dim = 16         # 16
keep_rate = 0.9         # 0.5
l2_norm = 3e-4          # 5e-4
renorm = True           # True
improved = False        # False

# configuration
cora_path = r'../gammagraphlibrary/datasets/raw_file/cora/'
best_model_path = r'./best_models/'
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if len(physical_gpus) > 0:
    # dynamic allocate gpu memory
    tf.config.experimental.set_memory_growth(physical_gpus[0], True)


# load cora dataset
dataset = Cora(cora_path)
graph, idx_train, idx_val, idx_test = dataset.process()


# build GCN
class GCNModel(tl.layers.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.conv1 = GCNConv(dataset.feature_dim, hidden_dim, renorm=renorm, improved=improved)
        self.conv2 = GCNConv(hidden_dim, dataset.num_class, renorm=renorm, improved=improved)
        self.relu = tl.ReLU()
        self.dropout = tl.Dropout(keep=keep_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

model = GCNModel(name="GCN")
train_weights = model.trainable_weights
optimizer = tl.optimizers.Adam(learning_rate)
best_val_acc = 0.


print('training starts')
x, edge_index = graph.node_feat, graph.edge_index
start_time = time.time()

for epoch in range(n_epoch):
    # forward and optimize
    model.set_train()
    with tf.GradientTape() as tape:
        logits = model(x, edge_index)
        train_logits = tf.gather(logits, idx_train)
        train_labels = tf.gather(graph.node_label, idx_train)
        train_loss = tl.cost.softmax_cross_entropy_with_logits(train_logits, train_labels)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tape.watched_variables()]) * l2_norm
        # l2_loss = tf.nn.l2_loss(tape.watched_variables()[0]) * l2_norm  # Only perform normalization on first convolution.
        loss = train_loss + l2_loss
    grad = tape.gradient(loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))
    train_acc = np.mean(np.equal(np.argmax(train_logits, 1), train_labels))
    log_info = "epoch [{:0>3d}]  train loss: {:.4f}, l2_loss: {:.4f}, total_loss: {:.4f}, train acc: {:.4f}".format(epoch+1, train_loss, l2_loss, loss, train_acc)

    # evaluate
    model.set_eval()  # disable dropout
    logits = model(x, edge_index)
    val_logits = tf.gather(logits, idx_val)
    val_labels = tf.gather(graph.node_label, idx_val)
    val_loss = tl.cost.softmax_cross_entropy_with_logits(val_logits, val_labels)
    val_acc = np.mean(np.equal(np.argmax(val_logits, 1), val_labels))
    log_info += ",  eval loss: {:.4f}, eval acc: {:.4f}".format(val_loss, val_acc)
    if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(best_model_path+model.name+".npz")

    # # test when training
    # logits = model(x, edge_index)
    # test_logits = tf.gather(logits, idx_test)
    # test_labels = tf.gather(graph.node_label, idx_test)
    # test_loss = tl.cost.softmax_cross_entropy_with_logits(test_logits, test_labels)
    # test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_labels))
    # log_info += ",  test loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc)

    print(log_info)

end_time = time.time()
print("training ends in {t}s".format(t=int(end_time-start_time)))

# test
model.load_weights(best_model_path+model.name+".npz")
model.set_eval()
logits = model(x, edge_index)
test_logits = tf.gather(logits, idx_test)
test_labels = tf.gather(graph.node_label, idx_test)
test_loss = tl.cost.softmax_cross_entropy_with_logits(test_logits, test_labels)
test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_labels))
print("\ntest loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc))

