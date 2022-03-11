# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sgc_trainer.py
@Time    :   2021/12/16 20:31:03
@Author  :   Han Hui
'''


import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.abspath('../')) # adds path2gammagl to execute in command line.
import time
import numpy as np
import tensorflow as tf
import tensorlayerx as tlx
from gammagl.datasets import Cora
from gammagl.models import SGCModel, GCNModel
from gammagl.utils.config import Config

# parameters setting
n_epoch = 200
learning_rate = 0.2
itera_K=2
l2_norm = 1e-5

cora_path = r'../gammagl/datasets/raw_file/cora/'
best_model_path = r'./best_models/'
# physical_gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_gpus) > 0:
#     # dynamic allocate gpu memory
#     tf.config.experimental.set_memory_growth(physical_gpus[0], True)


# load cora dataset
dataset = Cora(cora_path)
graph, idx_train, idx_val, idx_test = dataset.process()
edge_weight = GCNModel.calc_gcn_norm(graph.edge_index, graph.num_nodes)
edge_index = tlx.ops.convert_to_tensor(graph.edge_index)
x = tlx.convert_to_tensor(graph.node_feat)
node_label = tlx.convert_to_tensor(graph.node_label)
idx_train = tlx.convert_to_tensor(idx_train)
idx_test = tlx.convert_to_tensor(idx_test)
idx_val = tlx.convert_to_tensor(idx_val)

# configurate and build model
cfg = Config(feature_dim=dataset.feature_dim,
             num_class=dataset.num_class,
             itera_K=itera_K)

model = SGCModel(cfg, name="SGC")
train_weights = model.trainable_weights
optimizer = tlx.optimizers.Adam(learning_rate)
best_val_acc = 0.

# fit the training set
print('training starts')
start_time = time.time()

for epoch in range(n_epoch):
    # forward and optimize
    model.set_train()
    with tf.GradientTape() as tape:
        logits = model(x, edge_index, edge_weight, graph.num_nodes)
        train_logits = tlx.gather(logits, idx_train)
        train_labels = tlx.gather(graph.node_label, idx_train)
        train_loss = tlx.losses.softmax_cross_entropy_with_logits(train_logits, train_labels)
        l2_loss = tlx.add_n([tf.nn.l2_loss(v) for v in tape.watched_variables()]) * l2_norm
        loss = train_loss + l2_loss
    grad = tape.gradient(loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))
    train_acc = np.mean(np.equal(np.argmax(train_logits, 1), train_labels))
    log_info = "epoch [{:0>3d}]  train loss: {:.4f}, l2_loss: {:.4f}, total_loss: {:.4f}, train acc: {:.4f}".format(epoch + 1, train_loss, l2_loss, loss, train_acc)

    # evaluate
    model.set_eval()  # disable dropout
    logits = model(x, edge_index, edge_weight, graph.num_nodes)
    val_logits = tlx.gather(logits, idx_val)
    val_labels = tlx.gather(graph.node_label, idx_val)
    val_loss = tlx.losses.softmax_cross_entropy_with_logits(val_logits, val_labels)
    val_acc = np.mean(np.equal(np.argmax(val_logits, 1), val_labels))
    log_info += ",  eval loss: {:.4f}, eval acc: {:.4f}".format(val_loss, val_acc)

    # save best model on evaluation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_weights(best_model_path + model.name + ".npz")

    # # test when training
    # logits = model(x, sparse_adj)
    # test_logits = tlx.gather(logits, idx_test)
    # test_labels = tlx.gather(graph.node_label, idx_test)
    # test_loss = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_labels)
    # test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_labels))
    # log_info += ",  test loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc)

    print(log_info)

end_time = time.time()
print("training ends in {t}s".format(t=int(end_time - start_time)))

# test performance
model.load_weights(best_model_path + model.name + ".npz")
model.set_eval()
logits = model(x, edge_index, edge_weight, graph.num_nodes)
test_logits = tlx.gather(logits, idx_test)
test_labels = tlx.gather(graph.node_label, idx_test)
test_loss = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_labels)
test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_labels))
print("\ntest loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc))
