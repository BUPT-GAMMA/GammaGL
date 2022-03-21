"""
@File    :   gin.py
@Time    :   2022/03/07
@Author  :   Tianyu Zhao
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# sys.path.insert(0, os.path.abspath('../')) # adds path2gammagl to execute in command line.
import sys
import time
import numpy as np
import tensorflow as tf
import tensorlayerx as tlx
from gammagl.datasets import TUDataset
from gammagl.loader import DataLoader
from gammagl.models import GINModel
from collections import defaultdict
a = defaultdict(list)
dataset = TUDataset('./', 'MUTAG')
print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')

data = dataset[0]  # Get the first graph object.
# dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# super parameters
n_epoch = 10  # 200
learning_rate = 0.01  # 0.01
hidden_dim = 16  # 16
keep_rate = 0.9  # 0.5
l2_norm = 3e-4  # 5e-4
renorm = True  # True
improved = False  # False

# configuration
best_model_path = r'./best_models/'


# physical_gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_gpus) > 0:
#     # dynamic allocate gpu memory
#     tf.config.experimental.set_memory_growth(physical_gpus[0], True)


model = GINModel(name="GIN", input_dim=dataset.num_features, hidden=64, out_dim=dataset.num_classes)
train_weights = model.trainable_weights
optimizer = tlx.optimizers.Adam(learning_rate)
best_val_acc = 0.

print('training starts')


def train():
    model.set_train()
    correct = 0
    all_loss = 0
    for data in train_loader:
        with tf.GradientTape() as tape:
            out = model(data.x, data.edge_index, data.batch)
            train_loss = tlx.losses.softmax_cross_entropy_with_logits(out, tlx.cast(data.y, dtype=tlx.int32))
            all_loss += train_loss
            pred = tlx.ops.argmax(out, axis=1)
            correct += tlx.ops.reduce_sum(tlx.cast((pred == data.y), dtype=tlx.int64))
        grad = tape.gradient(train_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))
    print(all_loss)
    print(correct / len(train_loader.dataset))
        # train_acc = np.mean(np.equal(np.argmax(train_logits, 1), train_labels))
        # log_info = "epoch [{:0>3d}]  train loss: {:.4f}, l2_loss: {:.4f}, total_loss: {:.4f}, train acc: {:.4f}".format(
        #     epoch + 1, train_loss, l2_loss, loss, train_acc)


def test(loader):
    model.set_eval()  # disable dropout
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = tlx.ops.argmax(out, axis=1)
        correct += tlx.ops.reduce_sum(tlx.cast((pred == data.y), dtype=tlx.int64))
    return correct / len(loader.dataset)


for epoch in range(1, 121):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
