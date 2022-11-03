import os
# os.environ['CUDA_VISIBLE_DEVICES']='5'
# os.environ['TL_BACKEND'] = 'paddle'
import sys
# sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
import numpy as np
from gammagl.datasets import Planetoid
# from gammagl.models import HCHA
from gammagl.utils import add_self_loops, calc_gcn_norm, mask_to_index, set_device
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models import HCHA

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'], data['edge_weight'],data['edge_attr'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


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


'''Data Preprocess'''
dataset = Planetoid(root='.', name='Cora')
dataset.process()
graph = dataset[0]
graph.tensor()
edge_index = np.array(graph.edge_index)  # 记录了哪些节点存在边
edge_weight = tlx.ones((edge_index.shape[1],))  # 记录了每天边的权重值
x = graph.x  # 每一个节点的特征
y = graph.y  # 每一个节点的label

'''label type'''
label_type = np.array(y)
label_type = set(list(label_type))
print(label_type)

'''hyper-edge construction'''
temp = []
hedge_map = {}
for i in range(len(edge_index[0])):
    if edge_index[0][i] not in temp:
        temp.append(edge_index[0][i])
        hedge_map[edge_index[0][i]] = [edge_index[0][i],edge_index[1][i]]
    else:
        hedge_map[edge_index[0][i]].append(edge_index[1][i])
hyperedge_index = [[],[]]
hyperedge_attr = []
for key, value in hedge_map.items():
    m = np.zeros(len(x[0]))
    count = 0
    for item in value:
        hyperedge_index[0].append(item) # node index
        hyperedge_index[1].append(key) # hyperedge index
        m += x[item]
        count += 1
    m = m/3
    hyperedge_attr.append(m)
hyperedge_attr = tlx.ops.convert_to_tensor(hyperedge_attr)
hyperedge_weight = tlx.ones((len(hyperedge_index[1]),))

'''training'''
lr = 0.01
n_epoch = 200
hidden_dim = 16
drop_rate = 0.5
l2_coef = 5e-4
best_model_path = r'./'

# for mindspore, it should be passed into node indices
train_idx = mask_to_index(graph.train_mask)
test_idx = mask_to_index(graph.test_mask)
val_idx = mask_to_index(graph.val_mask)
ea_len = len(hyperedge_attr[0])

net = HCHA(in_channels=dataset.num_node_features,
                hidden_channels=hidden_dim,
                out_channels=dataset.num_classes,
                name="HCHA",
                use_attention=False, 
                heads=1,
                negative_slope=0.2, dropout=drop_rate, bias=True,ea_len=ea_len)

optimizer = tlx.optimizers.Adam(lr=lr, weight_decay=l2_coef)
metrics = tlx.metrics.Accuracy()
train_weights = net.trainable_weights

loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

data = {
    "x": graph.x,
    "y": graph.y,
    "edge_index": tlx.ops.convert_to_tensor(hyperedge_index),
    "edge_weight": tlx.ops.convert_to_tensor(hyperedge_weight),
    "edge_attr": tlx.ops.convert_to_tensor(hyperedge_attr),
    "train_idx": train_idx,
    "test_idx": test_idx,
    "val_idx": val_idx,
    "num_nodes": graph.num_nodes,
}

best_val_acc = 0
for epoch in range(n_epoch):
    print('epoch',epoch,':')
    net.set_train()
    train_loss = train_one_step(data, graph.y)
    net.set_eval()
    logits = net(x=data['x'], hyperedge_index=data['edge_index'], hyperedge_weight=data['edge_weight'], hyperedge_attr=data['edge_attr'])
    val_logits = tlx.gather(logits, data['val_idx'])
    val_y = tlx.gather(data['y'], data['val_idx'])
    val_acc = calculate_acc(val_logits, val_y, metrics)

    print("Epoch [{:0>3d}] ".format(epoch+1)\
            + "  train loss: {:.4f}".format(train_loss.item())\
            + "  val acc: {:.4f}".format(val_acc))

    # save best model on evaluation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        net.save_weights(best_model_path+net.name+".npz", format='npz_dict')

net.load_weights(best_model_path+net.name+".npz", format='npz_dict')
if tlx.BACKEND == 'torch':
    net.to(data['x'].device)
net.set_eval()
logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
test_logits = tlx.gather(logits, data['test_idx'])
test_y = tlx.gather(data['y'], data['test_idx'])
test_acc = calculate_acc(test_logits, test_y, metrics)
print("Test acc:  {:.4f}".format(test_acc))