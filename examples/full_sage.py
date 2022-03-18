import tensorlayerx as tlx
import tensorflow as tf
from gammagl.datasets import Planetoid
from gammagl.layers.conv import SAGEConv
import numpy as np
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if len(physical_gpus) > 0:
    # dynamic allocate gpu memory
    tf.config.experimental.set_memory_growth(physical_gpus[0], True)

data = 'cora'

dataset = Planetoid("./", "Cora")
dataset.process()
g = dataset[0]


# g = Graph(edge_index=np.array([row, col]), edge_weight=tl.convert_to_tensor(weight), node_feat=x, node_label=y)

# g = Graph(edge_index=edge, x=x, y=y, num_nodes=2708)
# g.train_mask = train_mask
# g.val_mask = val_mask
# g.test_mask = test_mask
node_label = tlx.ops.convert_to_tensor(tlx.argmax(g.y,1))
# FULL Graph
class GraphSAGE(tlx.nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, num_layers, drop, act, aggr='gcn'):
        super(GraphSAGE, self).__init__()
        self.drop = tlx.layers.Dropout(keep=1-drop)
        self.act = act
        self.num_layers = num_layers
        self.convs = tlx.nn.SequentialLayer()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels=in_feat, out_channels=out_feat, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels=in_feat, out_channels=hid_feat, aggr=aggr))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(in_channels=hid_feat, out_channels=hid_feat, aggr=aggr))
            self.convs.append(SAGEConv(in_channels=hid_feat, out_channels=out_feat, aggr=aggr))

    def forward(self, feat, edge):
        h = feat
        # num_nodes = max(edge[1])
        for l, conv in enumerate(self.convs):
            # 输入的feat应为(src_feat, dst_feat),
            h = conv([h, h], edge)
            if l != self.num_layers - 1:
                h = self.act(h)
                h = self.drop(h)
        return h

model = GraphSAGE(in_feat=1433, hid_feat=32, out_feat=7, num_layers=1, drop=0.2, aggr="mean", act=tlx.nn.ReLU())

learning_rate = 0.005
n_epoch = 200
train_weights = model.trainable_weights
best_val_acc = 0.
l2 = 1e-4
print('training starts')
maxacc = 0.
optimizer = tlx.optimizers.Adam(learning_rate)
for epoch in range(n_epoch):
    # forward and optimize
    model.set_train()
    with tf.GradientTape() as tape:
        logits = model(g.x, g.edge_index)
        train_logits = logits[g.train_mask] # tlx.gather(logits, idx_train)
        train_labels = node_label[g.train_mask] # tlx.gather(node_label, idx_train)
        train_loss = tlx.losses.softmax_cross_entropy_with_logits(train_logits, train_labels)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tape.watched_variables()]) * l2
        # l2_loss = tf.nn.l2_loss(tape.watched_variables()[0]) * l2_norm  # Only perform normalization on first convolution.
        loss = train_loss + l2_loss
    grad = tape.gradient(loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))
    train_acc = np.mean(np.equal(np.argmax(train_logits, 1), train_labels))
    log_info = "epoch [{:0>3d}]  train loss: {:.4f}, l2_loss: {:.4f}, total_loss: {:.4f}, train acc: {:.4f}".format(epoch+1, train_loss, l2_loss, loss, train_acc)
    # evaluate
    model.set_eval()  # disable dropout
    logits = model(g.x, g.edge_index)
    val_logits = logits[g.val_mask]  # tlx.gather(logits, idx_val)
    val_labels = node_label[g.val_mask]  # tlx.gather(node_label, idx_val)
    val_loss = tlx.losses.softmax_cross_entropy_with_logits(val_logits, val_labels)
    val_acc = np.mean(np.equal(np.argmax(val_logits, 1), val_labels))
    log_info += ",  eval loss: {:.4f}, eval acc: {:.4f}".format(val_loss, val_acc)
    best_model_path = '/best_models'

    # save best model on evaluation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_weights(best_model_path + model.name + ".npz")
    # test when training
    logits = model(g.x, g.edge_index)
    test_logits = logits[g.test_mask]
    test_labels = node_label[g.test_mask]
    test_loss = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_labels)
    test_acc = np.mean(np.equal(np.argmax(test_logits, 1), test_labels))
    log_info += ",  test loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc)
    maxacc = max(maxacc, test_acc)
    print(log_info)
print(maxacc)





