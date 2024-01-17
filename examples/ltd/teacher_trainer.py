import os

os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorlayerx as tlx
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn, configs):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.configs = configs

    def forward(self, data, y):
        if self.configs['model'] == 'GCN':
            logits = self.backbone_network(data['x'], data['edge_index'], None, data['num_nodes'])
        elif self.configs['model'] == 'GAT':
            logits = self.backbone_network(data['x'], data['edge_index'], data['num_nodes'])
        else:
            raise ValueError('Unknown GNN: {}'.format(self.configs['model']))
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def teacher_trainer(graph, configs, net):
    print("-------------------teacher model training------------------------")
    edge_index, _ = add_self_loops(graph.edge_index, n_loops=1, num_nodes=graph.num_nodes)

    train_idx = mask_to_index(configs['t_train_mask'])
    test_idx = mask_to_index(configs['t_test_mask'])
    val_idx = mask_to_index(configs['t_val_mask'])

    loss = tlx.losses.softmax_cross_entropy_with_logits
    optimizer = tlx.optimizers.Adam(lr=configs['t_lr'], weight_decay=configs['l2_coef'])
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, loss, configs)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    for epoch in range(configs['n_epoch']):
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        if configs['model'] == 'GCN':
            logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
        elif configs['model'] == 'GAT':
            logits = net(data['x'], data['edge_index'], data['num_nodes'])
        else:
            raise ValueError('Unknown GNN: {}'.format(configs['model']))
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   train loss: {:.4f}".format(train_loss.item())
              + "   val acc: {:.4f}".format(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(net.name + ".npz", format='npz_dict')

    net.load_weights(net.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)
    net.set_eval()
    if configs['model'] == 'GCN':
        logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
    elif configs['model'] == 'GAT':
        logits = net(data['x'], data['edge_index'], data['num_nodes'])
    else:
        raise ValueError('Unknown GNN: {}'.format(configs['model']))
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))
    return logits, test_acc
