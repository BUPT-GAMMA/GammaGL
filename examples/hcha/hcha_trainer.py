import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['TL_BACKEND'] = 'tensorflow'
import sys
# sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
import numpy as np
from gammagl.datasets import Planetoid
from gammagl.utils import add_self_loops, calc_gcn_norm, mask_to_index, set_device
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models import HCHA
# tlx.set_device("GPU", 1)

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'], data['edge_weight'], data['edge_attr'])
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

def main(args):
    '''Data Preprocess'''
    dataset = Planetoid(root='.', name=args.dataset)
    dataset.process()
    graph = dataset[0]
    graph.tensor()
    edge_index = graph.edge_index # Record which nodes have edges
    # edge_weight = tlx.ones((edge_index.shape[1],))  # Record each edge weights
    x = graph.x  # Record features of each node
    # y = graph.y  # Record label of each node

    '''hyper-edge construction'''
    temp = []
    hedge_map = {}
    edge_index_numpy = tlx.convert_to_numpy(edge_index)
    for i in range(len(edge_index_numpy[0])):
        if edge_index_numpy[0][i] not in temp:
            temp.append(edge_index_numpy[0][i])
            hedge_map[edge_index_numpy[0][i]] = [edge_index_numpy[0][i],edge_index_numpy[1][i]]
        else:
            hedge_map[edge_index_numpy[0][i]].append(edge_index_numpy[1][i])
    hyperedge_index = [[],[]]
    hyperedge_attr = np.zeros((max(edge_index[0])+1,len(x[0])))
    for key, value in hedge_map.items():
        m = np.zeros(len(x[0]))
        count = 0
        for item in value:
            hyperedge_index[0].append(item) # node index
            hyperedge_index[1].append(key) # hyperedge index
            m += tlx.convert_to_numpy(x[int(item)])
            count += 1
        m = m/count
        edge_id = item
        hyperedge_attr[edge_id] = m

    hyperedge_attr = tlx.ops.convert_to_tensor(hyperedge_attr, dtype = tlx.float32)
    hyperedge_weight = tlx.ones((len(hyperedge_index[1]),))
    
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)
    ea_len = len(hyperedge_attr[0])
    net = HCHA(in_channels=dataset.num_node_features,
                    hidden_channels=args.hidden_dim,
                    out_channels=dataset.num_classes,
                    ea_len=ea_len,
                    name="HCHA",
                    use_attention=args.use_attention, 
                    heads=2,
                    negative_slope=0.2, dropout=args.drop_rate, bias=True)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": tlx.convert_to_tensor(hyperedge_index, dtype = tlx.int64),
        "edge_weight": hyperedge_weight,
        "edge_attr": hyperedge_attr,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
    }

    best_val_acc = 0
    for epoch in range(args.n_epoch):
        # print('epoch',epoch,':')
        net.set_train()
        train_loss = train_one_step(data, graph.y)
        net.set_eval()
        logits = net(x = data['x'], hyperedge_index = data['edge_index'], 
                     hyperedge_weight = data['edge_weight'], hyperedge_attr = data['edge_attr'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch+1)\
                + "  train loss: {:.4f}".format(train_loss.item())\
                + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path+net.name+".npz", format='npz_dict')

    net.load_weights(args.best_model_path+net.name+".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)
    net.set_eval()
    logits = net(data['x'], data['edge_index'], data['edge_weight'], data['edge_attr'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))

if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=64, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--use_attention", type=bool, default=False, help="use attention or not")
    args = parser.parse_args()
    main(args)