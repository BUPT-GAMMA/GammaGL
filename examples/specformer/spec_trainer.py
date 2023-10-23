"""
@File    :   spec_trainer.py
@Time    :   2023/10/15
@Author  :   ZZH
"""

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
os.environ['TL_BACKEND'] = "torch"
#
#
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorlayerx as tlx
from gammagl.datasets import WikipediaNetwork,Planetoid
from tensorlayerx.model import TrainOneStep
import argparse
import gammagl.transforms as T
import numpy as np
from gammagl.models.specformer import *
try:
    import cPickle as pickle
except ImportError:
    import pickle
if tlx.BACKEND == 'torch':  # when the backend is torch and you want to use GPU
    try:
        tlx.set_device(device='GPU', id=0)
    except:
        print("GPU is not available")

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




class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y,e,u):
        logits = self.backbone_network(data['x'], data['edge_index'],e,u)

        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss





def main(args):

    # 3 datasets : Chameleon , Squirrel,Cora
    if args.dataset == "cora":
            dataset = Planetoid(root="./", name=args.dataset,split="full")
    else:# Chameleon , Squirrel
            dataset = WikipediaNetwork(root="./", name=args.dataset, transform=T.NormalizeFeatures())


    graph = dataset[0]
    edge_index = graph.edge_index
    train_idx,val_idx,test_idx = get_split(y=graph.y , nclass = dataset.num_classes)

    net = Specformer(nclass=dataset.num_classes , nfeat=graph.num_node_features , nlayer=args.n_layer,
                     hidden_dim=args.hidden_dim , nheads=args.n_heads ,
                     tran_dropout=args.tran_dropout, feat_dropout=args.feat_dropout, prop_dropout=args.prop_dropout,)
    optimizer = tlx.optimizers.Adam(lr=args.lr,weight_decay=args.weight_decay)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights
    loss_func = SemiSpvzLoss(net=net, 
                             loss_fn=tlx.losses.softmax_cross_entropy_with_logits)
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
    best_val_acc = 0.0

    num_nodes = graph.num_nodes
    adj = np.zeros((num_nodes, num_nodes))
    source = edge_index[0]
    target = edge_index[1]
    for i in range(len(source)):
        adj[source[i], target[i]] = 1.
        adj[target[i], source[i]] = 1.

    # adj is the adjacent matrix of graph
    e, u = eigen_decompositon(adj)
    e = tlx.convert_to_tensor(e)
    u = tlx.convert_to_tensor(u)
    
    for now_epoch in range(args.n_epoch):

        net.set_train()
        train_loss = train_one_step(data, graph.y,e,u)
        net.set_eval()
        logits = net(data['x'], data['edge_index'],e,u)
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)
        #every epoch ,print val acc
        print("Epoch [{:0>3d}] ".format(now_epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            net.save_weights("./"+ net.name +"__" +args.dataset +".npz", format='npz_dict')


    # after 2000 epochs, get a best model parameter
    net.load_weights("./"+ net.name +"__" +args.dataset +".npz", format='npz_dict')
    net.set_eval()
    logits = net(data['x'], data['edge_index'],e,u)
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    #print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight_decay")
    parser.add_argument("--n_epoch", type=int, default=2000, help="number of epoch")
    parser.add_argument("--n_layer", type=int, default=2, help="number of Speclayers")
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads")
    parser.add_argument("--hidden_dim", type=int, default=32, help="dimension of hidden layers")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--tran_dropout", type=float, default=0.2, help="tran_dropout")
    parser.add_argument("--feat_dropout", type=float, default=0.6, help="feat_dropout")
    parser.add_argument("--prop_dropout", type=float, default=0.2, help="prop_dropout")


    args = parser.parse_args()

    number = []
    for i in range(10):
        acc = main(args)
        number.append(acc)

    print("All Test_acc :",number)
