# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import FusedGATModel
from gammagl.utils import add_self_loops, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.utils import sort_edge_index 
from gammagl.ops.sparse import ind2ptr
import time, torch, GPUtil


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data['x'], data['edge_index'], data['num_nodes'], **data['csc_csr_info'])
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
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index, _ = add_self_loops(graph.edge_index, n_loops=args.self_loops, num_nodes=graph.num_nodes)

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    _edge_index = edge_index
    num_edges = tlx.ops.get_tensor_shape(_edge_index)[1]
    _edge_index = sort_edge_index(_edge_index)
    row_ptr = ind2ptr(tlx.convert_to_numpy(_edge_index[0, :]), graph.num_nodes)
    col_ind = _edge_index[1, :]
    permute = tlx.ops.arange(0, num_edges)
    _edge_index, permute = sort_edge_index(_edge_index, permute, sort_by_row=False)
    row_ind = _edge_index[0, :]
    col_ptr = ind2ptr(tlx.convert_to_numpy(_edge_index[1, :]), graph.num_nodes)

    row_ptr = tlx.convert_to_tensor(row_ptr, tlx.int32)
    col_ind = tlx.convert_to_tensor(col_ind, tlx.int32)
    col_ptr = tlx.convert_to_tensor(col_ptr, tlx.int32)
    row_ind = tlx.convert_to_tensor(row_ind, tlx.int32)
    permute = tlx.convert_to_tensor(permute, tlx.int32)


    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
        "csc_csr_info": {
            'row_ptr': row_ptr,
            'col_ind': col_ind,
            'permute': permute,
            'row_ind': row_ind,
            'col_ptr': col_ptr
        }
    }
    
    net = FusedGATModel(feature_dim=dataset.num_node_features,
                        hidden_dim=args.hidden_dim,
                        num_class=dataset.num_classes,
                        heads=args.heads,
                        drop_rate=args.drop_rate,
                        num_layers=args.num_layers,
                        name="FusedGAT",
                        )

    loss = tlx.losses.softmax_cross_entropy_with_logits
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    
    net.set_train()
    max_memory = 0
    for epoch in range(10):
        train_loss = train_one_step(data, graph.y)
        GPUs = GPUtil.getGPUs()
        max_memory = max(GPUs[args.gpu].memoryUsed, max_memory)
    
    net.set_train()
    torch.cuda.synchronize()
    start_time = time.time()
    for epoch in range(args.n_epoch):
        train_loss = train_one_step(data, graph.y)
        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   train loss: {:.4f}".format(train_loss.item()))
        
    torch.cuda.synchronize()
    end_time = time.time()
    train_time = (end_time - start_time) / args.n_epoch
    
    net.set_eval()
    torch.cuda.synchronize()
    start_time = time.time()
    for epoch in range(args.n_epoch):
        logits = net(data['x'], data['edge_index'], data['num_nodes'], **data['csc_csr_info'])

    torch.cuda.synchronize()
    end_time = time.time()
    infer_time = (end_time - start_time) / args.n_epoch

    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))
    print('max memory: {}MB'.format(max_memory))
    print("train time: {}s".format(train_time))
    print("infer time: {}s".format(infer_time))



if __name__ == "__main__":
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.005, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=8, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.4, help="drop_rate")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument("--heads", type=int, default=32, help="number of heads for stablization")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--num_layers", type=int, default=3, help="number of gat layers")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
