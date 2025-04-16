import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import tensorlayerx as tlx
import numpy as np
from sklearn.metrics import f1_score
from partition import partition_patch
from gammagl.datasets import Planetoid
from gammagl.models.cobformer import CoBFormer
from tensorlayerx.model import TrainOneStep, WithLoss


def eval_f1(pred, label, num_classes):
    pred = tlx.convert_to_numpy(pred)
    label = tlx.convert_to_numpy(label)
    micro = f1_score(label, pred, average='micro')
    macro = f1_score(label, pred, average='macro')
    return micro, macro


class CoLoss(WithLoss):
    def __init__(self, model, loss_fn):
        super(CoLoss, self).__init__(backbone=model, loss_fn=loss_fn)
        self.alpha = model.alpha
        self.tau = model.tau

    def forward(self, data, label):
        pred1, pred2 = self.backbone_network(data['x'], data['patch'], data['edge_index'], edge_weight=data['edge_weight'], num_nodes=data['num_nodes'])
        l1 = tlx.losses.softmax_cross_entropy_with_logits(pred1[data['train_mask']], label[data['train_mask']])
        l2 = tlx.losses.softmax_cross_entropy_with_logits(pred2[data['train_mask']], label[data['train_mask']])
        
        pred1_scaled = pred1 * self.tau
        pred2_scaled = pred2 * self.tau
        
        l3 = tlx.losses.softmax_cross_entropy_with_logits(pred1_scaled[~data['train_mask']], tlx.nn.Softmax()(pred2_scaled)[~data['train_mask']])
        l4 = tlx.losses.softmax_cross_entropy_with_logits(pred2_scaled[~data['train_mask']], tlx.nn.Softmax()(pred1_scaled)[~data['train_mask']])
        
        return self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)


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
    # set_device(5)
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset)
    graph = dataset[0]

    graph.train_mask = tlx.convert_to_numpy(graph.train_mask)
    graph.val_mask = tlx.convert_to_numpy(graph.val_mask)
    graph.test_mask = tlx.convert_to_numpy(graph.test_mask)
    # Pad a dimension with value 0 at the end of each mask (1D array) using np.pad(mask, (0, 1), mode='constant')
    graph.train_mask = np.pad(graph.train_mask, (0, 1), mode='constant')
    graph.val_mask = np.pad(graph.val_mask, (0, 1), mode='constant')
    graph.test_mask = np.pad(graph.test_mask, (0, 1), mode='constant')

    patch = partition_patch(graph, args.n_patch)
        
    # try:
    #     patch_copy = tlx.cast(patch, dtype=tlx.int64)
    # except:
    #     patch_copy = tlx.convert_to_tensor(patch, dtype=tlx.int64)
    
    # Convert label to one-hot encoding and cast to float type
    label = tlx.nn.OneHot(dataset.num_classes)(graph.y)
    label = tlx.cast(label, dtype=tlx.float32)

    model = CoBFormer(graph.num_nodes, dataset.num_node_features, args.num_hidden, dataset.num_classes, layers=args.num_layers,
                      gcn_layers=args.gcn_layers, n_head=args.n_head, alpha=args.alpha, tau=args.tau, use_patch_attn=args.use_patch_attn)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    train_weights = model.trainable_weights
    
    loss_func = CoLoss(model, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    
    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        "edge_weight": None,
        "train_mask": graph.train_mask,
        "test_mask": graph.test_mask,
        "val_mask": graph.val_mask,
        "num_nodes": graph.num_nodes,
        'train': graph.train_mask,
        'valid': graph.val_mask,
        'test': graph.test_mask,
        'patch': patch
    }
    
    # best_val_acc = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        loss = train_one_step(data, label)
        model.set_eval()
        
        pred1, pred2 = model(data['x'], data['patch'], data['edge_index'], edge_weight=data['edge_weight'], num_nodes=data['num_nodes'])
        
        y = data['y']
            
        num_classes = int(tlx.reduce_max(y) + 1)
        
        y1_ = tlx.argmax(pred1, axis=1)
            
        micro_val1, macro_val1 = eval_f1(y1_[data['valid']], y[data['valid']], num_classes)
        # micro_test1, macro_test1 = eval_f1(y1_[data['test']], y[data['test']], num_classes)

        y2_ = tlx.argmax(pred2, axis=1)
        if len(y2_.shape) > 1:
            y2_ = y2_.view(-1)
            
        micro_val2, macro_val2 = eval_f1(y2_[data['valid']], y[data['valid']], num_classes)
        # micro_test2, macro_test2 = eval_f1(y2_[data['test']], y[data['test']], num_classes)
        
        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(loss.item())\
              + "  GCN  micro_val acc: {:.4f}".format(micro_val1)\
              + "  GCN  macro_val acc: {:.4f}".format(macro_val1)\
              + "  COB  micro_val acc: {:.4f}".format(micro_val2)\
              + "  COB  macro_val acc: {:.4f}".format(macro_val2))



if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--gcn_wd', type=float, default=5e-4)
    parser.add_argument('--num_hidden', type=int, default=64, help='Number of hidden units')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_epoch', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--use_patch_attn', action='store_true', help='transformer use patch attention')
    parser.add_argument('--show_details', type=bool, default=True)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--n_patch', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--train_prop', type=float, default=.6)
    parser.add_argument('--valid_prop', type=float, default=.2)
    parser.add_argument('--alpha', type=float, default=.8)
    parser.add_argument('--tau', type=float, default=.3)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu)
    else:
        tlx.set_device("CPU")

    main(args)
