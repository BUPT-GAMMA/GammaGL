import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import sys
import argparse
import gammagl.transforms as T
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from tensorlayerx.model import TrainOneStep, WithLoss
import argparse
import numpy as np
import warnings
import sys
import argparse
from gammagl.models import Hid_net
from gammagl.utils import mask_to_index
warnings.filterwarnings('ignore')


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, graph):
        logits = self.backbone_network(data['x'], data['edge_index'],num_nodes=data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_mask'])
        train_y = tlx.gather(data['y'], data['train_mask'])
        
        loss = self._loss_fn(train_logits,train_y)

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
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_weight = tlx.ones(shape=(graph.edge_index.shape[1], 1))

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        "edge_weight": edge_weight,
        "train_mask": train_idx,
        "test_mask": test_idx,
        "val_mask": val_idx,
        "num_nodes": graph.num_nodes,
    }

    model = Hid_net(in_feats=dataset.num_features,
                    hidden_dim=args.hidden_dim,
                    n_classes=dataset.num_classes,
                    num_layers=args.num_layers,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    add_bias=args.add_bias,
                    normalize=args.normalize,
                    drop_rate=args.drop_rate,
                    sigma1=args.sigma1,
                    sigma2=args.sigma2,
                    name="Hid_Net")
        
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(model, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    best_val_acc=0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = train_one_step(data, graph.y)

        model.set_eval()
        logits = model(data['x'], data['edge_index'], data['edge_weight'], num_nodes=data['num_nodes'])

        val_preds=tlx.gather(logits, data['val_mask'])
        val_y=tlx.gather(data['y'], data['val_mask'])
        val_acc=calculate_acc(val_preds,val_y,metrics)

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')
        
        print("Epoch [{:0>3d}] ".format(epoch+1)\
              + "  train loss: {:.4f}".format(train_loss.item())\
              + "  val acc: {:.4f}".format(val_acc))
            
    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    model.set_eval()
    logits = model(data['x'], data['edge_index'], data['edge_weight'], num_nodes=data['num_nodes'])
    test_preds = tlx.gather(logits, data['test_mask'])
    test_y = tlx.gather(data['y'], data['test_mask'])
    test_acc = calculate_acc(test_preds, test_y, metrics)    
    print("Test acc:  {:.4f}".format(test_acc))

if __name__=='__main__':
    # parameters settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument('--n_epoch', type=int, default=150, help='the num of epoch')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size')
    parser.add_argument('--head1', type=int, default=1, help='gat head1')
    parser.add_argument('--head2', type=int, default=1, help='gat head2')
    parser.add_argument('--drop_rate', type=float, default=0.55, help='dropout rate')
    parser.add_argument('--drop', type=str, default='False', help='whether to dropout or not')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--dataset_path', type=str, default='./data', help='path to save dataset')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=10, help='num_layers')
    parser.add_argument('--alpha', type=float, default=0.1, help='tolerance to stop EM algorithm')
    parser.add_argument('--beta', type=float, default=0.9, help='tolerance to stop EM algorithm')
    parser.add_argument('--gamma', type=float, default=0.3, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma1', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma2', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--gpu',  default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--add_bias', type=bool, default=True, help='if tune')
    parser.add_argument('--normalize', type=bool, default=True, help='if tune')
    args = parser.parse_args()

    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu) 
    else:
        tlx.set_device("CPU")

    main(args)
