import os
os.environ['TL_BACKEND'] = 'torch'
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
from gammagl.datasets import WikipediaNetwork
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
    
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(args.dataset_path, args.dataset)
        graph = dataset[0]
        train_idx = mask_to_index(graph.train_mask)
        test_idx = mask_to_index(graph.test_mask)
        val_idx = mask_to_index(graph.val_mask)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        "edge_weight": graph.edge_weight,
        "train_mask": train_idx,
        "test_mask": test_idx,
        "val_mask": val_idx,
        "num_nodes": graph.num_nodes,
    }

    model = Hid_net(in_feats=dataset.num_features,
                    n_hidden=args.hidden,
                    n_classes=dataset.num_classes,
                    k=args.k,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    bias=args.bias,
                    normalize=args.normalize,
                    drop=args.drop,
                    dropout=args.dropout,
                    sigma1=args.sigma1,
                    sigma2=args.sigma2)
        
    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights
    criterion = tlx.losses.softmax_cross_entropy_with_logits
    loss_func = SemiSpvzLoss(model, criterion)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    for epoch in range(args.n_epoch):
            
        best_val_acc=0
            
        model.set_train()
        train_loss = train_one_step(data, graph.y)

        model.set_eval()
        out= model(data['x'], data['edge_index'],data['edge_weight'],num_nodes=data['num_nodes'])

        train_preds=tlx.gather(out,data['train_mask'])
        train_y=tlx.gather(data['y'],data['train_mask'])
        train_acc=calculate_acc(train_preds,train_y,metrics)
            
        val_preds=tlx.gather(out,data['val_mask'])
        val_y=tlx.gather(data['y'],data['val_mask'])
        val_loss = criterion(val_preds, val_y)
        val_acc=calculate_acc(val_preds,val_y,metrics)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path+model.name+".npz", format='npz_dict')

        print(f"Epoch {epoch}: "
                f"Train loss = {train_loss:.2f}, "
                f"train acc = {train_acc * 100:.2f}, "
                f"val loss = {val_loss:.2f}, "
                f"val acc = {val_acc * 100:.2f} "             
                )
            
    model.load_weights(args.best_model_path+model.name+".npz", format='npz_dict')
    model.set_eval()
    out= model(data['x'], data['edge_index'],data['edge_weight'],num_nodes=data['num_nodes'])
    test_preds=tlx.gather(out,data['test_mask'])
    test_y=tlx.gather(data['y'],data['test_mask'])
    test_acc=calculate_acc(test_preds,test_y,metrics)    
    print("Test acc:  {:.4f}".format(test_acc))

if __name__=='__main__':
# Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument('--n_epoch', type=int, default=150, help='the num of epoch')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=128, help='hidden size')
    parser.add_argument('--head1', type=int, default=1, help='gat head1')
    parser.add_argument('--head2', type=int, default=1, help='gat head2')
    parser.add_argument('--dropout', type=float, default=0.55, help='dropout rate')
    parser.add_argument('--drop', type=str, default='False', help='whether to dropout or not')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('--dataset_path', type=str, default='./data', help='path to save dataset')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--k', type=int, default=10, help='k')
    parser.add_argument('--alpha', type=float, default=0.1, help='tolerance to stop EM algorithm')
    parser.add_argument('--beta', type=float, default=0.9, help='tolerance to stop EM algorithm')
    parser.add_argument('--gamma', type=float, default=0.3, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma1', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--sigma2', type=float, default=0.5, help='tolerance to stop EM algorithm')
    parser.add_argument('--gpu',  default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--bias', type=bool, default=False, help='if tune')
    parser.add_argument('--normalize', type=bool, default=True, help='if tune')
    args = parser.parse_args()

    print(args)
    if args.gpu >= 0:
        tlx.set_device("GPU", args.gpu) 
    else:
        tlx.set_device("CPU")

    main(args)
    os._exit(0)




