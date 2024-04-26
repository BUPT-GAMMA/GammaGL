import os
import sys
sys.path.insert(0, '/home/zgy/interns/Yzk/GammaGL')
import argparse
#import gammagl.transforms as T
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from tensorlayerx.model import TrainOneStep, WithLoss
import argparse
import numpy as np
import warnings
import sys
import argparse
from gammagl.datasets import WikipediaNetwork
from gammagl.models import hid_net
from gammagl.utils import mask_to_index
import torch
print(torch.__version__)
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
def get_split(y, nclass, seed=0):

    percls_trn = int(round(0.6 * len(y) / nclass))
    val_lb = int(round(0.2 * len(y)))

    indices = []
    for i in range(nclass):
        h = tlx.convert_to_numpy((y == i))
        res = np.nonzero(h)
        res = np.array(res).reshape(-1)
        index = tlx.convert_to_tensor(res)
        n = tlx.get_tensor_shape(index)[0]
        index = tlx.gather(index, np.random.permutation(n))
        indices.append(index)

    train_index = tlx.concat(values=[i[:percls_trn] for i in indices], axis=0)
    rest_index = tlx.concat(values=[i[percls_trn:] for i in indices], axis=0)
    m = tlx.get_tensor_shape(rest_index)[0]
    index2 = tlx.convert_to_tensor(np.random.permutation(m))
    index2 = tlx.cast(index2, dtype=tlx.int64)

    rest_index = tlx.gather(rest_index, index2)
    valid_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]

    return train_index, valid_index, test_index
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
def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--times', type=int, default=3, help='config times')
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--repeat', type=int, default=5, help='repeat time')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=128, help='hidden size')
    parser.add_argument('--head1', type=int, default=1, help='gat head1')
    parser.add_argument('--head2', type=int, default=1, help='gat head2')
    parser.add_argument('--dropout', type=float, default=0.55, help='dropout rate')
    parser.add_argument('--drop', type=str, default='False', help='whether to dropout or not')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'sbm'])
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
    parser.add_argument('--add_self_loops', type=bool, default=True, help='if tune')
    parser.add_argument('--normalize', type=bool, default=True, help='if tune')
    args = parser.parse_args()

    
    absolute_path = os.path.abspath(args.dataset_path)
    print(absolute_path)

    print(args)
    if args.gpu >= 0:
            
            tlx.set_device("GPU", args.gpu) 
            
    else:
            tlx.set_device("CPU")


    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(args.dataset_path, args.dataset)
        graph = dataset[0]
        train_idx = mask_to_index(graph.train_mask)
        test_idx = mask_to_index(graph.test_mask)
        val_idx = mask_to_index(graph.val_mask)
    elif args.dataset in ['chameleon', 'squirrel']:
        # dataset = WikipediaNetwork(root=f'./data', name=args.dataset,transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(root=f'./data', name=args.dataset)
        i = args.split
        graph = dataset[0]
        train_idx, val_idx, test_idx = get_split(y=graph.y, nclass=dataset.num_classes)
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
    accs = []

    for seed in range(args.repeat):
        tlx.set_seed(seed)
                # 将模型移动到指定的设备
        if args.gpu >= 0:
            tlx.set_device("GPU", args.gpu) 
        else:
            tlx.set_device("CPU")
        # 创建模型实例
        model = hid_net(in_feats=dataset.num_features,
                        n_hidden=args.hidden,
                        n_classes=dataset.num_classes,
                        k=args.k,
                        alpha=args.alpha,
                        beta=args.beta,
                        gamma=args.gamma,
                        bias=args.bias,
                        normalize=args.normalize,
                        add_self_loops=args.add_self_loops,
                        drop=args.drop,
                        dropout=args.dropout,
                        sigma1=args.sigma1,
                        sigma2=args.sigma2)
        #
        

        # TODO
        
        optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
        metrics = tlx.metrics.Accuracy()
        train_weights = model.trainable_weights
        criterion = tlx.losses.softmax_cross_entropy_with_logits
        loss_func = SemiSpvzLoss(model, criterion)

        val_accs = []
        test_accs = []


        
        train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

        for epoch in range(150):
            
            val_accs = []
            test_accs = []
            
            
            model.set_train()
            train_loss = train_one_step(data, graph.y)

    
            model.set_eval()
            out= model(data['x'], data['edge_index'],num_nodes=data['num_nodes'])

            
            train_preds=tlx.gather(out,data['train_mask'])
            train_y=tlx.gather(data['y'],data['train_mask'])
            train_acc=calculate_acc(train_preds,train_y,metrics)
            

            val_preds=tlx.gather(out,data['val_mask'])
            val_y=tlx.gather(data['y'],data['val_mask'])
            val_loss = criterion(val_preds, val_y)
            val_acc=calculate_acc(val_preds,val_y,metrics)
            val_accs.append(val_acc)


            test_preds=tlx.gather(out,data['test_mask'])
            test_y=tlx.gather(data['y'],data['test_mask'])
            test_acc=calculate_acc(test_preds,test_y,metrics)
            test_accs.append(test_acc)

            # print(f"Epoch {epoch}: "
            #       f"Train loss = {train_loss:.2f}, "
            #       f"train acc = {train_acc * 100:.2f}, "
            #       f"val loss = {val_loss:.2f}, "
            #       f"val acc = {val_acc * 100:.2f} "             
            #       f"test acc = {test_acc * 100:.2f} "
            #       )
            

            print(f"Epoch {epoch}: "
      f"Train loss = {train_loss.item():.2f}, "
      f"train acc = {train_acc.item() * 100:.2f}, "
      f"val loss = {val_loss.asnumpy().item():.2f}, "
      f"val acc = {val_acc.item() * 100:.2f} "
      f"test acc = {test_acc.item() * 100:.2f} ")


            
        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])


    print("\t[Classification] acc_mean {:.4f} acc_std {:.4f}"
        .format(
            np.mean(accs),
            np.std(accs),
            )
        )
if __name__=='__main__':
    main()
    os._exit(0)




