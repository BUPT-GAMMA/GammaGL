import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR

import argparse
import tensorlayerx as tlx

from gammagl.datasets import Planetoid
import gammagl.transforms as T
from gammagl.models import GCNModel as GCN, GraphSAGE_Full_Model as GraphSAGE
from gammagl.models.dcgc import EdgeWeight, TemperatureScaling
from gammagl.utils import mask_to_index
from tensorlayerx.model import WithLoss, TrainOneStep
import numpy as np

def ECELoss(logits, labels, n_bins=15):
    
    confidences = tlx.softmax(logits, axis=1)
    confidences = tlx.reduce_max(confidences, axis=1)
    predictions = tlx.argmax(logits, axis=1)
    accuracies = tlx.cast(predictions == labels, dtype=tlx.float64)
    
    bin_boundaries = tlx.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = tlx.zeros(shape=[1], dtype=tlx.float64)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        in_bin = tlx.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = tlx.reduce_mean(tlx.cast(in_bin, dtype=tlx.float64))
        if prop_in_bin > 0:
            indices = tlx.arange(0, len(in_bin))
            
            in_bin_indices = indices[in_bin]
            
            if len(in_bin_indices) > 0:
                accuracy_in_bin = tlx.reduce_mean(tlx.gather(accuracies, in_bin_indices))
                avg_confidence_in_bin = tlx.reduce_mean(tlx.gather(confidences, in_bin_indices))
                ece += tlx.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, idx_train, edge_weight=None):
        logits = self.backbone_network(data.x, data.edge_index, edge_weight, data.num_nodes)
        train_logits = tlx.gather(logits, idx_train)
        train_y = tlx.gather(data.y, idx_train)
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

def train(model, optimizer, data, idx_train, idx_val, idx_test, edge_weight=None, base=True):
    model.set_train()
    net_with_loss = SemiSpvzLoss(model, loss_fn)
    
    trainable_weights = model.trainable_weights
    train_one_step = TrainOneStep(net_with_loss, optimizer, trainable_weights)

    if edge_weight is not None and hasattr(edge_weight, 'clone'):
        edge_weight = edge_weight.clone().detach()

    loss = train_one_step(data, idx_train, edge_weight)
    return loss


def test(model, data, idx_train, idx_val, idx_test, edge_weight=None):
    model.set_eval()

    output = model(data.x, data.edge_index, edge_weight, data.num_nodes)
    pred = tlx.argmax(output, axis=-1)
    metrics = tlx.metrics.Accuracy()
    
    train_logits = tlx.gather(output, idx_train)
    train_y = tlx.gather(data.y, idx_train)
    train_acc = calculate_acc(train_logits, train_y, metrics)
    
    val_logits = tlx.gather(output, idx_val)
    val_y = tlx.gather(data.y, idx_val)
    val_acc = calculate_acc(val_logits, val_y, metrics)
    
    test_logits = tlx.gather(output, idx_test)
    test_y = tlx.gather(data.y, idx_test)
    test_acc = calculate_acc(test_logits, test_y, metrics)
    
    loss_train = loss_fn(tlx.gather(output, idx_train), tlx.gather(data.y, idx_train))
    loss_val = loss_fn(tlx.gather(output, idx_val), tlx.gather(data.y, idx_val))
    loss_test = loss_fn(tlx.gather(output, idx_test), tlx.gather(data.y, idx_test))
    
    ece_train = ECELoss(tlx.gather(output, idx_train), tlx.gather(data.y, idx_train), args.n_bins)
    ece_val = ECELoss(tlx.gather(output, idx_val), tlx.gather(data.y, idx_val), args.n_bins)
    ece_test = ECELoss(tlx.gather(output, idx_test), tlx.gather(data.y, idx_test), args.n_bins)
    
    return train_acc, val_acc, test_acc, loss_train, loss_val, loss_test, ece_train, ece_val, ece_test


def main_base(model, optimizer, data, idx_train, idx_val, idx_test, edge_weight=None, base=True):
    best = 0
    for epoch in range(args.epochs):
        train(model, optimizer, data, idx_train, idx_val, idx_test, edge_weight, base=base)
        acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, ece_train, ece_val, ece_test = \
            test(model, data, idx_train, idx_val, idx_test, edge_weight)
        
        if acc_val > best:
            model.save_weights('base_model.npz', format='npz_dict')
            best = acc_val
    
    model.load_weights('base_model.npz', format='npz_dict')
    model.set_eval()
    
    output = model(data.x, data.edge_index, edge_weight, data.num_nodes)
    
    metrics = tlx.metrics.Accuracy()
    
    train_logits = tlx.gather(output, idx_train)
    train_y = tlx.gather(data.y, idx_train)
    acc_train = calculate_acc(train_logits, train_y, metrics)
    
    val_logits = tlx.gather(output, idx_val)
    val_y = tlx.gather(data.y, idx_val)
    acc_val = calculate_acc(val_logits, val_y, metrics)
    
    test_logits = tlx.gather(output, idx_test)
    test_y = tlx.gather(data.y, idx_test)
    acc_test = calculate_acc(test_logits, test_y, metrics)
    
    ece_train = ECELoss(tlx.gather(output, idx_train), tlx.gather(data.y, idx_train), args.n_bins)
    ece_val = ECELoss(tlx.gather(output, idx_val), tlx.gather(data.y, idx_val), args.n_bins)
    ece_test = ECELoss(tlx.gather(output, idx_test), tlx.gather(data.y, idx_test), args.n_bins)
    
    print(f'acc_train: {acc_train:.4f}',
          f'acc_val: {acc_val:.4f}',
          f'acc_test: {acc_test:.4f}',
          f'ece_train: {ece_train:.4f}',
          f'ece_val: {ece_val:.4f}',
          f'ece_test: {ece_test:.4f}',
          )
    
    return acc_train, acc_val, acc_test, ece_train, ece_val, ece_test


def main_cali(model, optimizer, data, idx_train, idx_val, idx_test, edge_weight=None, base=False):
    best = 100
    bad_counter = 0
    
    # Ensure edge_weight is completely detached from the original computation graph
    if edge_weight is not None and hasattr(edge_weight, 'clone'):
        edge_weight = edge_weight.clone().detach()
    
    for epoch in range(args.cal_epochs):
        train(model, optimizer, data, idx_train, idx_val, idx_test, edge_weight, base=base)
        acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, ece_train, ece_val, ece_test = \
            test(model, data, idx_train, idx_val, idx_test, edge_weight)
        if loss_val < best:
            model.save_weights('calibration.npz', format='npz_dict')
            best = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        
        if bad_counter == args.patience:
            break
    
    model.load_weights('calibration.npz', format='npz_dict')
    model.set_eval()
    
    output = model(data.x, data.edge_index, edge_weight, data.num_nodes)
    
    metrics = tlx.metrics.Accuracy()
    
    train_logits = tlx.gather(output, idx_train)
    train_y = tlx.gather(data.y, idx_train)
    acc_train = calculate_acc(train_logits, train_y, metrics)
    
    val_logits = tlx.gather(output, idx_val)
    val_y = tlx.gather(data.y, idx_val)
    acc_val = calculate_acc(val_logits, val_y, metrics)
    
    test_logits = tlx.gather(output, idx_test)
    test_y = tlx.gather(data.y, idx_test)
    acc_test = calculate_acc(test_logits, test_y, metrics)
    
    ece_train = ECELoss(tlx.gather(output, idx_train), tlx.gather(data.y, idx_train), args.n_bins)
    ece_val = ECELoss(tlx.gather(output, idx_val), tlx.gather(data.y, idx_val), args.n_bins)
    ece_test = ECELoss(tlx.gather(output, idx_test), tlx.gather(data.y, idx_test), args.n_bins)
    
    print(f'acc_train: {acc_train:.4f}',
          f'acc_val: {acc_val:.4f}',
          f'acc_test: {acc_test:.4f}',
          f'ece_train: {ece_train:.4f}',
          f'ece_val: {ece_val:.4f}',
          f'ece_test: {ece_test:.4f}',
          )
    
    return acc_train, acc_val, acc_test, ece_train, ece_val, ece_test

def main(args):
    global loss_fn
    loss_fn = tlx.losses.softmax_cross_entropy_with_logits
    
    if str.lower(args.dataset) not in ['cora','pubmed','citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    nfeat = data.x.shape[1]
    nclass = data.y.max().item() + 1

    idx_train = mask_to_index(data.train_mask)
    idx_val = mask_to_index(data.val_mask)
    idx_test = mask_to_index(data.test_mask)
        
    old_acc_train, old_acc_val, old_acc_test = [], [], []
    old_ece_train, old_ece_val, old_ece_test = [], [], []
    
    new_acc_test0, new_ece_test0 = [], []
    new_acc_test1, new_ece_test1 = [], []
    
    for i in range(args.num1):
        print('---------------------------------')
        
        base_model = GCN(nfeat, args.hidden, nclass, drop_rate=args.dropout)
        optimizer_base = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
        
        acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
            main_base(base_model, optimizer_base, data, idx_train, idx_val, idx_test)
        
        old_acc_train.append(acc_train * 100)
        old_acc_val.append(acc_val * 100)
        old_acc_test.append(acc_test * 100)
        old_ece_train.append(ece_train * 100)
        old_ece_val.append(ece_val * 100)
        old_ece_test.append(ece_test * 100)
        
        # load base model
        base_model.load_weights('base_model.npz', format='npz_dict')
        
        for j in range(args.num2):
            print('---------------------------------')

            ew = EdgeWeight(nclass, base_model, args.dropout)
            optimizer_ew = tlx.optimizers.Adam(lr=0.2, weight_decay=args.l2_for_cal)
            
            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                main_cali(ew, optimizer_ew, data, idx_train, idx_val, idx_test)
            
            ew.set_eval()
            output = ew(data.x, data.edge_index, None, data.num_nodes)
            pred_orig = tlx.softmax(output, axis=1)
            
            pred = tlx.exp(tlx.ops.multiply(args.beta, pred_orig))
            pred = tlx.ops.divide(pred, tlx.reduce_sum(pred, axis=1, keepdims=True))
            
            col, row = data.edge_index
            diff = tlx.ops.subtract(tlx.gather(pred, col), tlx.gather(pred, row))
            squared_diff = tlx.square(diff)
            sum_squared_diff = tlx.reduce_sum(squared_diff, axis=1)
            coefficient = tlx.sqrt(sum_squared_diff)
            denominator = tlx.ops.add(coefficient, args.alpha)
            coefficient = tlx.ops.divide(tlx.ones_like(denominator), denominator)
            
            edge_weight = ew.get_weight(data.x, data.edge_index, data.num_nodes)
            edge_weight = tlx.convert_to_tensor(edge_weight.clone().detach())
            
            edge_weight_reshaped = tlx.reshape(edge_weight, [-1])
            edge_weight_weighted = tlx.ops.multiply(edge_weight_reshaped, coefficient)
            edge_weight_final = tlx.reshape(edge_weight_weighted, [data.num_edges, 1])
            
            
            ts = TemperatureScaling(base_model)
            optimizer_ts = tlx.optimizers.Adam(lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
            
            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                main_cali(ts, optimizer_ts, data, idx_train, idx_val, idx_test)
            new_acc_test0.append(acc_test * 100)
            new_ece_test0.append(ece_test * 100)
            
            edge_weight_final_copy = edge_weight_final.clone().detach()
            
            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                main_cali(ts, optimizer_ts, data, idx_train, idx_val, idx_test, edge_weight=edge_weight_final_copy)
            new_acc_test1.append(acc_test * 100)
            new_ece_test1.append(ece_test * 100)
    
    # calculate mean and std
    def mean_std(values):
        return np.mean(values), np.std(values)
    
    old_acc_mean, old_acc_std = mean_std(old_acc_test)
    old_ece_mean, old_ece_std = mean_std(old_ece_test)
    
    print(f'old_acc_test: {old_acc_mean:.2f}±{old_acc_std:.2f}',
          f'old_ece_test: {old_ece_mean:.2f}±{old_ece_std:.2f}')
    
    # print results
    for i, (acc_list, ece_list, name) in enumerate([
        (new_acc_test0, new_ece_test0, "TS"),
        (new_acc_test1, new_ece_test1, "TS+EW")
    ]):
        acc_mean, acc_std = mean_std(acc_list)
        ece_mean, ece_std = mean_std(ece_list)
        print(f'{name}: acc={acc_mean:.2f}±{acc_std:.2f}, ece={ece_mean:.2f}±{ece_std:.2f}')

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Pubmed",
                        help='dataset for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--cal_epochs', type=int, default=1500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--lr_for_cal', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--l2_for_cal', type=float, default=0,
                        help='Weight decay (L2 loss on parameters) for calibration.')
    parser.add_argument('--n_bins', type=int, default=20)
    parser.add_argument('--patience', type=int, default=700)
    parser.add_argument('--num1', type=int, default=1)
    parser.add_argument('--num2', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
        
    main(args)
