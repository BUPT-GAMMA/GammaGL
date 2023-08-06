# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   cagcn_trainer.py
@Time    :   2023/07/01 10:04:55
@Author  :   Yang Zhijie
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['TL_BACKEND'] = 'torch'
import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models.gcn import GCNModel
from gammagl.models.cagcn import CAGCNModel
from gammagl.utils import mask_to_index

if tlx.BACKEND == 'torch':  # when the backend is torch and you want to use GPU
    try:
        tlx.set_device(device='GPU', id=4)
    except:
        print("GPU is not available")


def load_dataset(args):
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    # get info
    features = graph.x
    labels = graph.y
    train_idx = list(mask_to_index(graph.train_mask))
    val_idx = list(mask_to_index(graph.val_mask))
    test_idx = list(mask_to_index(graph.test_mask))
    # train_idx = mask_to_index(graph.train_mask)
    # val_idx = mask_to_index(graph.val_mask)
    # test_idx = mask_to_index(graph.test_mask)

    # process train_mask
    for _ in range(2):
        if args.labelrate != 20:
            args.labelrate -= 20
            nclass = int(max(labels)) + 1
            start = max(val_idx) + 1
            end = start + args.labelrate * nclass
            for i in range(start, end):
                train_idx.append(i)
                # tlx.concat((train_idx, tlx.convert_to_tensor([i], dtype=tlx.int64)), axis=-1)

    return {
        "features": features,
        "labels": labels,
        "edge_index": graph.edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes
    }


def intra_distance_loss(output, target):
    output = tlx.nn.Softmax(axis=1)(output)
    pred_max_idx = tlx.ops.argmax(output, axis=1)

    output = tlx.ops.topk(output, 2, dim=1)
    pred, sub_pred = output[0][:, 0], output[0][:, 1]

    loss = 0
    for item in zip(pred_max_idx, target, pred, sub_pred):
        loss += (1 - item[2] + item[3] if item[0] == item[1] else item[2] - item[3])
    loss /= len(target)
    return loss


def nll_and_lambda_cal_loss_fn(Lambda):
    def func(output, target):
        nll = tlx.losses.softmax_cross_entropy_with_logits(output, target)
        cal = intra_distance_loss(output, target)
        return nll + Lambda * cal

    return func


class BaseModelLoss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(BaseModelLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'], data['edge_index'], None, data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


class CalModelLoss(tlx.model.WithLoss):
    def __init__(self, net, loss_fn):
        super(CalModelLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['edge_index'], None, data['num_nodes'], data['x'], data['edge_index'], None,
                                       data['num_nodes'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def train(epoch_id, Lambda, model, metrics, optimizer, data, train_weights, sign=False):
    if not sign:
        loss_fn = tlx.losses.softmax_cross_entropy_with_logits
        net_with_loss = BaseModelLoss(model, loss_fn)
    else:
        loss_fn = nll_and_lambda_cal_loss_fn(Lambda)
        net_with_loss = CalModelLoss(model, loss_fn)
    train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, train_weights)

    model.set_train()
    train_loss = train_one_step(data, data['y'])
    model.set_eval()
    if not sign:
        logits = model(data['x'], data['edge_index'], None, data['num_nodes'])
    else:
        logits = model(data['edge_index'], None, data['num_nodes'], data['x'], data['edge_index'], None,
                       data['num_nodes'])
    # val
    val_logits = tlx.gather(logits, data['val_idx'])
    val_y = tlx.gather(data['y'], data['val_idx'])
    val_loss = tlx.losses.softmax_cross_entropy_with_logits(val_logits, val_y)
    metrics.update(val_logits, val_y)
    val_acc = metrics.result()
    metrics.reset()

    if tlx.BACKEND == "paddle":
        train_loss = train_loss[0]
        val_loss = tlx.convert_to_numpy(val_loss)[0]

    print(f'epoch[{epoch_id:04d}]',
          f'train_loss {train_loss:.4f}',
          f'val_loss {val_loss:.4f}',
          f'val_acc {val_acc:.4f}')
    return val_loss


def main(args):
    data = load_dataset(args)
    nclass = int(max(data['labels'])) + 1

    if tlx.BACKEND == "paddle":
        temp = list()
        for i in range(len(data['train_idx'])):
            temp.append(tlx.convert_to_tensor(data['train_idx'][i]))
        data['train_idx'] = temp

    pesudo_data = {
        "x": data['features'],
        "y": tlx.ops.convert_to_tensor(data['labels']),
        "edge_index": data['edge_index'],
        "train_idx": data['train_idx'],
        "test_idx": data['test_idx'],
        "val_idx": data['val_idx'],
        "num_nodes": data['num_nodes'],
    }

    if tlx.BACKEND == 'torch':
        device = pesudo_data['x'].device
    acc_test_times_list = list()

    if not os.path.exists(args.best_model_path):
        os.makedirs(args.best_model_path)
    model_a_scaling = '%s/%s-%s-%s-%d-w-s.npz' % (
        args.best_model_path, args.model, args.dataset, args.threshold, args.labelrate)
    model_b_scaling = '%s/%s-%s-%s-%d-w_o-s.npz' % (
        args.best_model_path, args.model, args.dataset, args.threshold, args.labelrate)

    for times in range(0, args.stage):
        # phase_1
        # Base Model
        base_model = GCNModel(pesudo_data['x'].shape[1], args.hidden, nclass, args.dropout)
        base_optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.weight_decay)
        metrics = tlx.metrics.Accuracy()
        if tlx.BACKEND == 'torch':
            base_model.to(device)

        # Train base model normally
        # Save the best base model weights
        best_val_loss = 100
        bad_counter = 0
        for epoch in range(args.epochs):
            val_loss = train(epoch, args.Lambda, base_model, metrics, base_optimizer, pesudo_data,
                             base_model.trainable_weights)
            if val_loss < best_val_loss:
                base_model.save_weights(model_b_scaling, format='npz_dict')
                best_val_loss = val_loss
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

        # phase 2
        # Load the best base model weights
        base_model.load_weights(model_b_scaling, format='npz_dict')
        cal_model = CAGCNModel(base_model, nclass, 1, args.dropout)
        cal_optimizer = tlx.optimizers.Adam(lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
        if tlx.BACKEND == 'torch':
            cal_model.base_model.to(pesudo_data['x'].device)
            cal_model.cal_model.to(pesudo_data['x'].device)
            cal_model.to(pesudo_data['x'].device)

        # Train calibration model by training set or validation set (only the last time used)
        # Save the best calibration model weights
        best_val_loss = 100
        bad_counter = 0
        epochs = args.epoch_for_st if times != args.stage - 1 else args.epochs
        temp_data = pesudo_data
        if times == args.stage - 1:
            temp_data['train_idx'] = pesudo_data['val_idx']

        for epoch in range(epochs):
            val_loss = train(epoch, args.Lambda, cal_model, metrics, cal_optimizer, temp_data,
                             cal_model.cal_model.trainable_weights, True)
            if times != args.stage - 1:
                if epoch == epochs - 1:
                    cal_model.save_weights(model_a_scaling, format='npz_dict')
                continue
            if val_loss < best_val_loss:
                cal_model.save_weights(model_a_scaling, format='npz_dict')
                best_val_loss = val_loss
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break

        # phase 3
        # Load the best calibration model weights
        cal_model.load_weights(model_a_scaling, format='npz_dict')
        if tlx.BACKEND == 'torch':
            cal_model.base_model.to(device)
            cal_model.cal_model.to(device)
            cal_model.to(device)
        # add predictions satisfying (ground truth == prediction && confidence > threshold) to pesudo labels
        cal_model.set_eval()
        output = cal_model(pesudo_data['edge_index'], None, pesudo_data['num_nodes'], pesudo_data['x'],
                           pesudo_data['edge_index'], None, pesudo_data['num_nodes'])
        output = tlx.nn.Softmax()(output)
        pred_label = tlx.ops.argmax(output, axis=-1)
        confidence = tlx.ops.reduce_max(output, axis=-1)
        temp_y = tlx.convert_to_numpy(pesudo_data['y'])
        for i, conf in enumerate(confidence):
            if conf > args.threshold and (i not in pesudo_data['train_idx']) and (
                    i not in pesudo_data['test_idx']) and (i not in pesudo_data['val_idx']):
                pesudo_data['train_idx'].append(i)
                # tlx.concat((pesudo_data['train_idx'], tlx.convert_to_tensor([i], dtype=tlx.int64)), axis=-1)
                temp_y[i] = pred_label[i]
        pesudo_data['y'] = tlx.convert_to_tensor(temp_y)

        # phase 4
        # Testing
        test_model = cal_model
        test_model.load_weights(model_a_scaling, format='npz_dict')
        if tlx.BACKEND == 'torch':
            test_model.to(device)
        test_model.set_eval()
        logits = test_model(data['edge_index'], None, data['num_nodes'], data['features'], data['edge_index'], None,
                            data['num_nodes'])
        test_logits = tlx.gather(logits, data['test_idx'])
        test_labels = tlx.gather(data['labels'], data['test_idx'])
        test_loss = tlx.losses.softmax_cross_entropy_with_logits(test_logits, test_labels)
        metrics.update(test_logits, test_labels)
        test_acc = metrics.result()
        metrics.reset()

        print(f"Test set results with CaGCN:",
              f"loss = {test_loss:.4f}",
              f"accuracy = {test_acc:.4f}")

        acc_test_times_list.append(test_acc)

    print('acc_test:', acc_test_times_list)
    return acc_test_times_list[-1]


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default="Cora", help='dataset for training')
    parser.add_argument('--stage', type=int, default=1, help='times of retraining')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--epoch_for_st', type=int, default=200,
                        help='Number of epochs to calibration for self-training')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--lr_for_cal', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--l2_for_cal', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters) for calibration.')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--labelrate', type=int, default=60)
    parser.add_argument('--n_bins', type=int, default=20)
    parser.add_argument('--Lambda', type=float, default=0.5, help='the weight for ranking loss')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset, default in peer dir.")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()

    # main(args)

    import numpy as np

    number = []
    for i in range(3):
        acc = main(args)

        number.append(acc)

    print("实验结果：")
    print(np.mean(number))
    print(np.std(number))
    print(number)
