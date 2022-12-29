# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dgcnn_trainer.py
@Time    :   2022/8/16 0:32:45
@Author  :   Wang Xianglong
"""
import copy
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TL_BACKEND'] = 'paddle'

import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.datasets import ModelNet40
import numpy as np
from gammagl.models import DGCNNModel
from gammagl.loader import DataLoader
from tensorlayerx.model import TrainOneStep, WithLoss
import sklearn.metrics as metrics
# tlx.set_device("GPU", 2)

# if tlx.BACKEND == 'torch':  # when the backend is torch and you want to use GPU
#     try:
#         tlx.set_device(device='GPU', id=0)
#     except:
#         print("GPU is not available")


class CalLoss(WithLoss):
    def __init__(self, net):
        super(CalLoss, self).__init__(backbone=net, loss_fn=None)

    def forward(self, x, gold, smoothing=True):
        pred = self.backbone_network(x)
        gold = tlx.reshape(tlx.convert_to_tensor(gold), (-1,))

        if smoothing:
            eps = 0.2
            n_class = pred.shape[1]

            one_hot = nn.OneHot(depth=n_class)(gold)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            c = tlx.reduce_max(pred, 1, keepdims=True)
            log_prb = (pred - c) - tlx.log(tlx.reduce_sum(tlx.exp(pred - c), 1, keepdims=True))

            loss = -tlx.reduce_mean(tlx.reduce_sum(one_hot * log_prb, axis=1))
        else:
            loss = tlx.losses.softmax_cross_entropy_with_logits(pred, gold)

        return loss


def pre_transform(data_list):
    for data in data_list:
        x = tlx.random_uniform((3,), minval=2. / 3., maxval=3. / 2.)
        y = tlx.random_uniform((3,), minval=-0.2, maxval=0.2)
        data.x = tlx.add(tlx.multiply(data.x, x), y)
    return data_list


def main(args):
    dataset = ModelNet40(args.dataset_path, split='train', num_points=args.num_points, pre_transform=pre_transform)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(args.dataset_path, split='test', num_points=args.num_points),
                             batch_size=args.batch_size, shuffle=True, drop_last=False)
    print(train_loader)

    tlx.set_device("CPU" if args.no_cuda else "GPU")
    print(tlx.get_device())

    net = DGCNNModel(args.in_channel, args.k, args.emb_dims, args.num_points, args.dropout, args.out_channel)
    try:
        net.load_weights(args.best_model_path + "DGCNN.npz", format='npz_dict')
    except:
        print("no this file!")
    print(str(net))

    if args.use_sgd is True:
        scheduler = tlx.optimizers.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.epochs, eta_min=args.lr)
        print("Use SGD")
        opt = tlx.optimizers.SGD(lr=scheduler, momentum=args.momentum, weight_decay=1e-4)
    else:
        scheduler = tlx.optimizers.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.epochs, eta_min=args.lr)
        print("Use Adam")
        opt = tlx.optimizers.Adam(lr=scheduler, weight_decay=1e-4)

    train_weights = net.trainable_weights
    loss_func = CalLoss(net)
    train_one_step = TrainOneStep(loss_func, opt, train_weights)

    best_test_acc = 0

    for epoch in range(args.epochs):
        train_loss = 0.
        count = 0.
        scheduler.step()
        net.set_train()
        for data in train_loader:
            batch_size = data.num_graphs
            loss = train_one_step(data.x, data.y)
            count += batch_size
            train_loss += loss.item() * batch_size
            print(f'loss: {train_loss}')
        print(f'Train {epoch}, loss: {train_loss / count}')

        test_loss = 0.
        count = 0.
        net.set_eval()
        test_pred = []
        test_true = []
        for data in test_loader:
            batch_size = data.num_graphs
            logits = net(data.x)
            loss = loss_func(data.x, data.y)
            preds = tlx.argmax(logits, axis=1)
            count += batch_size
            if tlx.BACKEND == 'tensorflow':
                test_loss += loss.numpy().item() * batch_size
            elif tlx.BACKEND == 'torch':
                test_loss += loss.cpu().detach().numpy().item() * batch_size
            elif tlx.BACKEND == 'mindspore':
                test_loss += loss.asnumpy().item() * batch_size
            elif tlx.BACKEND == 'paddle':
                test_loss += loss.numpy().item() * batch_size

            test_true.append(np.array(data.y))
            test_pred.append(tlx.convert_to_numpy(preds)),
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        print(f'Test {epoch}, loss: {test_loss / count}, test acc: {test_acc}, test avg acc: {avg_per_class_acc}')

        if test_acc >= best_test_acc:
            ofile = open('best.txt', 'a+')
            print(f'Test {epoch}, loss: {test_loss / count}, test acc: {test_acc}, test avg acc: {avg_per_class_acc}', file=ofile)
            ofile.close()
            best_test_acc = test_acc
            print('save weights...')
            net.save_weights(args.best_model_path + "DGCNN.npz", format='npz_dict')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch')
    parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if '
                                                                              'using sgd)')
    parser.add_argument('--in_channel', type=int, default=3, help='input feature dimension')
    parser.add_argument('--out_channel', type=int, default=40, help='output feature dimension')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    args = parser.parse_args()

    main(args)
