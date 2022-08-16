# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pna_trainer.py
@Time    :   2022/5/26 20:07:55
@Author  :   huang le
"""
import copy
import os
import os.path as osp
import argparse
import sys



sys.path.insert(0, osp.abspath('../../'))  # adds path2gammagl to execute in command line.

import numpy
import torch

os.environ['TL_BACKEND'] = 'torch'
import tensorlayerx as tlx
from tensorlayerx import convert_to_tensor, convert_to_numpy
# from tensorlayerx.optimizers.lr import ReduceOnPlateau
from tensorlayerx.optimizers.lr import ReduceOnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gammagl.datasets import ZINC
from gammagl.loader import DataLoader
from gammagl.models import PNAModel
from gammagl.models import Net
from gammagl.utils import set_device
from gammagl.utils.degree import degree
from tensorlayerx.model import TrainOneStep, WithLoss
from tensorlayerx.losses import absolute_difference_error

set_device()

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = self._loss_fn(tlx.squeeze(logits, axis=1), label)
        return loss


def evaluate(model, loader):
    model.set_eval()
    for i in range(0, 4):
        model.batch_norms[i].is_train = False
    total_loss = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = absolute_difference_error(tlx.squeeze(x=out, axis=1), data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def main(args):
    # load datasets
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ZINC')
    train_dataset = ZINC(path, subset=True, split='train')
    val_dataset = ZINC(path, subset=True, split='val')
    test_dataset = ZINC(path, subset=True, split='test')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=tlx.int64)
        max_degree = max(max_degree, int(tlx.reduce_max(d)))
    #
    # Compute the in-degree histogram tensor
    deg = tlx.zeros((max_degree + 1,), dtype=tlx.int64)

    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=tlx.int64)
        deg_i = numpy.bincount(convert_to_numpy(d), minlength=len(deg))
        deg += convert_to_tensor(deg_i)

    model = PNAModel(deg)
    #检查模型初始化
    # init = copy.deepcopy(model.state_dict())
    # all = model.all_weights
    # # model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    # state = torch.load('./init')
    # new_state = state
    # # 测试embedding有没有问题---0809_17---没有问题---mae为0.1889
    # new_state['node_emb.embeddings'] = new_state.pop('node_emb.weight')
    # new_state['edge_emb.embeddings'] = new_state.pop('edge_emb.weight')
    # 测试batchnorm有没有问题---0809_17---mae增大为0.2198
    # for i in range(0, 4):
        # new_state[f'batch_norms_test.{i}.module.weight'] = copy.deepcopy(new_state[f'batch_norms.{i}.module.weight'])
        # new_state[f'batch_norms_test.{i}.module.bias'] = copy.deepcopy(new_state[f'batch_norms.{i}.module.bias'])
        # new_state[f'batch_norms_test.{i}.module.running_mean'] = copy.deepcopy(new_state[f'batch_norms.{i}.module.running_mean'])
        # new_state[f'batch_norms_test.{i}.module.running_var'] = copy.deepcopy(new_state[f'batch_norms.{i}.module.running_var'])

        # new_state[f'batch_norms.{i}.gamma'] = new_state.pop(f'batch_norms.{i}.module.weight')
        # new_state[f'batch_norms.{i}.beta'] = new_state.pop(f'batch_norms.{i}.module.bias')
        # new_state[f'batch_norms.{i}.moving_mean'] = new_state.pop(f'batch_norms.{i}.module.running_mean')
        # new_state[f'batch_norms.{i}.moving_var'] = new_state.pop(f'batch_norms.{i}.module.running_var')
        # new_state.pop(f'batch_norms.{i}.module.num_batches_tracked')
    # 测试mlp有没有问题---0809_17---mae维持0.1889
    # for i in [0, 1, 2]:
    #     new_state[f'mlp.{i}.weights'] = new_state.pop(f'mlp.{i*2}.weight').t()
    #     new_state[f'mlp.{i}.biases'] = new_state.pop(f'mlp.{i*2}.bias')
    #
    # model.load_state_dict(new_state)
    # torch.save(new_state, './pna_state')
    # state = torch.load('./pna_state')
    # model.load_state_dict(state)
    # best_test_mae = evaluate(model, test_loader)
    # state_dict = copy.deepcopy(model.state_dict())
    # new_state = copy.deepcopy(state)
    # new_state['node_emb.embeddings'] = new_state.pop('node_emb.weight')
    # new_state['edge_emb.embeddings'] = new_state.pop('edge_emb.weight')
    # for i in range(0, 4):
    #     new_state[f'batch_norms.{i}.gamma'] = new_state.pop(f'batch_norms.{i}.module.weight')
    #     new_state[f'batch_norms.{i}.beta'] = new_state.pop(f'batch_norms.{i}.module.bias')
    #     new_state[f'batch_norms.{i}.moving_mean'] = new_state.pop(f'batch_norms.{i}.module.running_mean')
    #     new_state[f'batch_norms.{i}.moving_var'] = new_state.pop(f'batch_norms.{i}.module.running_var')
    #     new_state.pop(f'batch_norms.{i}.module.num_batches_tracked')
    #
    # for i in [0, 1, 2]:
    #     new_state[f'mlp.{i}.weights'] = new_state.pop(f'mlp.{i*2}.weight').t()
    #     new_state[f'mlp.{i}.biases'] = new_state.pop(f'mlp.{i*2}.bias')
    # model.load_state_dict(new_state)
    # model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    # test = evaluate(model, test_loader)
    # for key in new_state.keys():
    #     if key not in state_dict.keys():
    #         print(key)

    # val = evaluate(model, val_loader)
    # test = evaluate(model, test_loader)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = ReduceOnPlateau(0.001, mode='min', factor=0.5, patience=10,
    #                             min_lr=0.00001)
    optimizer = tlx.optimizers.Adam(lr=0.001)

    train_weights = model.trainable_weights
    loss_func = SemiSpvzLoss(model, absolute_difference_error)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    # model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    # model.load_weights(args.best_model_path + "best.npz", format='npz_dict')
    # best_val_mae = evaluate(model, test_loader)
    best_val_mae = 1
    for epoch in range(1, 501):
        # init = copy.deepcopy(model.state_dict())
        model.set_train()
        for i in range(0, 4):
            model.batch_norms[i].is_train = True
        all_loss = 0
        for data in train_loader:
            loss = train_one_step(data, data.y)
            all_loss += loss.item() * data.num_graphs
        all_loss = all_loss / len(train_loader.dataset)
        loss = all_loss
        # loss = train(train_loader)
        # init = copy.deepcopy(model.state_dict())
        val_mae = evaluate(model, val_loader)
        # val_mae = 0
        # 测试验证模式是否会修改模型参数
        # i = 1
        # for name in model.state_dict():
        #     if not tlx.ops.equal(init[name], model.state_dict()[name]):
        #         print(i, name)
        #         i += 1
        test_mae = evaluate(model, test_loader)
        # test_mae = 0
        # scheduler.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
        # 可训练参数
        # i = 1
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(i, name)
        #         i += 1
        # 验证参数是否得到训练
        # i = 1
        # for name in model.state_dict():
        #     if not tlx.ops.equal(init[name], model.state_dict()[name]):
        #         print(i, name)
        #         i += 1

        # scheduler.step(val_mae)
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val_mae: {val_mae:.4f}, test_mae: {test_mae:.4f},lr: ', )
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val_mae: {val_mae:.4f}, test_mae: {test_mae:.4f},lr: ',
        #       optimizer.param_groups[0]['lr'])
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val_mae: {val_mae:.4f}, test_mae: {test_mae:.4f},')
              # f'lr: ', scheduler.last_lr)
        with open('./log.txt', mode='a', encoding='utf-8') as file:
            file.write(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, val_mae: {val_mae:.4f}, test_mae: {test_mae:.4f}\n')

    model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')
    test_mae = evaluate(model, test_loader)
    print("Test mae:  {:.4f}".format(test_mae))
    with open('./test_accuracy', mode='a+', encoding='utf-8') as file:
        file.write(f'PyTorch ZINC {test_mae}\n')


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=300, help="number of epoch")
    parser.add_argument("--l2_coef", type=float, default=1e-3, help="l2 loss coeficient")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")

    args = parser.parse_args()
    main(args)
