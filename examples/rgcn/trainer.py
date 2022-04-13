# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/20 13:47
# @Author  : clear
# @FileName: trainer.py.py
import os
os.environ['TL_BACKEND'] = 'paddle' # set your backend here, default `tensorflow`
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import RGCN
from gammagl.utils.loop import add_self_loops
from tensorlayerx.model import TrainOneStep, WithLoss

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self._backbone(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
        if tlx.BACKEND == 'mindspore':
            idx = tlx.convert_to_tensor([i for i, v in enumerate(data['train_mask']) if v], dtype=tlx.int64)
            train_logits = tlx.gather(logits,idx)
            train_label = tlx.gather(label,idx)
        else:
            train_logits = logits[data['train_mask']]
            train_label = label[data['train_mask']]
        loss = self._loss_fn(train_logits, train_label)
        return loss
