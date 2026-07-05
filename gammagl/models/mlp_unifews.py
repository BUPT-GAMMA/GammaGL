import os
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.gglspeedup.prunes_gamma import prune, rewind, ThrInPrune, ThrProdPrune

def reset_bn_(bn_module):
    if bn_module is None:
        return

    if hasattr(bn_module, 'weight') and bn_module.weight is not None:
        new_gamma = tlx.convert_to_tensor(
            tlx.initializers.Ones()(bn_module.weight.shape), dtype=bn_module.weight.dtype)
        tlx.assign(bn_module.weight, new_gamma)

    if hasattr(bn_module, 'bias') and bn_module.bias is not None:
        new_beta = tlx.convert_to_tensor(
            tlx.initializers.Zeros()(bn_module.bias.shape), dtype=bn_module.bias.dtype)
        tlx.assign(bn_module.bias, new_beta)

    if hasattr(bn_module, 'moving_mean') and bn_module.moving_mean is not None:
        new_mean = tlx.convert_to_tensor(
            tlx.initializers.Zeros()(bn_module.moving_mean.shape), dtype=bn_module.moving_mean.dtype)
        tlx.assign(bn_module.moving_mean, new_mean)

    if hasattr(bn_module, 'moving_var') and bn_module.moving_var is not None:
        new_var = tlx.convert_to_tensor(
            tlx.initializers.Ones()(bn_module.moving_var.shape), dtype=bn_module.moving_var.dtype)
        tlx.assign(bn_module.moving_var, new_var)

from gammagl.layers.conv.gat_unifews import (
    ThrInPrune, LayerNumLogger, rewind,
    reset_weight_, reset_bias_,
    add_remaining_self_loops
)
from gammagl.layers.conv.gcn_unifews import gcn_norm
from .gnn_unifews import layer_dict

kwargs_default = {
    'gcn': {
        'cached': False,
        'add_self_loops': False,
        'improved': False,
        'normalize': False,
        'rnorm': 0.5,
        'diag': 1.0,
        'depth_inv': False,
    },
    'gin': {
        'eps': 0.0,
        'train_eps': False,
        'rnorm': None,
        'diag': 1.0,
        'depth_inv': False,
    },
    'gat': {
        'heads': 1,
        'concat': True,
        'add_self_loops': False,
        'rnorm': None,
        'diag': 1.0,
        'depth_inv': False,
        'depth': 2,
    },
    'gcn2': {
        'alpha': 0.1,
        'beta': 0.5,
        'cached': False,
        'add_self_loops': False,
        'normalize': False,
        'rnorm': 0.5,
        'diag': 1.0,
        'depth_inv': True,
    },
    'gsage': {
        'aggr': 'mean',
        'improved': False,
        'normalize': False,
        'root_weight': True,
        'project': False,
        'bias': True,
        'rnorm': 0.5,
        'diag': 1.0,
        'depth_inv': False,
    },
}

def state2module(model, name):
    parts = name.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module

def set_attr(module, key, value):
    if hasattr(module, key):
        setattr(module, key, value)


class MLP_unifews(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass, dropout, thr_w=0.0, layer='sgc'):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.threshold_w = thr_w
        self.scheme_w = 'full'

        self.fcs = nn.ModuleList()
        if nlayer == 1:
            self.fcs.append(nn.Linear(nfeat, nclass))
        else:
            self.fcs.append(nn.Linear(nfeat, nhidden))
            for _ in range(nlayer-2):
                self.fcs.append(nn.Linear(nhidden, nhidden))
            self.fcs.append(nn.Linear(nhidden, nclass))

        for fc in self.fcs:
            fc.logger_w = LayerNumLogger(layer)
            object.__setattr__(fc, 'act', lambda x: x)
            fc.bias = fc._bias if hasattr(fc, '_bias') else None

    def reset_parameters(self):
        for lin in self.fcs:
            reset_weight_(lin.weights, lin.in_features, initializer='kaiming_uniform')
            reset_bias_(lin.biases, lin.in_features, initializer='uniform')

    def apply_prune(self, lin, x):
        log = lin.logger_w
        log.numel_before = 1
        log.numel_after = 1

    def forward(self, x, edge_idx=None, *args, **kwargs):
        for i, fc in enumerate(self.fcs[:-1]):
            self.apply_prune(fc, x)
            x = fc(x)
            x = self.act(x)
            x = self.dropout(x)
        self.apply_prune(self.fcs[-1], x)
        return self.fcs[-1](x)

    def set_scheme(self, scheme_a, scheme_w):
        self.scheme_w = scheme_w

    def get_numel(self):
        return 0, sum(fc.logger_w.numel_after for fc in self.fcs)/1e3