import os
os.environ['TL_BACKEND'] = 'torch'

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.gglspeedup.prunes_gamma import prune, rewind, ThrInPrune, ThrProdPrune

def reset_bn_(bn_module):
    
    if bn_module is None:
        return

    if hasattr(bn_module, 'weight') and bn_module.weight is not None:
        new_gamma = tlx.initializers.Ones()(bn_module.weight.shape)
        bn_module.weight.data = new_gamma

    if hasattr(bn_module, 'bias') and bn_module.bias is not None:
        new_beta = tlx.initializers.Zeros()(bn_module.bias.shape)
        bn_module.bias.data = new_beta

    if hasattr(bn_module, 'moving_mean') and bn_module.moving_mean is not None:
        new_mean = tlx.initializers.Zeros()(bn_module.moving_mean.shape)
        bn_module.moving_mean.data = new_mean

    if hasattr(bn_module, 'moving_var') and bn_module.moving_var is not None:
        new_var = tlx.initializers.Ones()(bn_module.moving_var.shape)
        bn_module.moving_var.data = new_var
    
from gammagl.layers.conv.gat_thr import (
    ThrInPrune, LayerNumLogger, rewind, 
    reset_weight_, reset_bias_, 
    gcn_norm, add_remaining_self_loops
)

from .gnn_model import layer_dict

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


class SandwitchGCNII(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass, alpha, beta, 
                 thr_a=0.0, thr_w=0.0, dropout=0.0, variant=False, layer='gcn2', **kwargs):
        super().__init__()
        self.nfeat = nfeat
        self.nhidden = nhidden
        self.nclass = nclass
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_bn = True
        
        self.lin_in = nn.Linear(in_features=nfeat, out_features=nhidden)
        self.lin_out = nn.Linear(in_features=nhidden, out_features=nclass)
        
        for lin in [self.lin_in, self.lin_out]:
            object.__setattr__(lin, 'act', lambda x: x)

        Conv = layer_dict[layer] 
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        thr_a = [thr_a] * nlayer if not isinstance(thr_a, list) else thr_a
        thr_w = [thr_w] * nlayer if not isinstance(thr_w, list) else thr_w

        for i in range(nlayer):
            self.convs.append(Conv(
                in_channels=nhidden, 
                out_channels=nhidden, 
                alpha=alpha, 
                beta=beta, 
                variant=variant,
                thr_a=thr_a[i], 
                thr_w=thr_w[i]
            ))
            self.norms.append(nn.BatchNorm1d(num_features=nhidden, momentum=0.1))

    def reset_parameters(self):
        if not hasattr(self.lin_in, 'weights'): self.lin_in.build((None, self.nfeat))
        reset_weight_(self.lin_in.weights, self.nfeat)
        reset_bias_(self.lin_in.biases, self.nfeat)
        
        if not hasattr(self.lin_out, 'weights'): self.lin_out.build((None, self.nhidden))
        reset_weight_(self.lin_out.weights, self.nhidden)
        reset_bias_(self.lin_out.biases, self.nhidden)
            
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'): conv.reset_parameters()
        for norm in self.norms:
            reset_bn_(norm)

    def forward(self, x, edge_idx, **kwargs):
       
        x = self.lin_in(x)
        x = x_0 = self.act(x)
        x = self.dropout(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, x_0, edge_idx)
            if self.use_bn: x = self.norms[i](x)
            x = self.act(x)
            x = self.dropout(x)
        return self.lin_out(x)

    def set_scheme(self, scheme_a, scheme_w):
        self.apply(lambda m: set_attr(m, 'scheme_a', scheme_a))
        self.apply(lambda m: set_attr(m, 'scheme_w', scheme_w))

    def get_numel(self):
        numel_a = sum(c.logger_a.numel_after for c in self.convs if hasattr(c, 'logger_a'))
        numel_w = sum(c.logger_w.numel_after for c in self.convs if hasattr(c, 'logger_w'))
        return numel_a/1e3, numel_w/1e3

class SandwitchThr(SandwitchGCNII):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 thr_a=0.0, thr_w=0.0, dropout: float = 0.0, layer: str = 'gcn2_thr',
                 **kwargs):
        
        alpha = kwargs.get('alpha', 0.1)
        beta = kwargs.get('beta', 0.5)
        variant = kwargs.get('variant', False)
        
        
        super().__init__(nlayer=nlayer, nfeat=nfeat, nhidden=nhidden, nclass=nclass, 
                         alpha=alpha, beta=beta, thr_a=thr_a, thr_w=thr_w, 
                         dropout=dropout, variant=variant, layer=layer, **kwargs)
        
        self.apply_thr = '_' in layer
        self.normalize_adj = kwargs.get('normalize', False)
        self.add_self_loops = kwargs.get('add_self_loops', False)

    def _process_graph(self, x, edge_idx):
        if not (self.normalize_adj or self.add_self_loops):
            return edge_idx
        edge_index, edge_weight = edge_idx if isinstance(edge_idx, (tuple, list)) else (edge_idx, None)
        num_nodes = x.shape[0]
        if self.normalize_adj:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=self.add_self_loops, dtype=x.dtype)
        elif self.add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        return (edge_index, edge_weight)

    def forward(self, x, edge_idx, node_lock=tlx.convert_to_tensor([]), verbose=False):
        edge_idx = self._process_graph(x, edge_idx)
        
        x = self.lin_in(x)
        x = x_0 = self.act(x)
        x = self.dropout(x)

        if self.apply_thr:
            for i, conv in enumerate(self.convs):

                x, edge_idx = conv(x, x_0, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn: x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
        else:
            for i, conv in enumerate(self.convs):
                x = conv(x, x_0, edge_idx)
                if self.use_bn: x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)

        x = self.lin_out(x)
        return x

    def remove(self):
        for conv in self.convs:
            if hasattr(conv, 'prune_lst'):
                for m in conv.prune_lst:
                    if prune.is_pruned(m): prune.remove(m, 'weights')

    def get_repre(self, x, edge_idx, layer=None, node_lock=tlx.convert_to_tensor([]), verbose=False):
        layer = layer or len(self.convs)
        edge_idx = self._process_graph(x, edge_idx)
        
        x = self.lin_in(x)
        x = x_0 = self.act(x)
        x = self.dropout(x)

        for i, conv in enumerate(self.convs[:layer]):
            if self.apply_thr:
                x, edge_idx = conv(x, x_0, edge_idx, node_lock=node_lock, verbose=verbose)
            else:
                x = conv(x, x_0, edge_idx)
            if self.use_bn: x = self.norms[i](x)
            x = self.act(x)
            x = self.dropout(x)
        return x

