import os
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

from gammagl.layers.conv.gcn_unifews import (
    ThrInPrune, LayerNumLogger, rewind,
    reset_weight_, reset_bias_,
    gcn_norm, add_remaining_self_loops,
    layer_dict_gcn,
    flops_modules_dict_gcn
)
from gammagl.layers.conv.gat_unifews import layer_dict_gat,flops_modules_dict_gat
from gammagl.layers.conv.gcn2_unifews import layer_dict_gcn2,flops_modules_dict_gcn2
from gammagl.layers.conv.gsage_unifews import layer_dict_gsage,flops_modules_dict_gsage

layer_dict = {
    **layer_dict_gcn,
    **layer_dict_gat,
    **layer_dict_gcn2,
    **layer_dict_gsage
}

flops_modules_dict = {
    **flops_modules_dict_gcn,
    **flops_modules_dict_gat,
    **flops_modules_dict_gcn2,
    **flops_modules_dict_gsage
}

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

class GNNThr(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass, thr_a=0.0, thr_w=0.0, dropout=0.0, layer='gcn', **kwargs):
        super().__init__()
        self.apply_thr = '_' in layer
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.use_bn = True
        self.kwargs = kwargs

        Conv = layer_dict[layer]
        layer_base = layer.split('_')[0]
        for k, v in kwargs_default[layer_base].items():
            self.kwargs.setdefault(k, v)

        self.is_gat = layer_base.startswith('gat')
        if self.is_gat:
            self.heads = self.kwargs['heads']
            self.concat = self.kwargs['concat']
            self.feat_dim = self.heads * nhidden if self.concat else nhidden
        else:
            self.feat_dim = nhidden

        thr_a = [thr_a] * nlayer if not isinstance(thr_a, list) else thr_a
        thr_w = [thr_w] * nlayer if not isinstance(thr_w, list) else thr_w

        self.depth_inv = self.kwargs.pop('depth_inv', False)
        self.normalize_adj = self.kwargs.pop('normalize', False)
        self.add_self_loops = self.kwargs.pop('add_self_loops', False)
        self.cached = self.kwargs.pop('cached', False)

        self.mid_kwargs = self.kwargs.copy()
        for k in ['improved', 'rnorm', 'diag']:
            self.mid_kwargs.pop(k, None)

        self.final_kwargs = self.mid_kwargs.copy()
        for k in ['heads', 'concat']:
            self.final_kwargs.pop(k, None)
        if self.is_gat:
            self.final_kwargs['heads'] = 1
            self.final_kwargs['concat'] = False

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(Conv(nfeat, nhidden, thr_a=thr_a[0], thr_w=thr_w[0], **self.mid_kwargs))
        self.norms.append(nn.BatchNorm1d(num_features=self.feat_dim, momentum=0.1))

        for i in range(1, nlayer-1):
            self.convs.append(Conv(nhidden, nhidden, thr_a=thr_a[i], thr_w=thr_w[i], **self.mid_kwargs))
            self.norms.append(nn.BatchNorm1d(num_features=self.feat_dim, momentum=0.1))

        final_in = self.feat_dim if self.is_gat else nhidden
        self.convs.append(Conv(final_in, nclass, thr_a=thr_a[-1], thr_w=thr_w[-1], **self.final_kwargs))

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
            elif 'BatchNorm' in type(norm).__name__:
                reset_bn_(norm)

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
        raw_edge = edge_idx

        if self.apply_thr:
            for i, conv in enumerate(self.convs[:-1]):
                x, _ = conv(x, raw_edge, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[-1](x, raw_edge, node_lock=node_lock, verbose=verbose)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, raw_edge)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[-1](x, raw_edge)
        return x

    def get_repre(self, x, edge_idx, layer=None, node_lock=tlx.convert_to_tensor([]), verbose=False):
        layer = layer or len(self.convs)-1
        edge_idx = self._process_graph(x, edge_idx)
        raw_edge = edge_idx
        if self.apply_thr:
            for i, conv in enumerate(self.convs[:layer]):
                x, _ = conv(x, raw_edge, node_lock=node_lock, verbose=verbose)
                if self.use_bn: x = self.norms[i](x)
                x = self.act(x); x = self.dropout(x)
            x, _ = self.convs[layer](x, raw_edge, node_lock=node_lock, verbose=verbose)
        else:
            for i, conv in enumerate(self.convs[:layer]):
                x = conv(x, raw_edge)
                if self.use_bn: x = self.norms[i](x)
                x = self.act(x); x = self.dropout(x)
            x = self.convs[layer](x, raw_edge)
        return x

    def set_scheme(self, scheme_a, scheme_w):
        self.apply(lambda m: set_attr(m, 'scheme_a', scheme_a))
        self.apply(lambda m: set_attr(m, 'scheme_w', scheme_w))

    def remove(self):
        for conv in self.convs:
            if hasattr(conv, 'prune_lst'):
                for m in conv.prune_lst:
                    if prune.is_pruned(m): prune.remove(m, 'weights')

    def get_numel(self):
        numel_a = sum(c.logger_a.numel_after for c in self.convs)
        numel_w = sum(c.logger_w.numel_after for c in self.convs)
        return numel_a/1e3, numel_w/1e3

    @classmethod
    def batch_counter_hook(cls, module, inp, out):
        if not hasattr(module, '__batch_counter__'): module.__batch_counter__ = 0
        module.__batch_counter__ += 1

class GNNLPThr(GNNThr):
    def __init__(self, nlayer, nfeat, nhidden, nclass, thr_a=0.0, thr_w=0.0, dropout=0.0, layer='gcn', **kwargs):
        super().__init__(nlayer, nfeat, nhidden, nhidden, thr_a, thr_w, dropout, layer, **kwargs)
        self.lin_out = nn.ModuleList([
            nn.Linear(nhidden, nhidden),
            nn.Linear(nhidden, nhidden),
            nn.Linear(nhidden, 1),
        ])

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lin_out:
            reset_weight_(lin.weights, lin.in_features, initializer='kaiming_uniform')
            reset_bias_(lin.biases, lin.in_features, initializer='uniform')

    def decode(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lin_out[:-1]:
            x = lin(x)
            x = self.act(x)
            x = self.dropout(x)
        return self.lin_out[-1](x)