import os
from math import log
import numpy as np
from typing import Optional, Tuple, Union, Any


import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.nn import Linear
import gammagl as ggl

Tensor = Any
Adj = Tensor
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, OptTensor]

from gammagl.layers.conv import (
   SAGEConv, MessagePassing
)

from gammagl.gglspeedup.prunes_gamma import ThrInPrune, rewind,prune, _tensor_numel
from gammagl.utils.logger_unifews import LayerNumLogger
from .gcn_unifews import identity_n_norm, gcn_norm, softmax, leaky_relu, scatter, maybe_num_nodes, reset_weight_, reset_bias_,ConvThr,add_remaining_self_loops,pow_with_pinv,normalize,norm

class SAGEConvRaw(SAGEConv):
    def __init__(self, in_channels, out_channels, rnorm=None, diag=1., depth_inv=False, *args, **kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        kwargs.pop('root_weight', None)
        kwargs.pop('project', None)
        kwargs.pop('bias',None)
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters()


    def reset_parameters(self):
        reset_weight_(self.fc_neigh.weights, self.in_feat, initializer='kaiming_uniform')
        if self.aggr != 'gcn':
            reset_weight_(self.fc_self.weights, self.in_feat, initializer='kaiming_uniform')

    def forward(self, x, edge_tuple: PairTensor, **kwargs):
        (edge_index, edge_weight) = edge_tuple
        self.logger_a.numel_after = edge_index.shape[1]
        total_w = _tensor_numel(self.fc_neigh.weights)
        if self.aggr != 'gcn':
            total_w += _tensor_numel(self.fc_self.weights)
        self.logger_w.numel_after = total_w
        return super().forward(x, edge_index)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, edge_tuple = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_tuple[0].shape[1]
        module.__flops__ += int(f_in * f_out * n) * (2 if module.aggr != 'gcn' else 1)
        module.__flops__ += (f_out if module.add_bias else 0) * n
        module.__flops__ += f_in * m

class SAGEConvThr(ConvThr, SAGEConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)

        self.prune_lst = [self.fc_neigh]
        if self.aggr != 'gcn':
            self.prune_lst.append(self.fc_self)
        self.register_forward_hook(self.prune_on_msg)

    def prune_on_msg(self, module, inputs, output):
        msg_tensor = output[0] if isinstance(output, (list, tuple)) else output
        num_edges = msg_tensor.shape[0]

        if self.scheme_a in ['pruneall', 'pruneinc']:

            norm_feat_msg = tlx.sqrt(tlx.reduce_sum(tlx.square(msg_tensor), axis=1))
            norm_all_msg = tlx.reduce_sum(tlx.abs(norm_feat_msg)) / num_edges
            mask_prune = norm_feat_msg < (self.threshold_a * 0.1 * norm_all_msg)

            final_mask_to_keep = tlx.logical_not(mask_prune)

            msg_tensor = tlx.where(tlx.expand_dims(final_mask_to_keep, 1), msg_tensor, tlx.zeros_like(msg_tensor))
            edge_indices = tlx.arange(0, num_edges, dtype=tlx.int64)
            self.idx_keep = tlx.cast(edge_indices[final_mask_to_keep], tlx.int64).squeeze()

        elif self.scheme_a == 'keep':
            keep_mask = tlx.zeros((num_edges,), dtype=tlx.bool)
            if self.idx_keep is not None:
                indices_to_save = self.idx_keep[self.idx_keep < num_edges]
                keep_mask = tlx.scatter_update(keep_mask, indices_to_save, tlx.ones_like(indices_to_save, dtype=tlx.bool))

            msg_tensor = tlx.where(tlx.expand_dims(keep_mask, 1), msg_tensor, tlx.zeros_like(msg_tensor))

        return (msg_tensor,) + output[1:] if isinstance(output, (list, tuple)) else msg_tensor

    def forward(self, x, edge_tuple: PairTensor, node_lock: OptTensor = None, verbose: bool = False):
        (edge_index, edge_weight) = edge_tuple
        x = (x, x) if tlx.is_tensor(x) else x
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.fc_neigh):
                    rewind(self.fc_neigh, 'weights')
            else:
                if prune.is_pruned(self.fc_neigh):
                    prune.remove(self.fc_neigh, 'weights')
            if self.aggr != 'gcn':
                if self.scheme_w == 'pruneall':
                    if prune.is_pruned(self.fc_self):
                        rewind(self.fc_self, 'weights')
                else:
                    if prune.is_pruned(self.fc_self):
                        prune.remove(self.fc_self, 'weights')
            norm_node_in = tlx.sqrt(tlx.reduce_sum(tlx.square(x[0]), axis=0))
            norm_all_in = tlx.reduce_sum(norm_node_in) / x[0].shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in
                ThrInPrune.apply(self.fc_neigh, 'weights', threshold_wi, dim=0)
                if self.aggr != 'gcn':
                    ThrInPrune.apply(self.fc_self, 'weights', threshold_wi, dim=0)
        total_w_before = _tensor_numel(self.fc_neigh.weights)
        total_w_after = int(tlx.convert_to_numpy(tlx.reduce_sum(tlx.cast(self.fc_neigh.weights != 0, tlx.float32))))
        if self.aggr != 'gcn':
            total_w_before += _tensor_numel(self.fc_self.weights)
            total_w_after += int(tlx.convert_to_numpy(tlx.reduce_sum(tlx.cast(self.fc_self.weights != 0, tlx.float32))))
        self.logger_w.numel_before = total_w_before
        self.logger_w.numel_after = total_w_after
        out = self.propagate(x[0], edge_index, edge_weight=edge_weight, num_nodes=x[1].shape[0])
        out = self.fc_neigh(out)
        if self.aggr != 'gcn':
            out += self.fc_self(x[1])

        if self.add_bias:
            out += self.bias

        self.idx_lock = None
        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            num_edges = edge_index.shape[1]
            self.logger_a.numel_before = num_edges

            if self.idx_keep is None:
                self.idx_keep = tlx.arange(start=0, limit=num_edges, dtype=tlx.int64)

            self.idx_keep = self.idx_keep[self.idx_keep < num_edges]
            self.idx_keep = tlx.cast(self.idx_keep, tlx.int64).squeeze()
            self.logger_a.numel_after = self.idx_keep.shape[0]

            edge_index = edge_index[:, self.idx_keep]
            if edge_weight is not None:
                edge_weight = edge_weight[self.idx_keep]

        return out, (edge_index, edge_weight)

def Linear_cnt_flops(module, input, output):
    input = input[0]
    pre_last = np.prod(input.shape[0:-1], dtype=np.int64)
    bias_flops = output.shape[-1] if module.bias is not None else 0
    module.__flops__ += int((input.shape[-1]*output.shape[-1] + bias_flops) * pre_last * (module.logger_w.ratio if hasattr(module,'logger_w') else 1))

layer_dict_gsage = {
    'gsage': SAGEConvRaw,
    'gsage_unifews': SAGEConvThr,
}


flops_modules_dict_gsage = {
    nn.Linear: Linear_cnt_flops,
    SAGEConvRaw: SAGEConvRaw.cnt_flops,
    SAGEConvThr: SAGEConvThr.cnt_flops,
}