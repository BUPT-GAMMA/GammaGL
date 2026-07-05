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

from gammagl.layers.conv import (GATV2Conv,MessagePassing)
GATv2Conv = GATV2Conv

from gammagl.gglspeedup.prunes_gamma import ThrInPrune, rewind,prune, _tensor_numel
from gammagl.utils.logger_unifews import LayerNumLogger
from .gcn_unifews import identity_n_norm, gcn_norm, softmax, leaky_relu, scatter, maybe_num_nodes, reset_weight_, reset_bias_,ConvThr,add_remaining_self_loops,pow_with_pinv,normalize,norm


class GATv2ConvRaw(GATV2Conv):
    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 rnorm=None, diag=1., depth_inv=False, heads: int = 1, concat: bool = True, **kwargs):

        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        if depth == 0:
            final_out = out_channels
            heads = 1
            concat = False
        else:

            final_out = out_channels
            heads = heads
            concat = concat

        super().__init__(in_channels, final_out, heads, concat,** kwargs)

        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()
        self.reset_parameters()


    def reset_parameters(self):

        reset_weight_(self.linear.weights, self.in_channels, initializer='glorot')

        reset_weight_(self.att_src, self.out_channels, initializer='glorot')
        reset_weight_(self.att_dst, self.out_channels, initializer='glorot')

        if self.bias is not None:
            reset_bias_(self.bias, self.heads * self.out_channels, initializer='zeros')

    def propagate(self, x, edge_index, edge_weight=None, num_nodes=None):
        return super().propagate(x, edge_index)


    def forward(self, x, edge_index, edge_weight=None):

        if isinstance(edge_index, (tuple,list)):
            edge_index, edge_weight = edge_index

        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = _tensor_numel(self.linear.weights)
        return super().forward(x, edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, edge_index = input
        f_in, f_h, f_c = x_in.shape[-1], module.heads, module.out_channels
        n, m = x_in.shape[0], edge_index[0].shape[1] if isinstance(edge_index, tuple) else edge_index.shape[1]
        flops_lin = f_in * f_h * f_c * n
        module.__flops__ += flops_lin
        flops_attn  = (2 * f_c + 2) * m * f_h
        module.__flops__ += flops_attn
        if module.bias is not None:
            module.__flops__ += (f_h * f_c if module.concat else f_c + 1) * n

class GATv2ConvThr(ConvThr, GATv2ConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.linear]
        self.idx_keep = None

        self.register_forward_hook(self.prune_on_msg)

    def prune_on_msg(self, module, inputs, output):
        msg_tensor = output[0] if isinstance(output, (list, tuple)) else output

        if len(tlx.get_tensor_shape(msg_tensor)) == 3:
            msg_tensor = tlx.reduce_mean(msg_tensor, dim=1)

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
        H, C = self.heads, self.out_channels

        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.linear):
                    rewind(self.linear, 'weights')
            else:
                if prune.is_pruned(self.linear):
                    prune.remove(self.linear, 'weights')

            norm_node_in = tlx.sqrt(tlx.reduce_sum(tlx.square(x), axis=0))
            norm_all_in = tlx.reduce_sum(norm_node_in) / x.shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in
                ThrInPrune.apply(self.linear, 'weights', threshold_wi)

            x = self.linear(x)

            x = tlx.reshape(x, (-1, H, C))

        elif self.scheme_w == 'keep':
            x = self.linear(x)

            x = tlx.reshape(x, (-1, H, C))
        else:
            raise NotImplementedError()


        self.logger_w.numel_before = _tensor_numel(self.linear.weights)
        self.logger_w.numel_after = int(tlx.convert_to_numpy(tlx.reduce_sum(tlx.cast(self.linear.weights != 0, tlx.float32))))


        #self.idx_lock = None

        out = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=tlx.get_tensor_shape(x)[0])


        if self.concat:
            out = tlx.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = tlx.reduce_mean(out, axis=1)

        if self.bias is not None:
            out = out + self.bias

        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            num_edges = edge_index.shape[1]
            self.logger_a.numel_before = edge_index.shape[1]

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

layer_dict_gat = {
    'gat': GATv2ConvRaw,
    'gat_unifews': GATv2ConvThr,
}


flops_modules_dict_gat = {
    nn.Linear: Linear_cnt_flops,
    GATv2ConvRaw: GATv2ConvRaw.cnt_flops,
    GATv2ConvThr: GATv2ConvThr.cnt_flops,
}