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
    MessagePassing,GCNIIConv
)


from gammagl.gglspeedup.prunes_gamma import ThrInPrune, rewind,prune, _tensor_numel
from gammagl.utils.logger_unifews import LayerNumLogger
from .gcn_unifews import identity_n_norm, gcn_norm, softmax, leaky_relu, scatter, maybe_num_nodes, reset_weight_, reset_bias_,ConvThr,add_remaining_self_loops,pow_with_pinv,normalize,norm



class GCNIIConvRaw(GCNIIConv):
    def __init__(self, in_channels, out_channels, alpha, beta, variant=False,
                 rnorm=None, diag=1., depth_inv=False, **kwargs):
        self.rnorm = rnorm
        self.diag = diag
        self.depth_inv = depth_inv
        
        kwargs.pop('thr_a', None)
        kwargs.pop('thr_w', None)
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            alpha=alpha,
            beta=beta,
            variant=variant
        )
        
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()

    def forward(self, x, x_0, edge_tuple: PairTensor):
        edge_index, edge_weight = edge_tuple
       
        self.logger_a.numel_after = edge_index.shape[1]
        total_w = _tensor_numel(self.linear.weights)
        if self.variant:
            total_w += _tensor_numel(self.linear0.weights)
        self.logger_w.numel_after = total_w
        
        num_nodes = tlx.get_tensor_shape(x)[0]
        return super().forward(x0=x_0, x=x, edge_index=edge_index, 
                               edge_weight=edge_weight, num_nodes=num_nodes)

    def reset_parameters(self):
        reset_weight_(self.linear.weights, self.in_channels, initializer='glorot')
        if self.variant:
            reset_weight_(self.linear0.weights, self.in_channels, initializer='glorot')
        if hasattr(self, 'bias') and self.bias is not None:
            reset_bias_(self.bias, self.out_channels, initializer='zeros')

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, x_0, (edge_index, edge_weight) = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]
        module.__flops__ += int(f_in * f_out * n) * (2 if module.variant else 1)
        module.__flops__ += f_in * m



class GCNIIConvThr(ConvThr, GCNIIConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.linear]
        if self.variant:
            self.prune_lst.append(self.linear0)
        
        self.idx_keep = None

    def forward(self, x, x_0, edge_tuple, node_lock=None, verbose=False):
        (edge_index, edge_weight) = edge_tuple
        num_nodes = tlx.get_tensor_shape(x)[0]
        num_edges = edge_index.shape[1]
        self.current_edge_count = num_edges

        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.linear): rewind(self.linear, 'weights')
                if self.variant and prune.is_pruned(self.linear0): rewind(self.linear0, 'weights')
            else:
                if prune.is_pruned(self.linear): prune.remove(self.linear, 'weights')
                if self.variant and prune.is_pruned(self.linear0): prune.remove(self.linear0, 'weights')
            
            norm_node_in = tlx.sqrt(tlx.reduce_sum(tlx.square(x), axis=0))
            norm_all_in = tlx.reduce_sum(norm_node_in) / (x.shape[1] + 1e-10)
            
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in
                ThrInPrune.apply(self.linear, 'weights', threshold_wi)
                if self.variant:
                    ThrInPrune.apply(self.linear0, 'weights', threshold_wi)

        self.logger_w.numel_before = _tensor_numel(self.linear.weights)
        self.logger_w.numel_after = int(tlx.convert_to_numpy(tlx.reduce_sum(tlx.cast(self.linear.weights != 0, tlx.float32))))

        self.logger_a.numel_before = num_edges
        if self.idx_keep is None:
            self.idx_keep = tlx.arange(start=0, limit=num_edges, dtype=tlx.int64)
        
        self.idx_keep = self.idx_keep[self.idx_keep < num_edges]
        self.idx_keep = tlx.cast(self.idx_keep, tlx.int64).squeeze()
        self.logger_a.numel_after = self.idx_keep.shape[0] if self.idx_keep.ndim > 0 else 1
        
        edge_index = tlx.gather(edge_index, self.idx_keep, axis=1)
        if edge_weight is not None:
            edge_weight = tlx.gather(edge_weight, self.idx_keep)

        m = self.propagate(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        
        m = m * (1 - self.alpha)
        x_0_part = x_0[:num_nodes] * self.alpha
        
        if not self.variant:
            out = m + x_0_part
            out = out * (1 - self.beta) + tlx.matmul(out, self.linear.weights) * self.beta
        else:
            out = (m * (1 - self.beta) + tlx.matmul(m, self.linear.weights) * self.beta + 
                   x_0_part * (1 - self.beta) + tlx.matmul(x_0_part, self.linear0.weights) * self.beta)

        return out, (edge_index, edge_weight)

def Linear_cnt_flops(module, input, output):
    input = input[0]
    pre_last = np.prod(input.shape[0:-1], dtype=np.int64)
    bias_flops = output.shape[-1] if module.bias is not None else 0
    module.__flops__ += int((input.shape[-1]*output.shape[-1] + bias_flops) * pre_last * (module.logger_w.ratio if hasattr(module,'logger_w') else 1))

layer_dict_gcn2 = {
    'gcn2': GCNIIConvRaw,
    'gcn2_unifews': GCNIIConvThr,
}



flops_modules_dict_gcn2 = {
    nn.Linear: Linear_cnt_flops,
    GCNIIConvRaw: GCNIIConvRaw.cnt_flops,
    GCNIIConvThr: GCNIIConvThr.cnt_flops,
}