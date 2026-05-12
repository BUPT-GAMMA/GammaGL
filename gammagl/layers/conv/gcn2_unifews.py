import os
if 'TL_BACKEND' not in os.environ:
    os.environ['TL_BACKEND'] = 'torch'

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


from gammagl.gglspeedup.prunes_gamma import ThrInPrune, rewind,prune
from gammagl.utils.logger_gamma import LayerNumLogger


def norm(x, p=2, axis=None, keepdims=False):
    """
    Manual implementation of the norm function for TensorLayerX.
    """
    # Ensure we are working with absolute values for the norm calculation
    x_abs = tlx.abs(x)
    
    if p == 1:
        return tlx.reduce_sum(x_abs, axis=axis, keepdims=keepdims)
    elif p == 2:
        # Standard Euclidean norm: sqrt(sum(x^2))
        return tlx.sqrt(tlx.reduce_sum(x_abs * x_abs, axis=axis, keepdims=keepdims))
    else:
        # General Lp norm
        sum_p = tlx.reduce_sum(tlx.pow(x_abs, p), axis=axis, keepdims=keepdims)
        return tlx.pow(sum_p, 1.0/p)

def normalize(x, p=2., axis=-1):
    # Use our new manual norm function here
    norm_val = norm(x, p=p, axis=axis, keepdims=True)
    return x / (norm_val + 1e-12)


def reset_weight_(weight: Tensor, in_channels: int, initializer: Optional[str] = None) -> Tensor:
    if weight is None:
        return weight
    shape = weight.shape
    device = weight.device if hasattr(weight, 'device') else None
    if in_channels <= 0:
        return
    elif initializer == 'glorot':
        new_weight = tlx.initializers.XavierUniform()(shape)
    elif initializer == 'uniform':
        bound = 1.0 / np.sqrt(in_channels)
        new_weight = tlx.initializers.RandomUniform(-bound, bound)(shape)
    elif initializer == 'kaiming_uniform' or initializer is None:
        new_weight = tlx.initializers.HeUniform()(shape)
    else:
        raise RuntimeError(f"Weight initializer '{initializer}' not supported")
    if device is not None:
        new_weight = tlx.convert_to_tensor(new_weight, device=device)
    if hasattr(weight, 'data'):
        weight.data = new_weight
    else:
        weight.copy_(new_weight)
    return weight

def reset_bias_(bias: Optional[Tensor], in_channels: int, initializer: Optional[str] = None) -> Optional[Tensor]:
    if bias is None or in_channels <= 0:
        return bias
    shape = bias.shape
    device = bias.device if hasattr(bias, 'device') else None
    if initializer == 'zeros':
        new_bias = tlx.initializers.Zeros()(shape)
    elif initializer == 'uniform' or initializer is None:
        bound = 1.0 / np.sqrt(in_channels)
        new_bias = tlx.initializers.RandomUniform(-bound, bound)(shape)
    else:
        raise RuntimeError(f"Bias initializer '{initializer}' not supported")
    if device is not None:
        new_bias = tlx.convert_to_tensor(new_bias, device=device)
    if hasattr(bias, 'data'):
        bias.data = new_bias
    else:
        bias.copy_(new_bias)
    return bias

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    return int(tlx.reduce_max(edge_index)) + 1

def scatter(src, index, axis=0, dim_size=None, reduce='sum'):
    if dim_size is None:
        dim_size = int(tlx.reduce_max(index)) + 1
    if reduce == 'sum':
        return tlx.unsorted_segment_sum(src, index, num_segments=dim_size)
    elif reduce == 'max':
        return tlx.unsorted_segment_max(src, index, num_segments=dim_size)
    elif reduce == 'mean':
        return tlx.unsorted_segment_mean(src, index, num_segments=dim_size)
    elif reduce == 'min':
        return tlx.unsorted_segment_min(src, index, num_segments=dim_size)
    else:
        raise ValueError(f"Unsupported reduce type: {reduce}")

def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1.0, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    
    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    edge_index = edge_index[:, mask]
    
    loop_index = tlx.convert_to_tensor(np.arange(num_nodes), dtype=edge_index.dtype)
    loop_index = tlx.stack([loop_index, loop_index], axis=0)
    
    edge_index = tlx.concat([edge_index, loop_index], axis=1)
    
    if edge_weight is not None:
        loop_weight = tlx.ones((num_nodes,), dtype=edge_weight.dtype) * fill_value
        edge_weight = tlx.concat([edge_weight, loop_weight], axis=0)
        
    return edge_index, edge_weight

def pow_with_pinv(x, p: float):
    x = tlx.convert_to_tensor(x)
    x_pow = tlx.pow(x, p)
    x_safe = tlx.where(tlx.is_inf(x_pow), tlx.zeros_like(x_pow), x_pow)
    return x_safe

def leaky_relu(x, negative_slope=0.2):
    return tlx.where(x > 0.0, x, x * negative_slope)

def softmax(src, index, ptr=None, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(tlx.reduce_max(index)) + 1
    src_max = scatter(src, index, axis=0, dim_size=num_nodes, reduce='max')
    src_max_gathered = tlx.gather(src_max, index)
    out = tlx.exp(src - src_max_gathered)
    out_sum = scatter(out, index, axis=0, dim_size=num_nodes, reduce='sum')
    out_sum_gathered = tlx.gather(out_sum, index)
    return out / (out_sum_gathered + 1e-16)

def gcn_norm(edge_index, edge_weight, num_nodes, improved=False, add_self_loops=True, flow="source_to_target", dtype=tlx.float32):
    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1.0, num_nodes)
    if edge_weight is None:
        edge_weight = tlx.ones((edge_index.shape[1],), dtype=dtype)
    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, axis=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = pow_with_pinv(deg, -0.5)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, edge_weight

def identity_n_norm(edge_index, edge_weight=None, num_nodes=None,
                    rnorm=None, diag=1., dtype=tlx.float32):
    if tlx.is_tensor(edge_index):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if diag is not None:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, diag, num_nodes)
        if rnorm is None:
            return edge_index
        else:
            edge_weight = tlx.ones((edge_index.shape[1], ), dtype=dtype,
                                     device=edge_index.device)
            row, col = edge_index[0], edge_index[1]
            idx = col
            deg = scatter(edge_weight, idx, axis=0, dim_size=num_nodes, reduce='sum')
            deg_inv_sqrt = pow_with_pinv(deg, -0.5)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight
    raise NotImplementedError()



class ConvThr(nn.Module):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_a = thr_a
        self.threshold_w = thr_w
        self.idx_keep = tlx.convert_to_tensor([])
        self.prune_lst = []
        self.scheme_a = 'full'
        self.scheme_w = 'full'

    def propagate_forward_print(self, module, inputs, output):
        print(inputs[0], inputs[0].shape, inputs[1])
        print(output, output.shape)

    def get_idx_lock(self, edge_index, node_lock):
    # Ensure starting with a valid empty tensor for concatenation
        idx_lock = tlx.convert_to_tensor([], dtype=tlx.int64)
        if edge_index.shape[1] == 0:
            return idx_lock

        bs = int(2**28 / edge_index.shape[1]) if edge_index.shape[1] > 0 else 1

        for i in range(0, node_lock.shape[0], bs):
            batch = node_lock[i:min(i+bs, node_lock.shape[0])]
            mask = tlx.reduce_any(tlx.expand_dims(edge_index[1], 0) == tlx.expand_dims(batch, 1), axis=0)
            batch_idx = tlx.arange(0, edge_index.shape[1])[mask]
            idx_lock = tlx.concat((idx_lock, tlx.cast(batch_idx, tlx.int64)), axis=0)
        diag_mask = edge_index[0] == edge_index[1]
        idx_diag = tlx.arange(0, edge_index.shape[1])[diag_mask]
    
        idx_lock = tlx.concat((idx_lock, tlx.cast(idx_diag, tlx.int64)), axis=0)
        return tlx.unique(idx_lock)

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
        total_w = self.linear.weights.numel()
        if self.variant:
            total_w += self.linear0.weights.numel()
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

        self.logger_w.numel_before = self.linear.weights.numel()
        self.logger_w.numel_after = int(tlx.reduce_sum(tlx.cast(self.linear.weights != 0, tlx.float32)).item())

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