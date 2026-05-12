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

from gammagl.layers.conv import (GATV2Conv,MessagePassing)
GATv2Conv = GATV2Conv

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
        self.logger_w.numel_after = self.linear.weights.numel()
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
      
        if msg_tensor.dim() == 3:
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
        

        self.logger_w.numel_before = self.linear.weights.numel()
        self.logger_w.numel_after = tlx.reduce_sum(self.linear.weights != 0).item()

     
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