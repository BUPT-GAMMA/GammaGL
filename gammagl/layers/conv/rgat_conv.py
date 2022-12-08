# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/11/10
# @Author  : Hongzhe Bi
# @FileName: rgat_conv.py.py


from selectors import BaseSelector
import tensorlayerx as tlx
from tensorlayerx.nn.initializers import Ones, Zeros, XavierNormal
from gammagl.layers.conv import MessagePassing
import torch

def masked_edge_index(edge_index, edge_mask):
    if tlx.BACKEND == 'mindspore':
        idx = tlx.convert_to_tensor([i for i, v in enumerate(edge_mask) if v], dtype=tlx.int64)
        return tlx.gather(edge_index, idx)
    else:
        return tlx.transpose(tlx.transpose(edge_index)[edge_mask])


class RGATConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_relations,
        attention_mechanism,
        attention_mode,
        heads,
        dim,
        concat,
        negative_slope,
        bias,
    ):
        super().__init__()
        self.heads = heads
        self.negative_slope = negative_slope
        self.activation = tlx.nn.ReLU()
        self.concat = concat
        self.attention_mode = attention_mode
        self.attention_mechanism = attention_mechanism
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        #self.in_channels_l = in_channels[0]
        self.leaky_relu = tlx.layers.LeakyReLU()
        self.lin = tlx.nn.Linear(in_channels, out_channels)
        initor = tlx.initializers.truncated_normal()

        if(self.attention_mechanism != "within-relation"
            and self.attention_mechanism != "across-relation"):
            raise ValueError('attention mechanism must either be'
                            '"within-relation" or "across-relation"')
        
        if(self.attention_mode != "additive-self-attention"
            and self.attention_mode != "multiplicative-self-attention"):
            raise ValueError('attention mode must either be '
                              '"additive-self-attention" or '
                              '"multiplicative-self-attention"')
        
        if self.attention_mode == "additive-self-attention" and self.dim > 1:
            raise ValueError('"additive-self-attention mode cannot be '
                             'applied when value of d is greater than 1. '
                             'Use "multiplicative-self-attention" instead.')
        
        # The learnable parameters to compute both attention logits and
        # attention coefficients:
        self.q = self._get_weights(var_name="q",
                        shape=(self.heads * self.out_channels,
                        self.heads * self.dim),
                        init=initor, order=True)
        self.k = self._get_weights(var_name="k",
                        shape=(self.heads * self.out_channels,
                        self.heads * self.dim),
                        init=initor, order=True)
        
        if bias and concat:
            self.bias = self._get_weights(var_name="bias",
                        shape=(self.heads * self.dim * self.out_channels),
                        init=initor, order=True)
        elif bias and not concat:
            self.bias = self._get_weights(var_name="bias",
                        shape=(self.dim * self.out_channels),
                        init=initor, order=True)
        #else:
        #    tlx.nn.Parameter(None, 'bias')

        self.weight = self._get_weights(var_name="weight",shape=(self.num_relations,self.in_channels,
                                                                    self.heads * self.out_channels),
                                                                    init=initor, order=True)
        #self.w = tlx.nn.Parameter(tlx.ops.ones((self.out_channels), dtype=tlx.int32, device='cpu'),'w')
        self.w = self._get_weights(var_name="w",shape=(self.out_channels,1), init=initor, order=True)
        self.l1 = self._get_weights(var_name="l1",shape=(1, self.out_channels),
                                     init=initor, order=True)
        self.b1 = self._get_weights(var_name="b1",shape=(1, self.out_channels),
                                     init=initor, order=True)
        self.l2 = self._get_weights(var_name="l2",shape=(self.out_channels, self.out_channels),
                                     init=initor, order=True)
        self.b2 = self._get_weights(var_name="b2",shape=(1, self.out_channels),
                                     init=initor, order=True)

    def forward(self, x , edge_index, edge_type = None):
        
        weight = self.weight
        print('weight',weight.shape)
        inchannels_l = self.in_channels
        x_l = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = tlx.arange(0, inchannels_l, dtype=tlx.int64)
        x_r = x_l
        if isinstance(x, tuple):
            x_r = x[1]
        print("x_r",x_r.shape)
        size = (x_l.shape[0], x_r.shape[0])

        x = tlx.zeros(shape=(x_l.shape[0], self.out_channels), dtype=tlx.float32)
        out = self.propagate(edge_index=edge_index, edge_type=edge_type, x=x)
        return out
    
    def message(self, x, edge_type, edge_index):
        row, col = edge_index
        #x = self.lin(x)
        x_i = x[row]
        x_j = x[col]
        alpha_edge, alpha = 0, 0  #???
        w = self.weight
        w = torch.index_select(w, 0, edge_type) #???
        print('w',w.shape)
        print('x',x.shape)
        print('xi',x_i.shape)
        #x_i = tlx.nn.Transpose(perm=[1,0], conjugate=False, name='trans')(x_i)
        #outi = tlx.squeeze(tlx.bmm(x_i.unqueeze(1),w), axis=-2)
        #a = tlx.ops.expand_dims(x_i,axis=2)
        #print('a',a.shape)
        outi = tlx.squeeze(tlx.bmm(tlx.ops.expand_dims(x_i,axis=1), w), axis=-2)
        outj = tlx.squeeze(tlx.bmm(tlx.ops.expand_dims(x_j,axis=1), w), axis=-2)
        qi = tlx.matmul(outi,self.q)
        kj = tlx.matmul(outj,self.k)
        if self.attention_mode == "additive-self-attention":
            alpha = tlx.add(qi,kj)
            alpha = self.leaky_relu(alpha,self.negative_slope)
        elif self.attention_mode == "multiplicate-self-attention":
            alpha = qi * kj
        self._alpha = alpha
        if self.attention_mechanism == "within-relation":
            across_out = tlx.zeros_like(alpha)
            for r in range(self.num_relations):
                mask = edge_type == r
                across_out[mask] = tlx.nn.activation.Softmax(alpha[mask], edge_index[mask])
        elif self.attention_mechanism == "across-relation":
            alpha = tlx.nn.activation.Softmax(alpha, edge_index)
        if self.attention_mode == "additive-self-attention":
            return tlx.reshape(alpha, [-1, self.heads, 1]) * tlx.reshape(outj, [-1, self.heads, self.out_channels])
        else:
            return tlx.reshape(alpha, [-1, self.heads, self.dim, 1]) * tlx.reshape(outj, [1, self.heads, 1, self.out_channels])

    def update(self, aggr_out):
        if self.attention_mode == "additive-self-attention":
            if self.concat is True:
                aggr_out = tlx.ops.reshape(aggr_out, [-1, self.bias * self.out_channels])
            else:
                aggr_out = tlx.ops.reduce_mean(aggr_out, axis=1)     
            if self.bias is not None:
                aggr_out = aggr_out + self.bias
            return aggr_out
        else:
            if self.concat is True:
                aggr_out = tlx.ops.reshape(aggr_out, [-1, self.heads * self.dim * self.out_channels])
            else:
                aggr_out = tlx.ops.reduce_mean(aggr_out, axis=1)
                aggr_out = tlx.ops.reshape(aggr_out, [-1, self.dim * self.out_channels])    
            if self.bias is not None:
                aggr_out = aggr_out + self.bias    
            return aggr_out