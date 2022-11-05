# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/04/10 11:57
# @Author  : clear
# @FileName: rgcn_conv.py.py

import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.mpops import *
import torch

def masked_edge_index(edge_index, edge_mask):
    if tlx.BACKEND == 'mindspore':
        idx = tlx.convert_to_tensor([i for i, v in enumerate(edge_mask) if v], dtype=tlx.int64)
        return tlx.gather(edge_index, idx)
    else:
        return tlx.transpose(tlx.transpose(edge_index)[edge_mask])


class CompConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, op='sub', add_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.op = op
        self.add_bias = add_bias
        self.w_loop = tlx.layers.Linear(out_features=out_channels,
                                        in_features=in_channels,
                                        W_init='xavier_uniform',
                                        b_init=None)
        self.w_in = tlx.layers.Linear(out_features=out_channels,
                                        in_features =in_channels,
                                        W_init='xavier_uniform',
                                        b_init=None)
        self.w_out = tlx.layers.Linear(out_features=out_channels,
                                        in_features=in_channels,
                                        W_init='xavier_uniform',
                                        b_init=None)
        self.w_rel = tlx.layers.Linear(out_features=out_channels,
                                        in_features=in_channels,
                                        W_init='xavier_uniform',
                                        b_init=None)
        self.initor = tlx.initializers.truncated_normal()
        if self.add_bias:
            self.bias = self._get_weights(var_name="bias", shape=(out_channels,),  init=self.initor)

        return
    def forward(self, x, edge_index, edge_type=None,ref_emb=None):

        edge_half_num = int(edge_index.shape[1]/2)
        edge_in_index = edge_index[:,:edge_half_num]
        edge_out_index = edge_index[:,edge_half_num:]
        edge_in_type = edge_type[:edge_half_num]
        edge_out_type = edge_type[edge_half_num:]

        loop_index = [n for n in range(0, x.shape[0])]
        loop_index = tlx.ops.convert_to_tensor(loop_index)
        loop_index = tlx.ops.stack([loop_index,loop_index])
        loop_type = [self.num_relations for n in range(0, x.shape[0])]
        loop_type = tlx.ops.convert_to_tensor(loop_type)
        in_res = self.propagate(x,edge_in_index,edge_in_type,linear=self.w_in,rel_emb=ref_emb)
        out_res = self.propagate(x,edge_out_index,edge_out_type,linear=self.w_out,rel_emb=ref_emb)
        loop_res = self.propagate(x,loop_index,loop_type,linear=self.w_loop,rel_emb=ref_emb)
        ref_emb = self.w_rel(ref_emb)
        res = in_res*(1/3) + out_res*(1/3) + loop_res*(1/3)

        if self.add_bias:
            res = res + self.bias
        return res,ref_emb


    def propagate(self, x, edge_index,edge_type, aggr='sum', **kwargs):
        """
        Function that perform message passing.
        Args:
            x: input node feature
            edge_index: edges from src to dst
            aggr: aggregation type, default='sum', optional=['sum', 'mean', 'max']
            kwargs: other parameters dict

        """

        if 'num_nodes' not in kwargs.keys() or kwargs['num_nodes'] is None:
            kwargs['num_nodes'] = x.shape[0]

        coll_dict = self.__collect__(x, edge_index,edge_type, aggr, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        msg_kwargs['linear'] = kwargs['linear']
        msg_kwargs['rel_emb'] = kwargs['rel_emb']
        msg_kwargs['edge_type'] = edge_type
        msg = self.message(**msg_kwargs)
        x = self.aggregate(msg, edge_index, num_nodes=kwargs['num_nodes'], aggr=aggr,dim_size=x.shape[0])
        x = self.update(x)
        return x

    def __collect__(self, x, edge_index,edge_type, aggr, kwargs):
        out = {}

        for k, v in kwargs.items():
            out[k] = v
        out['x'] = x
        out['edge_index'] = edge_index
        out['aggr'] = aggr
        out['edge_type'] = edge_type
        return out

    def message(self, x, edge_index,edge_type, edge_weight=None,rel_emb=None,linear=None):
        """
        Function that construct message from source nodes to destination nodes.
        Args:
            x (Tensor): input node feature
            edge_index (Tensor): edges from src to dst
            edge_weight (Tensor): weight of each edge
        Returns:

        """
        import torch
       # rel_emb = rel_emb[edge_type]
        rel_emb = torch.index_select(rel_emb, 0 ,edge_type)
        x_emb = x[edge_index[1]]
        if self.op == 'sub': x_rel_emb = x_emb - rel_emb
        elif self.op == 'mult': x_rel_emb = x_emb * rel_emb
        elif self.op == 'corr': x_rel_emb = self.ccorr(x_emb,rel_emb)
        msg = linear(x_rel_emb)
        if edge_weight is not None:
            edge_weight = tlx.expand_dims(edge_weight, -1)
            return msg * edge_weight
        else:
            return msg

    def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum',dim_size=None):
        """
        Function that aggregates message from edges to destination nodes.
        Args:
            msg (Tensor): message construct by message function
            edge_index (Tensor): edges from src to dst
            num_nodes (int): number of nodes of the graph
            aggr (str): aggregation type, default = 'sum', optional=['sum', 'mean', 'max']

        Returns:

        """
        dst_index = edge_index[0, :]
        if aggr == 'sum':
            return unsorted_segment_sum(msg, dst_index, num_nodes)
            #return unsorted_segment_sum(msg, dst_index, num_nodes)
        elif aggr == 'mean':
            return unsorted_segment_mean(msg, dst_index, num_nodes)
        elif aggr == 'max':
            return unsorted_segment_max(msg, dst_index, num_nodes)
        else:
            raise NotImplementedError('Not support for this opearator')


    def ccorr(self, a, b):
        """
        Compute circular correlation of two tensors.
        Parameters
        ----------
        a: Tensor, 1D or 2D
        b: Tensor, 1D or 2D
        Notes
        -----
        Input a and b should have the same dimensions. And this operation supports broadcasting.
        Returns
        -------
        Tensor, having the same dimension as the input a.
        """

        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        try:
            from torch import irfft
            from torch import rfft
        except ImportError:
            from torch.fft import irfft2
            from torch.fft import rfft2

            def rfft(x, d):
                t = rfft2(x, dim=(-d))
                return torch.stack((t.real, t.imag), -1)

            def irfft(x, d, signal_sizes):
                return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d))

        return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
