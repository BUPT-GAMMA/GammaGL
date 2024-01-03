# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/11/8 23:47
# @Author  : yijian
# @FileName: compgcn_conv.py.py

import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.mpops import *

def masked_edge_index(edge_index, edge_mask):
    if tlx.BACKEND == 'mindspore':
        idx = tlx.convert_to_tensor([i for i, v in enumerate(edge_mask) if v], dtype=tlx.int64)
        return tlx.gather(edge_index, idx)
    else:
        return tlx.transpose(tlx.transpose(edge_index)[edge_mask])


class CompConv(MessagePassing):
    '''
    Paper: Composition-based Multi-Relational Graph Convolutional Networks

    Code: https://github.com/MichSchli/RelationPrediction

    Parameters
    ----------
    in_channels: int
        the input dimension of the features.
    out_channels: int
        the output dimension of the features.
    num_relations: int
        the number of relations in the graph.
    op: str
        the operation used in message creation.
    add_bias: bool
        whether to add bias.

    '''
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

        Parameters
        ----------
        x: 
            input node feature.
        edge_index: 
            edges from src to dst.
        aggr: 
            aggregation type, default='sum', optional=['sum', 'mean', 'max'].
        kwargs: 
            other parameters dict.

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
        
        Parameters
        ----------
        x: tensor
            input node feature.
        edge_index: tensor
            edges from src to dst.
        edge_weight: tensor
            weight of each edge.

        Returns
        -------
        tensor
            output message.

        """
        rel_emb = tlx.gather(rel_emb, edge_type)
        x_emb = tlx.gather(x, edge_index[1])
        if self.op == 'sub': x_rel_emb = x_emb - rel_emb
        elif self.op == 'mult': x_rel_emb = x_emb * rel_emb
        msg = linear(x_rel_emb)
        if edge_weight is not None:
            edge_weight = tlx.expand_dims(edge_weight, -1)
            return msg * edge_weight
        else:
            return msg

    def aggregate(self, msg, edge_index, num_nodes=None, aggr='sum',dim_size=None):
        """
        Function that aggregates message from edges to destination nodes.

        Parameters
        ----------
        msg: tensor
            message construct by message function.
        edge_index: tensor
            edges from src to dst.
        num_nodes: int
            number of nodes of the graph.
        aggr: str
            aggregation type, default = 'sum', optional=['sum', 'mean', 'max'].

        Returns
        -------
        tensor
            output representation.

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

