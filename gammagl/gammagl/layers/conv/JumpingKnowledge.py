# -*- coding: utf-8 -*-
"""
@File   ： JumpingKnowledge.py
@Time   ： 2022/4/10 11:36 上午
@Author ： Jia Yiming
"""
import tensorlayerx as tlx
from tensorlayerx.nn.layers import LSTM, Linear


class JumpingKnowledge(tlx.nn.Module):
    r"""The Jumping Knowledge layer aggregation module from the
        `"Representation Learning on Graphs with Jumping Knowledge Networks"
        <https://arxiv.org/abs/1806.03536>`_ paper based on either


        - **concatenation** (:obj:`"cat"`)
        
        .. math::
            \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)}

        - **max pooling** (:obj:`"max"`)

        .. math::
            \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right)

        - **weighted summation**

        .. math::
            \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

        with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
        LSTM (:obj:`"lstm"`).

        Parameters
        ----------
        mode: str
            The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        channels: int, optional
            The number of channels per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers: int, optional
            The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)

        """

    def __init__(self, mode, channels=None, num_layers=None):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']
        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(input_size=channels,hidden_size= channels,
                             bidirectional=True, batch_first=True)
            self.att = Linear(in_features=2 * channels, out_features=1)

    def forward(self, xs):
        r"""Aggregates representations across different layers.

        Parameters
        ----------
        xs: list, tuple
            List containing layer-wise representations.

        """
        assert isinstance(xs, list) or isinstance(xs, tuple)


        if self.mode == 'cat':
            x = tlx.concat(xs, axis=-1)
            return x
        elif self.mode == 'max':
            x = tlx.stack(xs, axis=-1)
            x = tlx.reduce_max(x, axis=-1)
            return x
        elif self.mode == 'lstm':
            x = tlx.stack(xs, axis=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = tlx.squeeze(self.att(alpha),axis=-1)
            alpha = tlx.softmax(alpha, axis=-1)
            x = x * tlx.expand_dims(alpha,axis=-1)
            x = tlx.reduce_sum(x,axis=1)
            return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mode})'