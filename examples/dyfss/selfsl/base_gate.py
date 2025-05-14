r"""
Base gate with standard interface
"""

import tensorlayerx as tlx
import tensorlayerx.nn as nn


class BaseGate(nn.Module):
    def __init__(self, num_expert):
        super().__init__()
        self.tot_expert = num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None
