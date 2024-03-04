import tensorlayerx as tlx

class DeepGCNLayer(tlx.nn.Module):
    def __init__(self,
                 conv = None,
                 norm = None,
                 act = None,
                 block = None,
                 dropout = 0.0):
        super().__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.dropout = dropout
        self.drop = tlx.nn.Dropout(dropout)


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, *args, **kwargs):
        """"""
        args = list(args)
        x = args.pop(0)

        h = x
        if self.norm is not None:
            h = self.norm(h)
        if self.act is not None:
            h = self.act(h)
        h = self.drop(h)
        h = self.conv(h, *args, **kwargs)

        return x + h



    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block})'