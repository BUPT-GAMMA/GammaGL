import tensorlayerx as tlx
from tensorlayerx.nn.layers import Linear


class Identity(tlx.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(tlx.nn.Module):
    def __init__(self,
                 channel_list=None,
                 in_channels=None,
                 hidden_channels=None,
                 out_channels=None,
                 num_layers=None,
                 act=tlx.nn.LeakyReLU(),
                 act_first=False,
                 norm=tlx.nn.BatchNorm1d(gamma_init='ones'),
                 dropout=0.0,

                 bias=True,
                 plain_last=True
                 ):
        super(MLP, self).__init__()
        self.act_first = act_first
        self.plain_last = plain_last
        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.act = act

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            dropout[-1] = 0.
        assert len(dropout) == len(channel_list) - 1
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        assert len(bias) == len(channel_list) - 1

        self.lins = tlx.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)

        for i, o, _bias in iterator:
            if _bias:
                self.lins.append(Linear(in_features=i, out_features=o))
            else:
                self.lins.append(Linear(in_features=i, out_features=o, b_init=None))


        self.norms = tlx.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]

        if norm is not None:
            assert isinstance(norm, tlx.nn.BatchNorm1d)
        for hidden_channels in iterator:
            if norm:
                self.norms.append(norm)
            else:
                self.norms.append(Identity())

    @property
    def in_channels(self):
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        return self.channel_list[-1]

    def forward(self, x, return_emb=None):
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)

            x = tlx.nn.Dropout(p=self.dropout[i])(x)
            emb = x

        if self.plain_last:
            x = self.lins[-1](x)
            x = tlx.nn.Dropout(p=self.dropout[-1])(x)

        return (x, emb) if isinstance(return_emb, bool) else x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'