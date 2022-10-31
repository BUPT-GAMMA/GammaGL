import tensorlayerx as tlx
from gammagl.layers.conv.message_passing import MessagePassing
from gammagl.utils.film_utils import split_to_two

class FILMConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations=1,
                 act=tlx.nn.ReLU()):
        super(FILMConv, self).__init__()

        Linear = tlx.layers.Linear

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.act = act

        self.lins = tlx.nn.ModuleList()
        self.films = tlx.nn.ModuleList()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        for _ in range(num_relations):

            self.lins.append(Linear(in_features=in_channels[0], out_features=out_channels, b_init=None))
            self.films.append(Linear(in_features=in_channels[1], out_features=2 * out_channels))

        self.lin_skip = Linear(in_features=in_channels[1], out_features=out_channels)
        self.film_skip = Linear(in_features=in_channels[1], out_features=2 * out_channels, b_init=None)

    def forward(self, x, edge_index):

        x = (x, x)

        beta, gamma = split_to_two(self.film_skip(x[1]), axis=-1)
        out = tlx.multiply(gamma, self.lin_skip(x[1])) + beta

        if self.act is not None:
            out = self.act(out)

        beta, gamma = split_to_two(self.films[0](x[1]), axis=-1)
        out = out + self.propagate(x=self.lins[0](x[0]), edge_index=edge_index, beta=beta, gamma=gamma)

        return out

    def message(self, x, edge_index, beta, gamma, edge_weight=None):

        msg = tlx.gather(x, edge_index[0, :])
        beta = tlx.gather(beta, edge_index[0, :])
        gamma = tlx.gather(gamma, edge_index[0, :])

        msg = tlx.multiply(gamma, msg) + beta

        if self.act is not None:
            msg = self.act(msg)
        return msg
