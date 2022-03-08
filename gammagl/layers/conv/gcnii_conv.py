import tensorlayer as tl
from gammagl.layers.conv import MessagePassing

class GCNIIConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha,
                 beta,
                 variant=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.linear = tl.layers.Dense(n_units=self.out_channels,
                                      in_channels=self.in_channels,
                                      b_init=None)
        if self.variant:
            # what if same as linear
            self.linear0 = tl.layers.Dense(n_units=self.out_channels,
                                      in_channels=self.in_channels,
                                      b_init=None)
        
    def message_aggregate(self, x, sparse_adj):
        return sparse_adj @ x

    def forward(self, x0, x, sparse_adj, edge_weight=None):
        if self.variant:
            x = (1-self.alpha)*self.propagate(x, sparse_adj)
            x = (1-self.beta)*x + self.beta*self.linear(x)
            x0 = self.alpha*x0
            x0 = (1-self.beta)*x0 + self.beta * self.linear0(x0)
            x = x + x0

            # x = (1 - self.alpha) * self.propagate(x, sparse_adj)
            # x0 = self.alpha * x0
            # x = x + x0
            # x = (1-self.beta)*x + self.beta*self.linear(x)
        else:
            x = self.propagate(x, sparse_adj)
            x = (1-self.alpha)*x + self.alpha*x0
            x = (1-self.beta)*x + self.beta*self.linear(x)
        return x
