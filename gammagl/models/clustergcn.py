import tensorlayerx as tlx
from tensorlayerx.nn import Module, ModuleList, BatchNorm
from gammagl.layers.conv import SAGEConv
from tqdm import tqdm


class ClusterGCNModel_reddit(Module):
    """
    Parameters
    ----------
    in_channels(int): input  dimension
    hidden_dim(int): hidden dimension
    out_channels(int): number of classes
    drop_rate(float): dropout rate
    num_layers(int): number of layers
    """

    def __init__(self, in_channels,
             hidden_dim,
             out_channels,
             drop_rate=0.5,
             num_layers=2,
             ):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.conv = tlx.nn.ModuleList([SAGEConv(in_channels, out_channels)])
        else:
            self.conv = tlx.nn.ModuleList([SAGEConv(in_channels, hidden_dim)])
            for _ in range(1, num_layers - 1):
                self.conv.append(SAGEConv(hidden_dim, hidden_dim))
            self.conv.append(SAGEConv(hidden_dim, out_channels))
            self.relu = tlx.ReLU()
            self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return tlx.nn.LogSoftmax(dim=-1)(x)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.shape[0] * len(self.convs))
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        # print('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, adj, n_id in subgraph_loader:
                sg = adj[0]
                if tlx.BACKEND == 'torch':
                    edge_index = sg.edge.to(device)
                else:
                    edge_index = sg.edge
                size = sg.size
                if tlx.BACKEND == 'torch':
                    x = tlx.gather(x_all, n_id).to(device)
                else:
                    x = tlx.gather(x_all, n_id)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = tlx.nn.ReLU()(x)
                if tlx.BACKEND == 'torch':
                    xs.append(x.cpu())
                else:
                    xs.append(x)
                pbar.update(len(batch_size))

            x_all = tlx.concat(xs, 0)

        pbar.close()

        return x_all

class ClusterGCNModel_ppi(Module):
    """
        Parameters
        ----------
        in_channels(int): input  dimension
        hidden_channels(int): hidden dimension
        out_channels(int): number of classes
        drop_rate(float): dropout rate
        num_layers(int): number of layers
        """

    def __init__(self, in_channels, hidden_channels, out_channels, drop_rate=0.2,num_layers=6):
        super().__init__()
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(num_features=hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(num_features=hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(drop_rate)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = self.relu(x)
            x = self.dropout(x)
        return self.convs[-1](x, edge_index)
