import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_dropout(x: torch.Tensor, p: float, training: bool):
    x = x.coalesce()
    return torch.sparse_coo_tensor(x.indices(), F.dropout(x.values(), p=p, training=training),
                                   size=x.size())


class FFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.dropout(x)
        x = F.relu(self.lin1(x))
        return x
        # return self.lin2(x)


class SparseFFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(SparseFFN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = sparse_dropout(x, 0.5, self.training)
        x = F.relu(self.lin1(x))
        return self.lin2(x) + x
