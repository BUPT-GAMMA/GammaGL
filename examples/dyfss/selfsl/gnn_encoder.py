import tensorlayerx as tlx
from tensorlayerx import nn
import math


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self._get_weights('weight', shape=(in_features, out_features))
        if with_bias:
            self.bias = self._get_weights('bias', shape=(out_features,))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight = tlx.ops.assign(self.weight, 
                                   tlx.initializers.RandomUniform(-stdv, stdv)(self.weight.shape))
        if self.bias is not None:
            self.bias = tlx.ops.assign(self.bias,
                                    tlx.initializers.RandomUniform(-stdv, stdv)(self.bias.shape))

    def forward(self, input, adj):
        if isinstance(input, tlx.sparse_tensor):
            support = tlx.sparse_dense_matmul(input, self.weight)
        else:
            support = tlx.matmul(input, self.weight)
            
        if isinstance(adj, tuple):  
            indices, values, shape = adj
            output = tlx.sparse_dense_matmul(indices, values, shape, support)
        else:
            output = tlx.matmul(adj, support)
            
        if self.bias is not None:
            output += self.bias
        return output

    def batch_forward(self, seq, adj, sparse=False):
        weight = tlx.tile(tlx.expand_dims(self.weight, 0), [seq.shape[0], 1, 1])
        seq_fts = tlx.matmul(seq, weight)
        if sparse:
            indices, values, shape = adj
            out = tlx.expand_dims(
                tlx.sparse_dense_matmul(indices, values, shape, tlx.squeeze(seq_fts, 0)), 
                0
            )
        else:
            out = tlx.matmul(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return out


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, activation='prelu', dropout=0, nlayers=1, with_bn=False, with_res=False):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.act = nn.ReLU() if activation.lower() == 'relu' else nn.PReLU()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(nfeat, nhid))
        if with_bn:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(num_features=nhid))
        for _ in range(nlayers-1):
            self.layers.append(GraphConvolution(nhid, nhid))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(num_features=nhid))
        self.with_bn = with_bn
        self.with_res = with_res
        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, GraphConvolution):
                tlx.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    tlx.nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

    def forward(self, x, adj):
        if self.with_res:
            prev_x = 0
            for ix, layer in enumerate(self.layers):
                x = layer(x + prev_x, adj)
                x = self.bns[ix](x) if self.with_bn else x
                x = self.act(x)
                if ix != len(self.layers) - 1:
                    x = tlx.nn.Dropout(self.dropout)(x)
                prev_x = x
        else:
            for ix, layer in enumerate(self.layers):
                x = layer(x, adj)
                x = self.bns[ix](x) if self.with_bn else x
                x = self.act(x)
                if ix != len(self.layers) - 1:
                    x = tlx.nn.Dropout(self.dropout)(x)
        return x

    def batch_forward(self, seq, adj, sparse=False):
        x = seq
        for ix, layer in enumerate(self.layers):
            x = layer.batch_forward(x, adj, sparse)
            x = self.bns[ix](x) if self.with_bn else x
            x = self.act(x)
            if ix != len(self.layers) - 1:
                x = tlx.nn.Dropout(self.dropout)(x)
        return x

    def get_prob(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.act(x)
                x = tlx.nn.Dropout(self.dropout)(x)
        return tlx.log_softmax(x, axis=1)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return tlx.reduce_mean(seq, axis=1)
        else:
            msk = tlx.expand_dims(msk, -1)
            return tlx.reduce_sum(seq * msk, axis=1) / tlx.reduce_sum(msk)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.shape[-2] + tensor.shape[-1]))
        tlx.nn.init.uniform_(tensor, -stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tlx.nn.init.constant_(tensor, 0)


if __name__ == "__main__":
    encoder = GCN(10, nhid=10)
