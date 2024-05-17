import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn

class metapathSpecificGCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(metapathSpecificGCN, self).__init__()
        self.fc = nn.Linear(in_features=in_ft, out_features=out_ft, W_init="he_normal")
        self.act = nn.LeakyReLU()

        if bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, out_ft), init=initor)
        else:
            self.register_parameter('bias', None)


    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = tlx.matmul(adj, seq_fts) 
        if self.bias is not None:
            out += self.bias
        return self.act(out)
    
