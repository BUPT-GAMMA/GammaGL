import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.models import GCNModel

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        
        self.project = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=1)
        )
    
    def forward(self, x):
        w = self.project(x)
        beta = tlx.softmax(w, axis=1)
        return tlx.reduce_sum(beta * x, axis=1), beta

class SFGCNModel(nn.Module):
    def __init__(self, num_feat, num_class, num_hidden1, num_hidden2, dropout):
        super(SFGCNModel, self).__init__(name="SFGCN")
        
        self.SGCN1 = GCNModel(num_feat, num_hidden1, num_hidden2, dropout)
        self.SGCN2 = GCNModel(num_feat, num_hidden1, num_hidden2, dropout)
        self.CGCN = GCNModel(num_feat, num_hidden1, num_hidden2, dropout)
        
        self.dropout = dropout
        self.a = self._get_weights("a", shape=(num_hidden2, 1),
                                   init=tlx.initializers.xavier_uniform(gain=1.414))
        self.attention = Attention(num_hidden2)
        self.tanh = nn.Tanh()
        
        self.MLP = nn.Sequential(
            nn.Linear(in_features=num_hidden2, out_features=num_class),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x, edge_index_s, edge_index_f):
        emb1 = self.SGCN1(x, edge_index_s, None, None)
        com1 = self.CGCN(x, edge_index_s, None, None)
        emb2 = self.SGCN2(x, edge_index_f, None, None)
        com2 = self.CGCN(x, edge_index_f, None, None)
        Xcom = (com1 + com2) / 2
        
        # attention
        emb = tlx.stack([emb1, emb2, Xcom], axis=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        
        return output, att, emb1, com1, com2, emb2, emb