import tensorlayerx as tlx
from gammagl.layers.conv import gcn_conv
from tensorlayerx.nn.layers.activation import LogSoftmax

class GraphSAINTModel(tlx.nn.Module):
    def __init__(self,in_channels,n_hiddens,out_channels,p_dropout):
        super(GraphSAINTModel,self).__init__()
        self.convs=tlx.nn.ModuleList()
        # relu and dropout
        self.relu = tlx.ReLU()
        self.dropout = tlx.layers.Dropout(p_dropout)
        # convs
        self.convs.append(gcn_conv(in_channels,n_hiddens))
        self.convs.append(gcn_conv(n_hiddens,n_hiddens))
        self.convs.append(gcn_conv(n_hiddens,n_hiddens))
        # output
        self.linear=tlx.nn.Linear(3*n_hiddens,out_channels)

    def forward(self,x,edge_index,edge_weight,num_nodes):
        x1 = self.convs[0](x,edge_index,edge_weight,num_nodes)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.convs[1](x1,edge_index,edge_weight,num_nodes)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x3 = self.convs[2](x2,edge_index,edge_weight,num_nodes)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x = tlx.concat([x1,x2,x3],axis=-1)
        x = self.linear(x)
        return LogSoftmax(x)
    
    def set_aggr(self,aggr):
        for i in range(len(self.convs)):
            self.convs[i].aggr =aggr 
