import tensorlayerx as tlx
from tensorlayerx import nn
from gammagl.layers.conv import MessagePassing

class GNNPPConv(MessagePassing):
    def __init__(self,in_dim,out_dim,dropout=0.05,residual=True,ffn=True):
        super().__init__()
        self.out_dim=out_dim
        self.A=nn.Linear(in_features=in_dim,out_features=out_dim)
        self.B=nn.Linear(in_features=in_dim,out_features=out_dim)
        self.C=nn.Linear(in_features=in_dim,out_features=out_dim)
        self.D=nn.Linear(in_features=in_dim,out_features=out_dim)
        self.E=nn.Linear(in_features=in_dim,out_features=out_dim)
        self.residual=residual
        self.ffn=ffn
        self.bn_node_x=nn.BatchNorm1d(num_features=out_dim)
        self.bn_edge_e=nn.BatchNorm1d(num_features=out_dim)
        self.dropout_x=nn.Dropout(dropout)
        self.dropout_e=nn.Dropout(dropout)
        if ffn:
            self.norm1_local=nn.BatchNorm1d(num_features=out_dim)
            self.ff_linear1=nn.Linear(in_features=out_dim,out_features=out_dim*2)
            self.ff_linear2=nn.Linear(in_features=out_dim*2,out_features=out_dim)
            self.ff_dropout1=nn.Dropout(dropout)
            self.ff_dropout2=nn.Dropout(dropout)
            self.norm2=nn.BatchNorm1d(num_features=out_dim)

    def message(self,Bx,Dx,Ex,Ce,edge_index):
        src=edge_index[0]
        dst=edge_index[1]
        e_ij=Dx[dst]+Ex[src]+Ce
        sigma_ij=tlx.sigmoid(e_ij)
        msg_x=sigma_ij*Bx[src]
        return tlx.concat([msg_x,sigma_ij],axis=1)

    def _ff_block(self,x):
        x=self.ff_dropout1(tlx.relu(self.ff_linear1(x)))
        x=self.ff_dropout2(self.ff_linear2(x))
        return x

    def forward(self,batch):
        x=batch.x
        e=batch.edge_attr
        edge_index=batch.edge_index
        if self.residual:
            x_in=x
            e_in=e
        Ax=self.A(x)
        Bx=self.B(x)
        Ce=self.C(e)
        Dx=self.D(x)
        Ex=self.E(x)
        aggregated=self.propagate(
            x=x,
            edge_index=edge_index,
            Bx=Bx,
            Dx=Dx,
            Ex=Ex,
            Ce=Ce,
            num_nodes=x.shape[0],
            aggr='sum'
        )
        numerator=aggregated[:,:self.out_dim]
        denominator=aggregated[:,self.out_dim:]
        x=Ax+numerator/(denominator+1e-6)
        src=edge_index[0]
        dst=edge_index[1]
        e=Dx[dst]+Ex[src]+Ce
        x=self.bn_node_x(x)
        e=self.bn_edge_e(e)
        x=tlx.relu(x)
        e=tlx.relu(e)
        x=self.dropout_x(x)
        e=self.dropout_e(e)
        if self.residual:
            x=x_in+x
            e=e_in+e
        if self.ffn:
            x=self.norm1_local(x)
            x=x+self._ff_block(x)
            x=self.norm2(x)
        batch.x=x
        batch.edge_attr=e
        return batch