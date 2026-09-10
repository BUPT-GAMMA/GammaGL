import tensorlayerx as tlx
from tensorlayerx import nn
from gammagl.layers.conv import GNNPPConv
from gammagl.layers.pool import global_sum_pool

class TypeDictNodeEncoder(nn.Module):
    def __init__(self,emb_dim,num_types,rwse_dim=20,rwse_out_dim=28):
        super().__init__()
        if num_types<1:
            raise ValueError(f"Invalid 'num_types': {num_types}")
        node_emb_dim=emb_dim-rwse_out_dim
        self.encoder=nn.Embedding(
            num_embeddings=num_types,
            embedding_dim=node_emb_dim
        )
        self.rwse_bn=nn.BatchNorm1d(num_features=rwse_dim)
        self.rwse_encoder=nn.Linear(
            in_features=rwse_dim,
            out_features=rwse_out_dim
        )

    def forward(self,batch):
        node_indices=tlx.cast(
            tlx.reshape(batch.x[:,0],(-1,)),
            tlx.int64
        )
        x_type=self.encoder(node_indices)
        rwse=batch.pestat_RWSE
        rwse=self.rwse_bn(rwse)
        rwse=self.rwse_encoder(rwse)
        batch.x=tlx.concat([x_type,rwse],axis=1)
        return batch

class TypeDictEdgeEncoder(nn.Module):
    def __init__(self,emb_dim,num_types):
        super().__init__()
        if num_types<1:
            raise ValueError(f"Invalid 'num_types': {num_types}")
        self.encoder=nn.Embedding(
            num_embeddings=num_types,
            embedding_dim=emb_dim
        )
        self.encoder2=nn.Linear(
            in_features=20,
            out_features=emb_dim
        )

    def forward(self,batch):
        edge_type=tlx.cast(
            batch.edge_attr[:,0],
            tlx.int64
        )
        edge_rwse=batch.edge_attr[:,1:]
        edge_type_emb=self.encoder(edge_type)
        edge_rwse_emb=self.encoder2(edge_rwse)
        batch.edge_attr=edge_type_emb+edge_rwse_emb
        return batch

class GNNPPModel(nn.Module):
    def __init__(
        self,
        num_node_types=21,
        num_edge_types=4,
        emb_dim=70,
        num_layers=9,
        dropout=0.05,
        residual=True,
        ffn=True
    ):
        super().__init__()
        self.node_encoder=TypeDictNodeEncoder(
            emb_dim=emb_dim,
            num_types=num_node_types
        )
        self.edge_encoder=TypeDictEdgeEncoder(
            emb_dim=emb_dim,
            num_types=num_edge_types
        )
        self.layers=nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GNNPPConv(
                    in_dim=emb_dim,
                    out_dim=emb_dim,
                    dropout=dropout,
                    residual=residual,
                    ffn=ffn
                )
            )
        self.head1=nn.Linear(
            in_features=emb_dim,
            out_features=emb_dim//2
        )
        self.head2=nn.Linear(
            in_features=emb_dim//2,
            out_features=emb_dim//4
        )
        self.head3=nn.Linear(
            in_features=emb_dim//4,
            out_features=1
        )

    def forward(self,batch):
        batch=self.node_encoder(batch)
        batch=self.edge_encoder(batch)
        for layer in self.layers:
            batch=layer(batch)
        graph_repr=global_sum_pool(batch.x,batch.batch)
        graph_repr=self.head1(graph_repr)
        graph_repr=tlx.relu(graph_repr)
        graph_repr=self.head2(graph_repr)
        graph_repr=tlx.relu(graph_repr)
        return self.head3(graph_repr)