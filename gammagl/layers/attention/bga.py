import tensorlayerx as tlx
from .bga_layer import BGALayer


class BGA(tlx.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=True, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(BGA, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = tlx.nn.Dropout(p=dropout1)
        # 初始化线性层，直接使用nn.Linear
        self.attribute_encoder = tlx.nn.Linear(in_features=in_channels, out_features=hidden_channels)
        self.BGALayers = tlx.nn.ModuleList()
        for _ in range(0, layers):
            self.BGALayers.append(
                BGALayer(n_head, hidden_channels, use_patch_attn, dropout=dropout2))
        self.classifier = tlx.nn.Linear(in_features=hidden_channels, out_features=out_channels)
        self.attn = []

    def forward(self, x, patch, need_attn=False):
        patch_mask = tlx.cast(patch != self.num_nodes - 1, dtype=tlx.float32)
        patch_mask = tlx.expand_dims(patch_mask, axis=-1)
        attn_mask = tlx.cast(tlx.matmul(patch_mask, tlx.transpose(patch_mask, perm=[0, 2, 1])), dtype=tlx.int32)

        x = tlx.relu(self.attribute_encoder(x))
        
        for i in range(0, self.layers):
            x = self.BGALayers[i](x, patch, attn_mask, need_attn)
            if need_attn:
                self.attn.append(self.BGALayers[i].attn)
        x = self.dropout(x)
        x = self.classifier(x)
        return x