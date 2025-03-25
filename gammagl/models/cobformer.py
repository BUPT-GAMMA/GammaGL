import tensorlayerx as tlx
from gammagl.models import GCNModel as GCN
from gammagl.layers.attention.bga import BGA

class CoBFormer(tlx.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 activation, gcn_layers: int, gcn_type: int, layers: int, n_head: int, dropout1=0.5, dropout2=0.1,
                 alpha=0.8, tau=0.5, gcn_use_bn=False, use_patch_attn=True):
        super(CoBFormer, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = activation
        self.dropout = tlx.layers.Dropout(p=dropout1)
        self.gcn = GCN(
            feature_dim=in_channels,
            hidden_dim=hidden_channels,
            num_class=out_channels,
            drop_rate=dropout1,
            num_layers=gcn_layers
        )
        self.bga = BGA(num_nodes, in_channels, hidden_channels, out_channels, layers, n_head,
                       use_patch_attn, dropout1, dropout2)
        self.attn = None

    def forward(self, x, patch, edge_index, edge_weight=None, num_nodes=None, need_attn=False):
        z1 = self.gcn(x, edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
        z2 = self.bga(x, patch, need_attn)
        if need_attn:
            self.attn = self.bga.attn

        return z1, z2

    def loss(self, pred1, pred2, label, mask):
        # 计算损失
        l1 = tlx.losses.softmax_cross_entropy_with_logits(pred1[mask], label[mask])
        l2 = tlx.losses.softmax_cross_entropy_with_logits(pred2[mask], label[mask])
        
        # 缩放预测值
        pred1_scaled = pred1 * self.tau
        pred2_scaled = pred2 * self.tau
        
        # 计算蒸馏损失, 使用softmax_cross_entropy_with_logits
        l3 = tlx.losses.softmax_cross_entropy_with_logits(pred1_scaled[~mask], tlx.nn.Softmax()(pred2_scaled)[~mask])
        l4 = tlx.losses.softmax_cross_entropy_with_logits(pred2_scaled[~mask], tlx.nn.Softmax()(pred1_scaled)[~mask])
        
        return self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)
