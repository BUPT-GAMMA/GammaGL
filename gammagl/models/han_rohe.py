import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
from gammagl.layers.conv.gatconv_rohe import GATConv

class SemanticAttention(tlx.nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = tlx.nn.Sequential(
            tlx.layers.Linear(in_features=in_size, out_features=hidden_size),
            tlx.layers.Tanh(),
            tlx.layers.Linear(in_features=hidden_size, out_features=1, b_init=None)
        )

    def forward(self, z):
        w = tlx.reduce_mean(self.project(z), axis=1)
        beta = tlx.softmax(w, axis=0)
        beta = tlx.expand_dims(beta, axis=-1)
        return tlx.reduce_sum(beta * z, axis=0)

class HANConv(tlx.nn.Module):
    def __init__(self, in_channels, out_channels, metadata, num_heads, dropout_rate, settings):
        super(HANConv, self).__init__()
        # 如果in_channels是整数，假设它是隐藏层维度，直接传递
        if isinstance(in_channels, int):
            self.gat_layers = tlx.nn.ModuleDict({
                '__'.join(edge_type): GATConv(in_channels, out_channels, num_heads, dropout_rate, settings=settings[edge_type])
                for edge_type in metadata[1]
            })
        # 否则，处理字典，按节点类型传递
        else:
            self.gat_layers = tlx.nn.ModuleDict({
                '__'.join(edge_type): GATConv(in_channels[edge_type[0]], out_channels, num_heads, dropout_rate, settings=settings[edge_type])
                for edge_type in metadata[1]
            })
        
        self.semantic_attention = SemanticAttention(out_channels * num_heads)
        self.metadata = metadata

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        out_dict = {key: [] for key in x_dict.keys()}
        
        # 对每种边类型进行遍历，应用GATConv
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if len(edge_index) > 0 and src_type in x_dict:  # 确保存在边
                out = self.gat_layers['__'.join(edge_type)](x_dict[src_type], edge_index, num_nodes_dict[dst_type])
                out_dict[dst_type].append(out)

        # 堆叠并应用语义注意力机制
        for node_type, outs in out_dict.items():
            if outs:  # 确保存在输出
                out_dict[node_type] = self.semantic_attention(tlx.stack(outs))
            else:
                # 如果 outs 为空，为该节点类型初始化一个零张量
                out_dict[node_type] = tlx.zeros((num_nodes_dict[node_type], self.gat_layers[list(self.gat_layers.keys())[0]].out_channels * self.gat_layers[list(self.gat_layers.keys())[0]].num_heads))

        return out_dict


class HAN(tlx.nn.Module):
    def __init__(self, metadata, in_channels, hidden_size, out_size, num_heads, dropout_rate, settings):
        super(HAN, self).__init__()
        self.layers = tlx.nn.ModuleList()

        # 第一层使用字典形式的 in_channels
        self.layers.append(HANConv(in_channels, hidden_size, metadata, num_heads[0], dropout_rate, settings[0]))

        # 后续层次中，使用前一层的输出维度
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANConv(hidden_size * num_heads[l-1], hidden_size, metadata, num_heads[l], dropout_rate, settings[l])
            )
        
        self.predict = tlx.layers.Linear(in_features=hidden_size * num_heads[-1], out_features=out_size)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, num_nodes_dict)
        out = {node_type: self.predict(x) for node_type, x in x_dict.items()}
        return out
