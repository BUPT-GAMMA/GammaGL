from typing import Union

# import torch
# from torch import nn
from gammagl.data import Graph
import tensorlayerx as tlx

from ..graphormer.functional import shortest_path_distance, batched_shortest_path_distance
from ..graphormer.layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding

class Graphormer(tlx.nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int):
        """
        初始化Graphormer模型。

        :param num_layers: Graphormer层数
        :param input_node_dim: 节点特征的输入维度
        :param node_dim: 节点特征的隐藏维度
        :param input_edge_dim: 边特征的输入维度
        :param edge_dim: 边特征的隐藏维度
        :param output_dim: 输出节点特征的数量
        :param n_heads: 注意力头的数量
        :param max_in_degree: 节点的最大入度
        :param max_out_degree: 节点的最大出度
        :param max_path_distance: 两个节点之间的最大路径距离
        """
        super().__init__()

        # 设置模型参数
        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

        # 定义输入线性层
        self.node_in_lin = tlx.nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = tlx.nn.Linear(self.input_edge_dim, self.edge_dim)

        # 初始化中心性编码和空间编码模块
        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        # 创建Graphormer编码器层
        self.layers = tlx.nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        # 定义输出线性层
        self.node_out_lin = tlx.nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Union[Graph]):
        """
        Graphormer模型的前向传播。

        :param data: 输入图或图批次
        :return: torch.Tensor，输出节点嵌入
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        # 根据输入类型计算最短路径（单个图或批量图）
        if type(data) == Graph:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        # 应用输入线性变换
        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        # 应用中心性编码和空间编码
        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)


        # 通过Graphormer编码器层进行传播
        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        # 应用输出线性变换
        x = self.node_out_lin(x)

        return x
