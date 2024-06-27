from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx
import numpy as np
from gammagl.layers.attention.edge_encoder import EdgeEncoding
from gammagl.layers.attention.graphormer_layer import GraphormerLayer
import torch
def test_graphormer_layer():
    # 创建一个示例 GraphormerLayer 实例
    layer = GraphormerLayer(node_dim=64, edge_dim=16, n_heads=8, max_path_distance=5)

    # 构造测试数据
    x = torch.randn(32, 64)  # 32个节点，每个节点特征维度为64
    edge_attr = torch.randn(64, 16)  # 64条边，每条边特征维度为16
    b = torch.randn(32, 32)  # 示例中的 b，维度需匹配实际情况
    edge_paths = {
        0: {1: [0, 1, 2]},   # 0 -> 1 的路径包含边的索引
        1: {2: [1, 2]},      # 1 -> 2 的路径包含边的索引
        2: {}                # 2 -> 任何节点没有路径
    }
    ptr = None  # 一个指向节点范围的指针，需要根据实际情况设置

    # 执行前向传播
    output = layer(x, edge_attr, b, edge_paths, ptr)

    # 检查输出的形状和数据类型是否正确
    assert isinstance(output, torch.Tensor)
    assert output.shape == (32, 64)  # 输出形状应与输入节点特征维度一致

    
