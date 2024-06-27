from tensorlayerx import nn
from gammagl.utils import degree
import tensorlayerx as tlx
import numpy as np
from gammagl.layers.attention.edge_encoder import EdgeEncoding
import torch
def test_edge_encoder():
    # 创建一个示例 EdgeEncoding 实例
    edge_encoder = EdgeEncoding(edge_dim=3, max_path_distance=4)

    # 构造测试数据
    x = torch.randn(5, 10)  # 5个节点，每个节点特征维度为10
    edge_attr = torch.randn(10, 3)  # 10条边，每条边特征维度为3
    edge_paths = {
        0: {
            1: [0, 1, 2],   # 0 -> 1 的路径包含边的索引
            2: [0, 3]       # 0 -> 2 的路径包含边的索引
        },
        1: {
            2: [1, 2]       # 1 -> 2 的路径包含边的索引
        },
        2: {}              # 2 -> 任何节点没有路径
    }

    # 执行前向传播
    cij = edge_encoder(x, edge_attr, edge_paths)

    # 检查输出 cij 的形状和数据类型是否正确
    assert isinstance(cij, torch.Tensor)
    assert cij.shape == (5, 5)
    assert torch.all(torch.isfinite(cij))  # 确保输出没有 NaN 或 Infinity

