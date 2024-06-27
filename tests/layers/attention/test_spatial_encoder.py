import torch
from tensorlayerx import nn
import tensorlayerx as tlx
from gammagl.layers.attention.spatial_encoder import SpatialEncoding
def test_spatial_encoder():
    # 创建一个示例 SpatialEncoding 实例
    encoder = SpatialEncoding(max_path_distance=5)

    # 构造测试数据
    x = torch.randn(32, 64)  # 32个节点，每个节点特征维度为64
    paths = {
        0: {1: [0, 1, 2]},   # 0 -> 1 的路径包含边的索引
        1: {2: [1, 2]},      # 1 -> 2 的路径包含边的索引
        2: {}                # 2 -> 任何节点没有路径
    }

    # 执行前向传播
    spatial_matrix = encoder(x, paths)

    # 检查输出的形状和数据类型是否正确
    assert isinstance(spatial_matrix, torch.Tensor)
    assert spatial_matrix.shape == (32, 32)  # 输出形状应与输入节点数量对应

