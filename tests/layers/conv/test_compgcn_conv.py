import tensorlayerx as tlx
import numpy as np
import torch
from gammagl.layers.conv import MessagePassing

from gammagl.mpops import *
from gammagl.layers.conv.compgcn_conv import  CompConv
def test_compgcn_conv():
    # 创建一个示例 CompConv 实例
    conv = CompConv(in_channels=64, out_channels=128, num_relations=4)

    # 构造测试数据
    x = torch.randn(32, 64)  # 32个节点，每个节点特征维度为64
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # 4条边的起始节点索引
    edge_type = torch.tensor([0, 1, 2, 3])  # 每条边的类型
    ref_emb = torch.randn(4, 64)  # 参考嵌入的维度

    # 执行前向传播
    output, ref_emb = conv(x, edge_index, edge_type, ref_emb)

    # 检查输出的形状和数据类型是否正确
    assert isinstance(output, torch.Tensor)
    assert output.shape == (32, 128)  # 输出形状应为 128 维特征

    print("测试通过！")
