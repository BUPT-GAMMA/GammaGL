import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import sort_edge_index 
from gammagl.ops.sparse import ind2ptr
from gammagl.layers.conv import FusedGATConv
import numpy as np
def test_fusedgat_conv():
        # 创建一个示例 FusedGATConv 实例
    conv = FusedGATConv(in_channels=64, out_channels=128, heads=4, concat=True, dropout_rate=0.5)

    # 构造测试数据
    x = tlx.convert_to_tensor(np.random.randn(32, 64), dtype=tlx.float32)  # 32个节点，每个节点特征维度为64
    edge_index = tlx.convert_to_tensor(np.array([[0, 1, 1, 2], [1, 0, 2, 1]]), dtype=tlx.int32)  # 4条边的起始节点索引

    # 执行前向传播
    output = conv(x, edge_index)

    # 检查输出的形状和数据类型是否正确
    assert isinstance(output, tlx.Tensor)
    assert output.shape == (32, 128 * 4)  # 输出形状应为 512 维特征（4 个注意力头，每个头 128 维）
