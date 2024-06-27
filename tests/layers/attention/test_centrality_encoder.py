
import tensorlayerx as tlx
from tensorlayerx import nn
from gammagl.utils import degree, mask_to_index
from gammagl.layers.attention.centrality_encoder import CentralityEncoding
def test_centrality_encoder():
    # 创建一个简单的图
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 0, 3, 4, 4], [1, 0, 0, 2, 0, 3, 2]], dtype=tlx.int64)
    x = tlx.convert_to_tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5]], dtype=tlx.float32)
    
    # 设置最大入度和最大出度
    max_in_degree = 3
    max_out_degree = 3
    node_dim = 2
    
    # 创建 CentralityEncoding 模块
    ce = CentralityEncoding(max_in_degree=max_in_degree, max_out_degree=max_out_degree, node_dim=node_dim)
    
    # 调用 forward 方法
    x_encoded = ce(x, edge_index)
    
    # 检查输出维度
    assert tlx.get_tensor_shape(x_encoded) == tlx.get_tensor_shape(x), "输出的形状应该与输入的形状相同"
    
    # 检查编码后的特征不为空
    assert not tlx.ops.is_nan(tlx.reduce_sum(x_encoded)), "输出不应包含 NaN 值"
    
    # 打印输出以检查值
    print("x_encoded:", x_encoded)


