import tensorlayerx as tlx
import numpy as np
from gammagl.mpops import unsorted_segment_sum
from gammagl.utils import calc_gcn_norm


def test_calc_gcn_norm():
  
    edge_index = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ])
    
    # 节点数量
    num_nodes = 4
    weights = calc_gcn_norm(tlx.convert_to_tensor(edge_index), num_nodes)
    
    # 计算每个节点的度数
    degree = np.array([1, 2, 2, 1])
    
    # 计算度数的平方根倒数
    deg_inv_sqrt = np.power(degree, -0.5)
    
    # 预期的输出值 (根据GCN公式手动计算)
    expected_weights = np.array([
        deg_inv_sqrt[0] * deg_inv_sqrt[1],
        deg_inv_sqrt[1] * deg_inv_sqrt[0],
        deg_inv_sqrt[1] * deg_inv_sqrt[2],
        deg_inv_sqrt[2] * deg_inv_sqrt[1],
        deg_inv_sqrt[2] * deg_inv_sqrt[3],
        deg_inv_sqrt[3] * deg_inv_sqrt[2]
    ])

    # 将权重转换为numpy数组进行比较
    weights_np = tlx.convert_to_numpy(weights)
    print(weights_np)
    print(expected_weights)
    # 验证结果是否与预期输出匹配
    assert np.allclose(weights_np, expected_weights)


