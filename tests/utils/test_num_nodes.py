import numpy as np
import torch
from gammagl.utils.num_nodes import maybe_num_nodes, maybe_num_nodes_dict
from copy import copy
from gammagl.utils.check import check_is_numpy
import tensorlayerx as tlx

def test_num_nodes():
# 定义边索引字典
    edge_index_dict = {
        ('social', 'user-user'): np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        ('knowledge', 'concept-concept'): np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    }

    # 验证 maybe_num_nodes 函数
    edge_index_tensor = torch.tensor(edge_index_dict[('social', 'user-user')])  # 边索引的张量表示
    num_nodes = maybe_num_nodes(edge_index_tensor)  # 计算节点数量
    assert num_nodes == 3

    # 验证 maybe_num_nodes_dict 函数
    num_nodes_dict = maybe_num_nodes_dict(edge_index_dict)  # 计算每种图类型的节点数量
    expected_num_nodes_dict = {'social': 3, 'knowledge': 4}
    assert num_nodes_dict == expected_num_nodes_dict
