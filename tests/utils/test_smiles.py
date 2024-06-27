from typing import List, Dict, Any
import tensorlayerx as tlx
from rdkit import Chem, RDLogger
from gammagl.data import Graph
from gammagl.utils.smiles import from_smiles


# 假设我们已经定义好了 x_map 和 e_map 以及 from_smiles 函数
# 从之前的代码中我们知道这些字典的结构

# 定义测试函数
def test_from_smiles():
    # 定义一个简单的 SMILES 字符串，例如乙醇 (ethanol)
    smiles = 'CCO'  # 乙醇的 SMILES 表示

    # 调用 from_smiles 函数
    graph = from_smiles(smiles)

    # 检查节点特征
    expected_node_features = [
        [6, 0, 4, 5, 3, 0, 4, 0, 0],  # 第一个碳原子
        [6, 0, 4, 5, 2, 0, 4, 0, 0],  # 第二个碳原子
        [8, 0, 2, 5, 1, 0, 4, 0, 0],  # 氧原子
    ]
    expected_node_features = tlx.convert_to_tensor(expected_node_features, dtype=tlx.int64)

    # 检查边索引
    expected_edge_indices = [
        [0, 1], [1, 0],  # C-C
        [1, 2], [2, 1],  # C-O
    ]
    expected_edge_indices = tlx.convert_to_tensor(expected_edge_indices).T

    # 检查边特征
    expected_edge_features = [
        [1, 0, 0], [1, 0, 0],  # C-C
        [1, 0, 0], [1, 0, 0],  # C-O
    ]
    expected_edge_features = tlx.convert_to_tensor(expected_edge_features, dtype=tlx.int64)

    # 断言检查
    assert tlx.convert_to_numpy(graph.x).tolist() == tlx.convert_to_numpy(expected_node_features).tolist()
       
    assert tlx.convert_to_numpy(graph.edge_index).tolist() == tlx.convert_to_numpy(expected_edge_indices).tolist()
      
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == tlx.convert_to_numpy(expected_edge_features).tolist()
    
