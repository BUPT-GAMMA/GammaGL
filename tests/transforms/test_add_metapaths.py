from typing import List

import tensorlayerx as tlx
from gammagl.data import HeteroGraph
from gammagl.transforms import BaseTransform
from gammagl.typing import EdgeType
from gammagl.transforms import AddMetaPaths

def test_add_metapaths():
    edge_index_author_paper = tlx.convert_to_tensor([[0, 1, 2], [2, 3, 4]], dtype=tlx.int64)
    edge_index_paper_conference = tlx.convert_to_tensor([[2, 3, 4], [5, 5, 5]], dtype=tlx.int64)

    data = HeteroGraph()
    data['author', 'to', 'paper'].edge_index = edge_index_author_paper
    data['paper', 'to', 'conference'].edge_index = edge_index_paper_conference

    # 定义元路径
    metapaths = [
        [('author', 'to', 'paper'), ('paper', 'to', 'conference')]
    ]

    # 应用 AddMetaPaths 转换器
    transform = AddMetaPaths(metapaths)
    data = transform(data)

    # 测试新边类型
    expected_new_edge_index = tlx.convert_to_tensor([[0, 1, 2], [5, 5, 5]], dtype=tlx.int64)

    assert ('author', 'metapath_0', 'conference') in data.edge_types
    assert tlx.convert_to_numpy(data['author', 'metapath_0', 'conference'].edge_index).shape == (2, 3)
    assert tlx.convert_to_numpy(data['author', 'metapath_0', 'conference'].edge_index).tolist() == expected_new_edge_index.tolist()

    # 测试 metapath_dict
    assert ('author', 'metapath_0', 'conference') in data.metapath_dict
    assert data.metapath_dict[('author', 'metapath_0', 'conference')] == [('author', 'to', 'paper'), ('paper', 'to', 'conference')]

   
