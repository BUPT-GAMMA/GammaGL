import tensorlayerx as tlx
from gammagl.data import HeteroGraph
from gammagl.transforms import BaseTransform

def test_base_transform():
    # 示例子类，实现具体的转换逻辑
    class AddSelfLoops(BaseTransform):
        def __call__(self, data):
            for edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                num_nodes = max(int(tlx.reduce_max(edge_index[0])), int(tlx.reduce_max(edge_index[1]))) + 1
                self_loops = tlx.convert_to_tensor(range(num_nodes), dtype=tlx.int64)
                self_loops = tlx.stack([self_loops, self_loops], axis=0)
                data[edge_type].edge_index = tlx.concat([edge_index, self_loops], axis=1)
            return data

    # 创建一个简单的异构图
    edge_index_author_paper = tlx.convert_to_tensor([[0, 1], [1, 0]], dtype=tlx.int64)
    data = HeteroGraph()
    data['author', 'to', 'paper'].edge_index = edge_index_author_paper

    # 应用 AddSelfLoops 转换器
    transform = AddSelfLoops()
    data = transform(data)

    # 预期的边索引
    expected_edge_index = tlx.convert_to_tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=tlx.int64)

    # 断言测试
    assert ('author', 'to', 'paper') in data.edge_types
    assert tlx.convert_to_numpy(data['author', 'to', 'paper'].edge_index).shape == (2, 4)
    assert tlx.convert_to_numpy(data['author', 'to', 'paper'].edge_index).tolist() == tlx.convert_to_numpy(expected_edge_index).tolist()




