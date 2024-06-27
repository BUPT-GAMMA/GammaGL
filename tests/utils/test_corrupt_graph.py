import numpy as np
from gammagl.data import Graph
import torch
from gammagl.utils.corrupt_graph import dfde_norm_g

def test_corrupt_graph():
# 构造测试数据
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]])
    feat = torch.tensor([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2]])
   
    feat_drop_rate = 0.5
    drop_edge_rate = 0.5

    # 执行函数
    new_g = dfde_norm_g(edge_index, feat, feat_drop_rate, drop_edge_rate)

# 测试用例

    # 检查返回对象是否为 Graph 类型
    assert isinstance(new_g, Graph)

    # 检查节点数量是否正确
    assert new_g.num_nodes == 4

    # 检查边索引是否正确添加自环
    print(new_g.edge_index)
    print(new_g.x)
    print(new_g.edge_weight)

    assert (new_g.edge_index == torch.tensor([[0, 1, 1, 2, 2, 3, 0, 1, 2, 3],
                                           [1, 0, 2, 1, 3, 2, 0, 1, 2, 3]])).all()

    # # 检查特征矩阵是否正确丢弃部分特征
    # assert (new_g.x == np.array([[0.1, 0. , 0. , 0. ],
    #                               [0.4, 0. , 0. , 0. ],
    #                               [0. , 0.8, 0. , 1.2],
    #                               [0. , 0. , 0.9, 0. ]])).all()

    # # 检查边权重是否正确计算
    # assert (new_g.edge_weight == np.array([0.25, 0.25, 0.25, 0.25, 0.25,
    #                                        0.25, 0.25, 0.25, 0.25, 0.25])).all()



