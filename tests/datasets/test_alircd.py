from gammagl.datasets.alircd import AliRCD


root="./data"
def alircd():
    data_alircd=AliRCD(root)
    assert len(data_alircd)==1
    assert data_alircd.num_node_features=={'b': 256, 'f': 256, 'a': 256, 'item': 256, 'c': 256, 'e': 256, 'd': 256}
    assert data_alircd.num_features=={'b': 256, 'f': 256, 'a': 256, 'item': 256, 'c': 256, 'e': 256, 'd': 256}
    #assert data_alircd.num_classes报错：'HeteroGraph' has no attribute 'y'
    assert data_alircd.data.num_edges==157814864
    assert data_alircd.data.num_nodes==13806619
    assert data_alircd.data.metadata()==(['b', 'f', 'a', 'item', 'c', 'e', 'd'], [('b', 'A_1', 'item'), ('f', 'B', 'item'), ('a', 'G_1', 'f'), ('f', 'G', 'a'), ('a', 'H_1', 'e'), ('f', 'C', 'd'), ('f', 'D', 'c'), ('c', 'D_1', 'f'), ('f', 'F', 'e'), ('item', 'B_1', 'f'), ('item', 'A', 'b'), ('e', 'F_1', 'f'), ('e', 'H', 'a'), ('d', 'C_1', 'f')])
    assert data_alircd.data.keys==['test_idx', 'edge_index', 'y', 'x', 'val_idx', 'n_id', 'num_nodes', 'train_idx']


# if __name__ == '__main__':
#     test_alircd()
#如果某次下载被中断将导致该方法再次运行时会报错，可能是缺少一些文件体积的判断？
"""
output数据格式:
Num of total nodes: 13806619
Node_types: ['b', 'f', 'a', 'item', 'c', 'e', 'd']
Node_type Num Num_lack_feature:
b 45602 0
f 1063739 0
a 204128 0
item 11933366 0
c 178777 0
e 379320 0
d 1687 0
Num of true labeled nodes:85562
data_alircd.data
HeteroGraph(
  b={
    x=[45602, 256],
    num_nodes=45602,
    n_id=[45602]
  },
  f={
    x=[1063739, 256],
    num_nodes=1063739,
    n_id=[1063739]
  },
  a={
    x=[204128, 256],
    num_nodes=204128,
    n_id=[204128]
  },
  item={
    x=[11933366, 256],
    num_nodes=11933366,
    y=[11933366],
    train_idx=[68449],
    val_idx=[8556],
    test_idx=[8557],
    n_id=[11933366]
  },
  c={
    x=[178777, 256],
    num_nodes=178777,
    n_id=[178777]
  },
  e={
    x=[379320, 256],
    num_nodes=379320,
    n_id=[379320]
  },
  d={
    x=[1687, 256],
    num_nodes=1687,
    n_id=[1687]
  },
  (b, A_1, item)={ edge_index=[2, 3781140] },
  (f, B, item)={ edge_index=[2, 11683940] },
  (a, G_1, f)={ edge_index=[2, 59208411] },
  (f, G, a)={ edge_index=[2, 59208411] },
  (a, H_1, e)={ edge_index=[2, 1194499] },
  (f, C, d)={ edge_index=[2, 521220] },
  (f, D, c)={ edge_index=[2, 1269499] },
  (c, D_1, f)={ edge_index=[2, 1269499] },
  (f, F, e)={ edge_index=[2, 1248723] },
  (item, B_1, f)={ edge_index=[2, 11683940] },
  (item, A, b)={ edge_index=[2, 3781140] },
  (e, F_1, f)={ edge_index=[2, 1248723] },
  (e, H, a)={ edge_index=[2, 1194499] },
  (d, C_1, f)={ edge_index=[2, 521220] }
)

"""