import tensorlayerx as tlx
import numpy as np
import networkx as nx
import metis

def partition_patch(graph, n_patches, load_path=None):

    if load_path is not None:
        # 使用 numpy 加载数据，假设数据是保存为 .npy 文件
        patch = np.load(load_path)
        patch = tlx.convert_to_tensor(patch, dtype=tlx.int64)
    else:
        if n_patches == 1:
            patch = np.arange(graph.num_nodes + 1)
            patch = tlx.convert_to_tensor(patch, dtype=tlx.int64)
            patch = tlx.expand_dims(patch, axis=0)
        else:
            patch = metis_partition(g=graph, n_patches=n_patches)

        print('metis done!!!')

    print('patch done!!!')

    # Graph update operations
    
    # torch版本的pad,可对比看是否使用正确
    graph.num_nodes += 1
    '''
    torch版本
    graph.x = F.pad(graph.x, [0, 0, 0, 1])
    # label = F.pad(label, [0, 1])
    graph.y = F.pad(graph.y, [0, 1])
    '''
        # 对x进行padding
    x_shape = graph.x.shape
    padded_x = np.pad(tlx.convert_to_numpy(graph.x), 
                      pad_width=((0, 1), (0, 0)), 
                      mode='constant',
                      constant_values=0)
    graph.x = tlx.convert_to_tensor(padded_x)
    
    # 对y进行padding
    padded_y = np.pad(tlx.convert_to_numpy(graph.y),
                      pad_width=(0, 1),
                      mode='constant',
                      constant_values=0)
    graph.y = tlx.convert_to_tensor(padded_y)
    
    return patch

def metis_partition(g, n_patches=50):

    if g.num_nodes < n_patches:
        # 如果节点数小于需要的分割数，则直接随机分配
        membership = np.random.permutation(n_patches)
        membership = tlx.convert_to_tensor(membership, dtype=tlx.int64)
    else:
        # 如果节点数大于或等于分割数，使用 METIS 进行分割
        adjlist = g.edge_index.T  # 获取边的邻接列表
        G = nx.Graph()  # 创建一个空的无向图
        G.add_nodes_from(np.arange(g.num_nodes))  # 添加节点
        G.add_edges_from(adjlist.tolist())  # 添加边
        
        # 使用 METIS 分割图
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    # 确保每个节点的归属部分数量不小于节点数
    assert len(membership) >= g.num_nodes
    membership = tlx.convert_to_tensor(membership[:g.num_nodes], dtype=tlx.int64)

    patch = []  # 用于存储每个分割部分的节点索引
    max_patch_size = -1  # 用于记录最大的子图大小

    for i in range(n_patches):
        patch.append(list())
        # 使用 numpy 的 np.where 来代替 torch.where
        patch[-1] = np.where(membership == i)[0].tolist()  # 归属到 i 号部分的节点
        max_patch_size = max(max_patch_size, len(patch[-1]))  # 更新最大的子图大小

    # 填充所有子图，使它们的大小一致
    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [g.num_nodes] * (max_patch_size - l)

    patch = tlx.convert_to_tensor(patch, dtype=tlx.int64)  # 返回最终的分割结果

    return patch
