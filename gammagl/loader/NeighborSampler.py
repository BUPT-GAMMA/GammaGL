from tensorlayerx.dataflow import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
from gammagl.utils.sample import sample_subset


class subg(object):
    def __init__(self, edge, size):
        '''
        Args:
            edge: 从新编号过后的子图
            size: [int , int ]，存储all_node的数量以及dst_node的数量，主要切片来获取dst_node
        '''
        self.edge = edge
        self.size = size


class Mydata(Dataset):
    def __init__(self, idx):
        self.id = idx

    def __getitem__(self, item):
        return self.id[item]

    def __len__(self):
        return self.id.shape[0]


class Neighbor_Sampler(DataLoader):
    def __init__(self, edge_index,
                 dst_nodes,
                 sample_lists,
                 **kwargs):
        self.dstnode = Mydata(dst_nodes)
        self.sample_list = sample_lists
        # transpose [N,2]
        self.edge_index = np.array(edge_index, dtype=np.int64).T
        self.e_id = np.arange(0, edge_index.shape[1])

        csr = sp.csr_matrix((self.e_id, edge_index))
        self.rowptr = csr.indptr
        super(Neighbor_Sampler, self).__init__(
            self.dstnode, collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        adjs = []
        dst_node = batch
        for sample in self.sample_list:
            dst_node = np.array(dst_node, dtype=np.int64)
            self.rowptr = np.array(self.rowptr, dtype=np.int64)
            self.edge_index = np.array(self.edge_index, dtype=np.int64)

            dst_node, size, edge = sample_subset(sample, dst_node, self.rowptr, self.edge_index)

            sg = subg(edge.T, size)
            adjs.append(sg)
        # dst_node最终就是采样到最外层所用到的节点
        all_node = dst_node
        # return adjs[::-1], all_node
        return [np.array(batch), adjs[::-1], all_node]