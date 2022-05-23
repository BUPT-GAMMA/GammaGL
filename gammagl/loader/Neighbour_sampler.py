from gammagl.sample import sample_subset
import numpy as np


class subg(object):
    def __init__(self, edge, size):
        '''
        Parameters
        ----------
        edge: 
            Renumbered subgraph
        size: int , int
            store all_node's size and dst_node's size, get dst_node by slicing
        '''
        self.edge = tlx.convert_to_tensor(edge)
        self.size = size
    def to(self, device):
        self.edge = self.edge.to(device)


from tensorlayerx.dataflow import DataLoader
import tensorlayerx as tlx


class Neighbor_Sampler(DataLoader):
    def __init__(self, edge_index,
                 dst_nodes,
                 sample_lists, indptr=None, replace=False, is_sorted=False,
                 **kwargs):
        '''
        Parameters
        ----------
        edge_index: np.ndarray
            shape = [2, E] ,[src, dst], dst is sortted
        dst_nodes: np.ndarray
            target nodes, which need to sample its neighbour
        sample_lists: list
            sample number
        indptr:  np.ndarray
            csr matrix's parameters, use for sample, default = None
        replace: bool
            sample with or without replacement default = False
        '''
        self.sample_list = sample_lists
        if indptr is None:
            self.rowptr = np.concatenate(([0], np.bincount(edge_index[1]).cumsum()))
        else:
            self.rowptr = indptr
        if is_sorted == False:
            ind = np.argsort(edge_index[1], axis=0)
            edge_index = np.array(edge_index.T[ind]).T
        self.edge_index = edge_index.T
        self.dst_nodes = dst_nodes
        self.replace = replace

        super(Neighbor_Sampler, self).__init__(
            dst_nodes.tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        adjs = []
        dst_node = batch
        cur = 0
        for sample in self.sample_list:
            dst_node = np.array(dst_node, dtype=np.int64)
            dst_node, size, edge = sample_subset(sample, dst_node, self.rowptr, self.edge_index,
                                                 self.replace)
            sg = subg(edge.T, size)
            adjs.append(sg)
        # Finally, dst_node will all nodes obtained by sampling
        all_node = dst_node
        # return adjs[::-1], all_node
        return [tlx.convert_to_tensor(batch), adjs[::-1], tlx.convert_to_tensor(all_node)]

    def test(self, batch):
        return batch





class Neighbor_Sampler_python(DataLoader):
    '''
    python's sampler
    it equals Neighbor_Sampler, but it will cost too many times on sample per batch.
    it just use for test speed
    '''
    def __init__(self, edge_index, indptr,
                 dst_nodes,
                 sample_lists, replace,
                 **kwargs):

        self.sample_list = sample_lists
        # transpose [N,2]
        self.edge_index = tlx.transpose(edge_index).numpy()
        self.dst_nodes = dst_nodes
        self.rowptr = indptr.numpy()
        self.replace = replace
        super(Neighbor_Sampler_python, self).__init__(
            dst_nodes.numpy().tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        adjs = []
        dst_node = batch
        for sample in self.sample_list:
            dst_node = np.array(dst_node, dtype=np.int64)
            dst_node, size, edge = self.sample_subset(sample, dst_node, self.rowptr, self.edge_index)
            sg = subg(edge.T, size)
            adjs.append(sg)
        all_node = dst_node
        return [np.array(batch), adjs[::-1], all_node]

    def sample_subset(self, sample, dst_nodes, rowptr, edge_index):
        self.all_node = {}
        e_id = []
        # dst_node will always place first place
        for i, dst_node in enumerate(dst_nodes):
            self.all_node[dst_node] = i
        a = len(self.all_node)
        # sample function
        if sample == -1:
            for dst_node in dst_nodes:
                '''-1 is all neighbor sample'''
                st = rowptr[dst_node]
                len1 = rowptr[dst_node + 1] - st
                for i in range(st, st + len1):
                    if self.all_node.get(edge_index[i][0]) is None:
                        self.all_node[edge_index[i][0]] = len(self.all_node)
                    e_id.append(i)
        elif self.replace == False:
            for dst_node in dst_nodes:
                '''sample without replacement,
                    it has two situations
                    1. k >= num_sample
                    2. k < num_sample 
                    we will let situation 1 to use random.choice.
                    let situation 2 to put all neighbor node to sample list
                '''
                st = self.rowptr[dst_node]
                len1 = self.rowptr[dst_node + 1] - st
                if sample >= len1:
                    for i in range(st, st + len1):
                        if self.all_node.get(self.edge_index[i][0]) is None:
                            self.all_node[self.edge_index[i][0]] = len(self.all_node)
                        e_id.append(i)
                else:
                    ind = np.random.choice(np.arange(0, len1), sample, replace=False)
                    for i in ind:
                        if self.all_node.get(self.edge_index[st + i][0]) is None:
                            self.all_node[self.edge_index[st + i][0]] = len(self.all_node)
                        e_id.append(st + i)
        else:
            for dst_node in dst_nodes:
                '''sample with replacement
                    use random choice
                '''
                st = self.rowptr[dst_node]
                len1 = self.rowptr[dst_node + 1] - st
                ind = np.random.choice(np.arange(0, len1), sample, replace=True)
                for i in ind:
                    if self.all_node.get(self.edge_index[st + i][0]) is None:
                        self.all_node[self.edge_index[st + i][0]] = len(self.all_node)
                    e_id.append(st + i)
        new_edges = self.small_g(e_id)
        b = len(self.all_node)
        return list(self.all_node.keys()), (b, a), new_edges

    def small_g(self, e_id):
        '''

        Parameters
        ----------
        nodes:
            Nodes to be renumbered
        e_id: 
            Edge serial number to be renumbered

        Returns
        -------
        list
            Edge index after renumbering

        '''
        edges = self.edge_index[e_id]
        new_edge = []
        for (a, b) in edges:
            new_edge.append([self.all_node[a], self.all_node[b]])
        return np.array(new_edge)
