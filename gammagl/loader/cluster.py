import copy
import os.path as osp
import sys
from typing import Optional

import torch
from tensorlayerx.dataflow import Dataset,DataLoader
import scipy.sparse as sp
import numpy as np
from gammagl.sample import metis_partition,ind2ptr
import tensorlayerx as tlx

class ClusterData(Dataset):

    def __init__(self, data, num_parts: int, recursive: bool = False,
                 save_dir: Optional[str] = None, log: bool = True):

        assert data.edge_index is not None

        self.num_parts = num_parts

        recursive_str = '_recursive' if recursive else ''
        filename = f'partition_{num_parts}{recursive_str}.npy'
        path = osp.join(save_dir or '', filename)
        if save_dir is not None and osp.exists(path):
            #adj, partptr, perm = sp.load_npz(path)
            dic = tlx.files.load_npy_to_any(save_dir,filename)
            adj = dic['adj']
            partptr = dic['partptr']
            perm = dic['perm']

        else:
            if log:  # pragma: no cover
                print('Computing METIS partitioning...', file=sys.stderr)

            N, E = data.num_nodes, data.num_edges
            adj = {'row':data.edge_index[0],'col':data.edge_index[1],
                   'value':tlx.arange(0,E),'sparse_sizes':(N, N)}

            # do partition
            adj, partptr, perm = self.partition(data,num_parts, recursive)

            if save_dir is not None:
                #sp.save_npz((adj, partptr, perm), path)
                tlx.files.save_any_to_npy({'adj': adj,'partptr':partptr, 'perm':perm}, name = path)

            if log:  # pragma: no cover
                print('Done!', file=sys.stderr)

        self.data = self.__permute_data__(data, perm, adj)
        self.partptr = partptr
        self.perm = perm

    def partition(self, data, num_parts, recursive):

        assert num_parts >= 1
        if num_parts == 1:
            partptr = tlx.ops.convert_to_tensor([0, data.num_nodes])
            perm = tlx.arange(0, data.num_nodes)

        csr = sp.csr_matrix((np.arange(data.num_edges), data.edge_index.numpy()))
        rowptr = np.array(csr.indptr, dtype=np.int64)
        col = np.array(csr.indices, dtype=np.int64)


        cluster = metis_partition(rowptr, col, nparts=num_parts, recursive=recursive)

        #cluster, perm = self.cluster_sort(cluster, data.num_nodes,num_parts)
        if tlx.BACKEND == 'torch':
            cluster, perm = tlx.ops.convert_to_tensor(cluster).sort()
            csr = sp.csr_matrix((np.arange(data.num_edges), data.edge_index.numpy()))
            a = csr[perm.numpy()]
            out = a[:, perm.numpy()]
        else:
            cluster, perm = self.cluster_sort(cluster,data.num_nodes,num_parts)
            csr = sp.csr_matrix((np.arange(data.num_edges), data.edge_index.numpy()))
            a = csr[perm]
            out = a[:, perm]

        partptr = ind2ptr(num_parts,np.array(cluster))

        return out, partptr, perm

    def cluster_sort(self, cluster, num_nodes,num_parts):
        part_row = []
        part_col = []
        for i in range(num_parts):
            for node_idx in range(num_nodes):
                if cluster[node_idx] == i:
                    part_row.append(i)
                    part_col.append(node_idx)
        #print("test_sort: ",part_row[220000])
        return part_row, part_col

    def __permute_data__(self, data, node_idx, adj):
        out = copy.copy(data)

        if tlx.BACKEND == 'torch':
            node_idx = tlx.ops.convert_to_tensor(node_idx)
            for key, value in data.items():
                if data.is_node_attr(key):
                    out[key] = value[node_idx]
        else:
            for key, value in data.items():
                if data.is_node_attr(key):
                    temp = np.array(value)
                    node_idx = np.array(node_idx)
                    out[key] = tlx.ops.convert_to_tensor(temp[node_idx])
        out.edge_index = None
        out.adj = adj

        return out

    def __len__(self):
        return len(self.partptr) - 1

    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        end = int(self.partptr[idx + 1])
        length = end - start

        N, E = self.data.num_nodes, self.data.num_edges
        data = copy.copy(self.data)
        del data.num_nodes
        adj, data.adj = data.adj, None

        #adj = adj.narrow(0, start, length).narrow(1, start, length)
        adj = adj[start:end]
        adj = adj[:,start:end]
        #edge_idx = adj.storage.value()
        edge_idx = adj.tocoo.data

        for key, item in data:
            if tlx.ops.is_tensor(item) and item.size(0) == N:
                data[key] = item.narrow(0, start, length)
            elif tlx.ops.is_tensor(item) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        row, col, _ = adj.coo()
        data.edge_index = tlx.stack([row, col], 0)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'  data={self.data},\n'
                f'  num_parts={self.num_parts}\n'
                f')')


class ClusterLoader(DataLoader):

    def __init__(self, cluster_data, **kwargs):
        self.cluster_data = cluster_data

        super().__init__(range(len(cluster_data)), collate_fn=self.__collate__,
                         **kwargs)

    def __collate__(self, batch):
        if not tlx.ops.is_tensor(batch):
            batch = tlx.ops.convert_to_tensor(batch)

        N = self.cluster_data.data.num_nodes
        E = self.cluster_data.data.num_edges
        start = self.cluster_data.partptr[tlx.convert_to_numpy(batch)].tolist()
        end = self.cluster_data.partptr[tlx.convert_to_numpy(batch) + 1].tolist()
        node_idx = tlx.concat([tlx.arange(s, e) for s, e in zip(start, end)],0)

        data = copy.copy(self.cluster_data.data)
        del data.num_nodes
        adj, data.adj = self.cluster_data.data.adj, None

        adj = sp.vstack([adj[s:e] for s, e in zip(start, end)])
        adj = adj[:,tlx.convert_to_numpy(node_idx)]
        row = adj.tocoo().row
        col = adj.tocoo().col
        edge_idx = adj.tocoo().data
        data.edge_index = tlx.stack([tlx.ops.convert_to_tensor(row), tlx.ops.convert_to_tensor(col)], 0)

        if tlx.BACKEND == 'torch':
            for key in data.keys:
                if key == 'x':
                    data[key] = data.x[node_idx]
                elif key == 'y':
                    data[key] = data.y[node_idx]
                elif key == 'train_mask':
                    data[key] = data.train_mask[node_idx]
                elif key == 'test_mask':
                    data[key] = data.test_mask[node_idx]
                elif key == 'val_mask':
                    data[key] = data.val_mask[node_idx]
                elif key == 'edge_index':
                    if data.edge_index.size(0) == E:
                        data[key] = data.edge_index[edge_idx]
                    data[key] = data.edge_index
        else:
            for key in data.keys:
                if key == 'x':
                    data[key] = tlx.convert_to_tensor(data.x.numpy()[node_idx])
                elif key == 'y':
                    data[key] = tlx.convert_to_tensor(data.y.numpy()[node_idx])
                elif key == 'train_mask':
                    data[key] = tlx.convert_to_tensor(data.train_mask.numpy()[node_idx])
                elif key == 'test_mask':
                    data[key] = tlx.convert_to_tensor(data.test_mask.numpy()[node_idx])
                elif key == 'val_mask':
                    data[key] = tlx.convert_to_tensor(data.val_mask.numpy()[node_idx])
                elif key == 'edge_index':
                    if data.edge_index.shape[0] == E:
                        data[key] = tlx.convert_to_tensor(data.val_mask.numpy()[edge_idx])
                    else:
                        data[key] = data.edge_index


        return data
