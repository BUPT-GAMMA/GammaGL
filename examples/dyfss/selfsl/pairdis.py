import tensorlayerx as tlx
from tensorlayerx import nn
import scipy.sparse as sp
import numpy as np
import networkx as nx
import os
from numba import njit
from numba.typed import List
from multiprocessing import Pool
from examples.dyfss.utils.layers import GraphConvolution


class PairwiseDistance(nn.Module):
    """
    Faster sampling
    """
    def __init__(self, data, processed_data, encoder, nhid1, nhid2, dropout, **kwargs):
        super(PairwiseDistance, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        self.nclass = 4
        self.disc1 = nn.Linear(in_features=nhid1, out_features=self.nclass, name="linear_15")
        self.build_distance()
        self.cnt = 0

        self.gcn = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = nn.Linear(in_features=nhid2, out_features=self.nclass, name="linear_16")

    def build_distance(self):
        if self.args.dataset == 'arxiv':
            A = self.data.adj
            self.num_nodes = A.shape[0]
            A_2 = A @ A
            A_2_aug = A + A_2 + sp.eye(self.num_nodes)
            self.A_3_aug = A_2_aug
            self.A_list = [A, A_2]
            self.pos_edge_index = tlx.convert_to_tensor(np.array(A_2_aug.nonzero()), dtype=tlx.int64)
        else:
            A = self.data.adj
            self.num_nodes = A.shape[0]
            A_2 = A @ A
            A_3 = A_2 @ A
            A_3_aug = A + A_2 + A_3 + sp.eye(self.num_nodes)
            self.A_3_aug = A_3_aug
            self.A_list = [A, A_2, A_3]
            self.pos_edge_index = tlx.convert_to_tensor(np.array(A_3_aug.nonzero()), dtype=tlx.int64)

    def multi_sample(self, n, kn, A_list, all):
        params = [(n, kn, A_list, all), (n, kn, A_list, all),
                (n, kn, A_list, all), (n, kn, A_list, all)]

        pool = Pool(processes=4)
        data = pool.map(work_wrapper, params)
        pool.close()
        pool.join()
        return data

    def sample(self, n=256):
        labels = []
        sampled_edges = []

        runs = 4
        kn = 1000
        all_target_nodes = np.random.default_rng(self.args.seed).choice(self.num_nodes,
                        runs*n, replace=False).astype(np.int32)
        multi = False
        if multi:
            data = self.multi_sample(n, kn, self.A_list, all_target_nodes)
            sampled_edges = [tlx.concat([d[1] for d in data], axis=1)]
            labels = [tlx.concat([d[0] for d in data])]
            ii = 2
        else:
            for _ in range(runs):
                for ii, A in enumerate(self.A_list):
                    target_nodes = all_target_nodes[ii*n: ii*n+n]
                    A = A[target_nodes]
                    if A.nnz > 1e5:
                        A = A[:, all_target_nodes]
                    edges = tlx.convert_to_tensor(A.nonzero(), dtype=tlx.int64)
                    num_edges = edges.shape[1]

                    if num_edges <= kn:
                        idx_selected = np.arange(num_edges)
                    else:
                        idx_selected = np.random.default_rng(self.args.seed).choice(num_edges,
                                        kn, replace=False).astype(np.int32)
                    labels.append(tlx.ones((len(idx_selected),), dtype=tlx.int64) * ii)
                    sampled_edges.append(edges[:, idx_selected])
                    kn = len(idx_selected)

        if self.num_nodes > 5000:
            runs = 10
        # 在sample方法中
        if self.cnt % runs == 0:
            self.cnt += 1
            pos_edge_index = tlx.convert_to_tensor(self.A_3_aug[all_target_nodes].nonzero(), dtype=tlx.int64)
            neg_edges = negative_sampling(
                        edge_index=pos_edge_index, num_nodes=self.num_nodes,
                        num_neg_samples=kn*n, seed=self.args.seed)  # 添加种子参数
            self.neg_edges = neg_edges
            neg_edges = self.neg_edges
        neg_edges = self.neg_edges
        sampled_edges.append(neg_edges)
        labels.append(tlx.ones((neg_edges.shape[1],), dtype=tlx.int64) * (ii+1))

        labels = tlx.concat(labels, axis=0)
        sampled_edges = tlx.concat(sampled_edges, axis=1)
        return sampled_edges, labels

    def make_loss_stage_two(self, encoder_features, adj_norm):
        node_pairs, labels = self.sample()
        embeddings = self.gcn2_forward(encoder_features, adj_norm)

        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.disc2(tlx.abs(embeddings0 - embeddings1))
        # 修改：使用 softmax_cross_entropy_with_logits 替代 log_softmax 和 nll_loss
        loss = tlx.losses.softmax_cross_entropy_with_logits(embeddings, labels)
        return loss

    def make_loss_stage_one(self, embeddings):
        node_pairs, labels = self.sample()
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.disc1(tlx.abs(embeddings0 - embeddings1))
        # 修改：使用 softmax_cross_entropy_with_logits 替代 log_softmax 和 nll_loss
        loss = tlx.losses.softmax_cross_entropy_with_logits(embeddings, labels)
        return loss

    def gcn2_forward(self, input, adj):
        embeddings = self.gcn2(input, adj)
        return embeddings

    def get_label(self):
        graph = nx.from_scipy_sparse_matrix(self.data.adj)

        if not os.path.exists(f'saved/node_distance_{self.args.dataset}.npy'):
            path_length = dict(nx.all_pairs_shortest_path_length(graph, cutoff=self.nclass-1))
            distance = - np.ones((len(graph), len(graph))).astype(int)

            for u, p in path_length.items():
                for v, d in p.items():
                    distance[u][v] = d

            distance[distance==-1] = distance.max() + 1
            distance = np.triu(distance)
            np.save(f'saved/node_distance_{self.args.dataset}.npy', distance)
        else:
            print('loading distance matrix...')
            distance = np.load(f'saved/node_distance_{self.args.dataset}.npy')
        self.distance = distance
        return tlx.convert_to_tensor(distance - 1, dtype=tlx.int64)


def work_wrapper(args):
    return worker(*args)


# 修改worker函数，添加种子参数
def worker(n, kn, A_list, all_target_nodes, seed=42):
    labels = []
    sampled_edges = []
    for ii, A in enumerate(A_list):
        target_nodes = all_target_nodes[ii*n: ii*n+n]
        A = A[target_nodes]
        edges = tlx.convert_to_tensor(A.nonzero(), dtype=tlx.int64)
        num_edges = edges.shape[1]

        if num_edges <= kn:
            idx_selected = np.arange(num_edges)
        else:
            # 使用固定种子确保结果可复现
            idx_selected = np.random.default_rng(seed).choice(num_edges,
                            kn, replace=False).astype(np.int32)
        labels.append(tlx.ones((len(idx_selected),), dtype=tlx.int64) * ii)
        sampled_edges.append(edges[:, idx_selected])
        kn = len(idx_selected)
    labels = tlx.concat(labels, axis=0)
    sampled_edges = tlx.concat(sampled_edges, axis=1)
    return labels, sampled_edges

# 修改work_wrapper函数，传递种子参数
def work_wrapper(args):
    if len(args) == 4:
        n, kn, A_list, all_target_nodes = args
        return worker(n, kn, A_list, all_target_nodes)
    else:
        n, kn, A_list, all_target_nodes, seed = args
        return worker(n, kn, A_list, all_target_nodes, seed)

# 修改multi_sample方法，传递种子参数
def multi_sample(self, n, kn, A_list, all):
    seed = getattr(self.args, 'seed', 42)
    params = [(n, kn, A_list, all, seed), (n, kn, A_list, all, seed),
            (n, kn, A_list, all, seed), (n, kn, A_list, all, seed)]

    pool = Pool(processes=4)
    data = pool.map(work_wrapper, params)
    pool.close()
    pool.join()
    return data

# 修改negative_sampling函数，添加种子参数
def negative_sampling(edge_index, num_nodes, num_neg_samples, seed=42):
    """生成负采样边"""
    # 转换为numpy数组处理
    edge_index_np = tlx.convert_to_numpy(edge_index)
    
    # 创建所有可能的边的集合
    existing_edges = set()
    for i in range(edge_index_np.shape[1]):
        existing_edges.add((edge_index_np[0, i], edge_index_np[1, i]))
    
    # 使用固定种子的随机数生成器
    rng = np.random.default_rng(seed)
    
    # 生成负采样边
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        i = rng.integers(0, num_nodes)
        j = rng.integers(0, num_nodes)
        if i != j and (i, j) not in existing_edges:
            neg_edges.append([i, j])
            existing_edges.add((i, j))
    
    # 转换为张量
    neg_edges = tlx.convert_to_tensor(np.array(neg_edges).T, dtype=tlx.int64)
    return neg_edges
