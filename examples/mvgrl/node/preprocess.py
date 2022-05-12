import networkx as nx
import numpy as np
from scipy.linalg import fractional_matrix_power, inv

from gammagl.data import Graph
import scipy.sparse as sp

def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph, dtype=np.float32)
    if self_loop:
        a = a + np.eye(a.shape[0], dtype=np.float32)  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0], dtype=np.float32) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1

def process_dataset(name, graph):
    # if name == 'cora':
    #     dataset = CoraGraphDataset()
    # elif name == 'citeseer':
    #     dataset = CiteseerGraphDataset()

    feat = graph.x
    label = graph.y

    train_mask = graph.train_mask
    val_mask = graph.val_mask
    test_mask = graph.test_mask
    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    test_idx = np.nonzero(test_mask)[0]

    src, dst = graph.edge_index.numpy()
    nx_g = nx.DiGraph()
    nx_g.add_nodes_from(range(graph.num_nodes))
    for e, (u, v) in enumerate(zip(src, dst)):
        nx_g.add_edge(u, v, id=e)
    print('computing ppr')
    diff_adj = compute_ppr(nx_g, 0.2)
    print('computing end')

    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    coo = sp.coo_matrix((diff_weight, np.array([diff_edges[0], diff_edges[1]])), shape=(feat.shape[0], feat.shape[0]))
    # graph = graph.add_self_loop()

    return np.array(coo.todense()), feat, label, train_idx, val_idx, test_idx
