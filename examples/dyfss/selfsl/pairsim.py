import tensorlayerx as tlx
from tensorlayerx import nn
import scipy.sparse as sp
import numpy as np
import os
from examples.dyfss.utils.layers import GraphConvolution


class PairwiseAttrSim(nn.Module):

    def __init__(self, data, processed_data, encoder, nhid1, nhid2, dropout, **kwargs):
        super(PairwiseAttrSim, self).__init__()
        self.args = kwargs['args']
        self.data = data
        self.processed_data = processed_data
        # self.device = device
        self.num_nodes = data.adj.shape[0]
        self.nclass = 2
        self.disc1 = nn.Linear(in_features=nhid1, out_features=self.nclass, name="linear_17")
        self.build_knn(self.processed_data.features, k=10)

        self.gcn = encoder
        self.gcn2 = GraphConvolution(nhid2, nhid2, dropout, act=lambda x: x)
        self.disc2 = nn.Linear(in_features=nhid2, out_features=self.nclass, name="linear_18")
        

    def build_knn(self, X, k=10):
        args = self.args
        if not os.path.exists(f'saved/{args.dataset}_knn_{k}.npz'):
            print("performance buliding knn...")
            from sklearn.neighbors import kneighbors_graph
            if not isinstance(X, np.ndarray):
                X = tlx.convert_to_numpy(X)
            A_knn = kneighbors_graph(X, k, mode='connectivity',
                            metric='cosine', include_self=True, n_jobs=4)
            if not os.path.exists('saved'):
                os.makedirs('saved')
            sp.save_npz(f'saved/{args.dataset}_knn_{k}.npz', A_knn)
        else:
            A_knn = sp.load_npz(f'saved/{args.dataset}_knn_{k}.npz')
        self.edge_index_knn = tlx.convert_to_tensor(np.array(A_knn.nonzero()), dtype=tlx.int64)

    def sample(self, n_samples=4000):
        labels = []
        sampled_edges = []
        num_edges = self.edge_index_knn.shape[1]
        idx_selected = np.random.default_rng(self.args.seed).choice(num_edges,
                        n_samples, replace=False).astype(np.int32)

        # labels.append(torch.ones(len(idx_selected), dtype=torch.long))
        labels.append(tlx.ones((len(idx_selected),), dtype=tlx.int64))

        sampled_edges.append(self.edge_index_knn[:, idx_selected])
        neg_edges = self.negative_sampling(
                    edge_index=self.edge_index_knn, num_nodes=self.num_nodes,
                    num_neg_samples=n_samples)
        sampled_edges.append(neg_edges)

        # labels.append(torch.zeros(neg_edges.shape[1], dtype=torch.long))
        labels.append(tlx.zeros((neg_edges.shape[1],), dtype=tlx.int64))

        # labels = torch.cat(labels).to(self.device)
        # sampled_edges = torch.cat(sampled_edges, axis=1)
        labels = tlx.concat(labels, axis=0)
        sampled_edges = tlx.concat(sampled_edges, axis=1)

        return sampled_edges, labels

    def negative_sampling(self, edge_index, num_nodes, num_neg_samples):
        edge_index_np = tlx.convert_to_numpy(edge_index)
        
        edge_set = set([(i, j) for i, j in zip(edge_index_np[0], edge_index_np[1])])
        
        neg_edges = []
        while len(neg_edges) < num_neg_samples:
            i = np.random.randint(0, num_nodes)
            j = np.random.randint(0, num_nodes)
            if (i, j) not in edge_set and i != j:
                neg_edges.append([i, j])
        
        return tlx.convert_to_tensor(np.array(neg_edges).T, dtype=tlx.int64)

    def gcn2_forward(self, input, adj):
        embeddings = self.gcn2(input, adj)
        return embeddings

    def make_loss_stage_two(self, encoder_features, adj_norm):
        node_pairs, labels = self.sample()
        embeddings = self.gcn2_forward(encoder_features, adj_norm)
        embeddings0 = tlx.gather(embeddings, node_pairs[0])
        embeddings1 = tlx.gather(embeddings, node_pairs[1])
        embeddings = self.disc2(tlx.abs(embeddings0 - embeddings1))
        loss = tlx.losses.softmax_cross_entropy_with_logits(embeddings, labels)
        return loss

    def make_loss_stage_one(self, embeddings):
        node_pairs, labels = self.sample()
        node_pairs = [tlx.convert_to_tensor(pair) for pair in node_pairs]
        labels = tlx.convert_to_tensor(labels)
        embeddings0 = tlx.gather(embeddings, node_pairs[0])
        embeddings1 = tlx.gather(embeddings, node_pairs[1])
        embeddings = self.disc1(tlx.ops.abs(embeddings0 - embeddings1))  
        loss = tlx.losses.softmax_cross_entropy_with_logits(embeddings, labels)  

        return loss


