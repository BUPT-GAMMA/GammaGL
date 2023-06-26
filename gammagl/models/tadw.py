import tensorlayerx as tlx
import numpy as np
from ..utils.num_nodes import maybe_num_nodes
from gammagl.utils import add_self_loops
from gammagl.utils import degree
from scipy.sparse.linalg import svds

lower_control = 10 ** (-15)


class TADWModel(tlx.nn.Module):
    r"""The TADW model from the
    `"Network Representation Learning with Rich Text Information"
    <https://www.ijcai.org/Proceedings/15/Papers/299.pdf>`_ paper
    where random walks of length :obj:`walk_length` are sampled in
    a given graph, and node embeddings are learned via negative
    sampling optimization.
    Parameters
    ----------
        edge_index: Iterable
            The edge indices.
        embedding_dim: int
            The size of each embedding vector.
        lr: float
            The learning rate.
        lamda: float
            A harmonic factor to balance two components.
        svdft: int
            The size of each text feature vector.
        node_feature: Iterable
            The matrix of Node feature.
        name: str
            model name
    """

    def __init__(
            self,
            edge_index,
            embedding_dim,
            lr,
            lamda,
            svdft,
            node_feature,
            num_nodes=None,
            name=None
    ):
        super(TADWModel, self).__init__(name=name)
        self.edge_index = edge_index
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.lamda = lamda
        self.svdft = svdft
        self.node_feature = node_feature
        self.N = maybe_num_nodes(edge_index, num_nodes)
        self.M = self._create_target_matrix()
        self.T = self._create_tfifd_matrix()
        self.T = np.transpose(self.T)

        self.W = np.random.uniform(0, 1, (self.embedding_dim, self.M.shape[0]))
        self.H = np.random.uniform(0, 1, (self.embedding_dim, self.T.shape[0]))

    def _create_target_matrix(self):
        edge_index, _ = add_self_loops(self.edge_index, num_nodes=self.N, n_loops=1)
        num_nodes = self.N
        src = edge_index[0]
        degs = degree(src, num_nodes=num_nodes, dtype=tlx.float32)
        norm_degs = tlx.convert_to_numpy(1.0 / degs)
        A = [[0] * num_nodes for _ in range(num_nodes)]
        length = edge_index.shape[1]
        for i in range(length):
            src = edge_index[0][i]
            dst = edge_index[1][i]
            A[src][dst] = norm_degs[src]
        M = (A + np.dot(A, A)) / 2.
        return M

    def _create_tfifd_matrix(self):
        num_nodes = self.N
        num_feats = self.node_feature.shape[1]
        feature = tlx.convert_to_numpy(self.node_feature)
        for i in range(num_feats):
            temp = tlx.convert_to_numpy(tlx.reduce_sum(tlx.convert_to_tensor(feature[:, i])))
            if temp > 0:
                IDF = tlx.convert_to_numpy(tlx.log(tlx.convert_to_tensor(num_nodes / temp)))
                feature[:, i] = feature[:, i] * IDF
        U, S, V = svds(feature, k=self.svdft)
        text_feature = U.dot(np.diag(S))
        length = len(text_feature)
        for i in range(length):
            text_feature[i] = text_feature[i] / np.linalg.norm(text_feature[i], ord=2)
        # text_feature = text_feature * 0.1
        return text_feature

    def fit(self):
        """
        Gradient descent updates.
        """
        loss = self.loss()
        self.update_W()
        self.update_H()
        return loss

    def update_W(self):
        """
        A single update of the node embedding matrix.
        """
        H_T = np.dot(self.H, self.T)
        grad = self.lamda * self.W - np.dot(H_T, self.M - np.dot(np.transpose(H_T), self.W))
        self.W = self.W - self.lr * grad
        # Overflow control
        self.W[self.W < lower_control] = lower_control

    def update_H(self):
        """
        A single update of the feature basis matrix.
        """
        inside = self.M - np.dot(np.dot(np.transpose(self.W), self.H), self.T)
        grad = self.lamda * self.H - np.dot(np.dot(self.W, inside), np.transpose(self.T))
        self.H = self.H - self.lr * grad
        # Overflow control
        self.H[self.H < lower_control] = lower_control

    def loss(self):
        # self.score_matrix = self.M - np.dot(np.dot(np.transpose(self.W), self.H), self.T)
        # main_loss = np.sum(np.square(self.score_matrix))

        self.score_matrix = self.M - np.dot(np.dot(np.transpose(self.W), self.H), self.T)
        # main_loss = np.mean(np.square(self.score_matrix))
        # regul_1 = self.lamda * np.mean(np.square(self.W)) / 2
        # regul_2 = self.lamda * np.mean(np.square(self.H)) / 2
        main_loss = np.sum(np.square(self.score_matrix))
        regul_1 = self.lamda * np.sum(np.square(self.W)) / 2
        regul_2 = self.lamda * np.sum(np.square(self.H)) / 2
        main_loss = main_loss + regul_1 + regul_2

        return main_loss

    def campute(self):
        to_concat = [np.transpose(self.W), np.transpose(np.dot(self.H, self.T))]
        return np.concatenate(to_concat, axis=1)
