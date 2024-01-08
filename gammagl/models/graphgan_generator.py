import tensorlayerx as tlx

class Generator(tlx.nn.Module):
    r"""The generator of GraphGAN Model operator from the `"GraphGAN: Graph Representation Learning with Generative Adversarial Nets"
        <https://arxiv.org/pdf/1711.08267.pdf>`_ paper.

        Compute the gradient of :math:`V(G,D)` with respect to :math:`\Theta _{G}` by

        .. math::
            \bigtriangledown _{\Theta _{G}}V(G,D)= \sum_{c=1}^{V}{E_{v\sim G(\cdot |v_{c}))}}[\bigtriangledown _{\Theta _{G}}logG(v|v_{c})log(1-G(v,v_{c}))]

        The relevance probability of :math:`v_{i}` given :math:`v` as

        .. math::
            p_{c}(v_{i}|v)= \frac{exp(g_{v_{i}}^{\top }g_{v})}{{\sum _{v_{j}\epsilon {N}_{c}(v)}exp(g_{v_{i}}^{\top }g_{v})}}

        Then the graph softmax defines :math:`G(v|v_{c};\Theta _{G})` as follows:

        .. math::
            G(v|v_{c})=\left (\prod _{j=1}^{m}p_{c}(v_{r_{j}}|v_{r_{j-1}}) \right )\cdot p_{c}(v_{r_{m-1}}|v_{r_{m}})

        Parameters
        ----------
        n_node: int
            Number of nodes in the graph.
        node_emb_init: numpy.ndarray
            Pre trained generator node embedding.

    """

    def __init__(self, n_node, node_emb_init):
        super(Generator, self).__init__()
        self.n_node = n_node
        self.node_emb_init = node_emb_init

        # embedding = tlx.initializers.Constant(value=self.node_emb_init)
        # init_bias = np.zeros(((self.n_node, 1)))
        # invitor = tlx.nn.initializers.constant(init_bias)
        embedding = tlx.initializers.Constant(value=self.node_emb_init)
        invitor = tlx.initializers.Zeros()
        self.embedding_matrix = self._get_weights("g_embedding_matrix", shape=self.node_emb_init.shape,
                                                  init=embedding)
        self.bias_vector = self._get_weights(
            "g_bias", shape=(self.n_node, 1), init=invitor)
            
        if tlx.BACKEND == 'torch':
            embedding_matrix = self.embedding_matrix.clone().detach()
            embedding_matrix_transpose = tlx.transpose(embedding_matrix)
            self.all_scores = tlx.matmul(
                embedding_matrix, embedding_matrix_transpose) + self.bias_vector.detach()
        else:
            self.all_scores = tlx.matmul(self.embedding_matrix, self.embedding_matrix,
                                         transpose_b=True) + self.bias_vector

    def forward(self, data):
        node_embedding = tlx.gather(self.embedding_matrix, data['node_1'])
        node_neighbor_embedding = tlx.gather(
            self.embedding_matrix, data['node_2'])
        bias = tlx.gather(self.bias_vector, data['node_2'])
        score = tlx.nn.Reshape(shape=bias.shape)(
            tlx.reduce_sum(node_embedding * node_neighbor_embedding, axis=1)) + bias
        prob = tlx.clip_by_value(tlx.sigmoid(score), 1e-5, 1)
        return node_embedding, node_neighbor_embedding, prob

    def get_all_scores(self):
        """
        Compute the relevance probability of :math:`v_{i}` given :math:`v`

        Returns
        -------
        Tensor
            the relevance probability of :math:`v_{i}` given :math:`v`

        """
        if tlx.BACKEND == 'torch':
            embedding_matrix = self.embedding_matrix.clone().detach()
            embedding_matrix_transpose = tlx.transpose(embedding_matrix)
            self.all_scores = tlx.matmul(
                embedding_matrix, embedding_matrix_transpose) + self.bias_vector.detach()
        else:
            self.all_scores = tlx.matmul(self.embedding_matrix, self.embedding_matrix,
                                         transpose_b=True) + self.bias_vector
        return self.all_scores
