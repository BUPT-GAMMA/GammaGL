import tensorlayerx as tlx

class Discriminator(tlx.nn.Module):
    r"""The discriminator of GraphGAN Model operator from the `"GraphGAN: Graph Representation Learning with Generative Adversarial Nets"
    <https://arxiv.org/pdf/1711.08267.pdf>`_ paper

    :math:`D` as the sigmoid function of the inner product of two input vertices is defined as

    .. math::
        D(v|v_{c})=\sigma (d_{v}^{\top }d_{v_{c}})=\frac{1}{1+exp(-d_{v}^{\top }d_{v_{c}})}

    Then update only :math:`d_{v}` and :math:`d_{v_{c}}` by ascending the gradient with respect to them:

    .. math::
        \bigtriangledown _{\Theta _{D}}V(G,D)=\left\{\begin{matrix}\bigtriangledown _{\Theta_{D}}logD(v,v_{c}),if\,v\sim p_{true} \\ \bigtriangledown _{\Theta _{D}}(1-logD(v,if\,v\sim p_{true})),if\,v\sim G\end{matrix}\right.

    Parameters
    ----------
        n_node: int
            Number of nodes in the graph
        node_emb_init: ndarray
            Pre trained discriminator node embedding

    """

    def __init__(self, n_node, node_emb_init):
        super(Discriminator, self).__init__()
        self.n_node = n_node
        self.node_emb_init = node_emb_init

        # embedding = tlx.nn.initializers.constant(node_emb_init)
        # init_bias = np.zeros(((self.n_node, 1)))
        # invitor = tlx.nn.initializers.constant(init_bias)
        embedding = tlx.initializers.Constant(value=self.node_emb_init)
        invitor = tlx.initializers.Zeros()
        self.embedding_matrix = self._get_weights("d_embedding_matrix", shape=self.node_emb_init.shape,
                                                  init=embedding)
        self.bias_vector = self._get_weights(
            "d_bias", shape=(self.n_node, 1), init=invitor)

    def forward(self, data):
        node_embedding = tlx.gather(
            self.embedding_matrix, data['center_nodes'])
        node_neighbor_embedding = tlx.gather(
            self.embedding_matrix, data['neighbor_nodes'])
        bias = tlx.gather(self.bias_vector, data['neighbor_nodes'])

        scores = tlx.nn.Reshape(shape=bias.shape)(
            tlx.reduce_sum(tlx.multiply(node_embedding, node_neighbor_embedding), axis=1)) + bias
        scores = tlx.clip_by_value(
            scores, clip_value_min=-10, clip_value_max=10)
        return node_embedding, node_neighbor_embedding, bias, scores

    def get_reward(self, data):
        """
        Compute :math:`D(v,v_{c})`

        Parameters
        ----------
        data : dict
            ID of center_nodes and corresponding neighbor nodes and number of nodes in data

        Returns
        -------
        Tensor
            :math:`D(v,v_{c})`

        """
        node_embedding = tlx.gather(
            self.embedding_matrix, data['center_nodes'])
        node_neighbor_embedding = tlx.gather(
            self.embedding_matrix, data['neighbor_nodes'])
        bias = tlx.gather(self.bias_vector, data['neighbor_nodes'])

        scores = tlx.nn.Reshape(shape=bias.shape)(
            tlx.reduce_sum(tlx.multiply(node_embedding, node_neighbor_embedding), axis=1)) + bias
        scores = tlx.clip_by_value(
            scores, clip_value_min=-10, clip_value_max=10)
        reward = tlx.log(1 + tlx.exp(scores))
        return reward
