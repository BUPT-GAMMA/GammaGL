import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import segment_softmax
import math
class MultiHead(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
      Unified Message Passing Model for Semi-Supervised Classification"
      <https://arxiv.org/abs/2009.03509>`_ paper

      .. math::
          \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
          \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

      where the attention coefficients :math:`\alpha_{i,j}` are computed via
      multi-head dot product attention:

      .. math::
          \alpha_{i,j} = \textrm{softmax} \left(
          \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
          {\sqrt{d}} \right)

      Args:
          in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
              derive the size from the first input(s) to the forward method.
              A tuple corresponds to the sizes of source and target
              dimensionalities.
          out_channels (int): Size of each output sample.
          heads (int, optional): Number of multi-head-attentions.
              (default: :obj:`1`)
              .. math::
                  \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
                  (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
                  \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}
          beta
              with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
              [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
              \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)

              .. math::
                  \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
                  \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
                  \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
                  \right),

              where the attention coefficients :math:`\alpha_{i,j}` are now
              computed via:

              .. math::
                  \alpha_{i,j} = \textrm{softmax} \left(
                  \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
                  (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
                  {\sqrt{d}} \right)
    """

    def __init__(self, in_features, out_features, n_heads,num_nodes,beta=True):
        super().__init__()
        self.beta=beta
        self.heads=n_heads
        self.num_nodes=num_nodes
        self.out_channels=out_features
        self.linear = tlx.layers.Linear(out_features=out_features* n_heads,
                                        in_features=in_features)

        self.lin_key = tlx.layers.Linear(in_features=in_features, out_features=n_heads * out_features, bias=True)
        self.lin_query = tlx.layers.Linear(in_features=in_features, out_features=n_heads * out_features, bias=True)
        self.lin_value = tlx.layers.Linear(in_features=in_features, out_features=n_heads * out_features, bias=True)
        self.lin_skip = tlx.layers.Linear(in_features=in_features, out_features=n_heads * out_features, bias=True)
        if beta:
            self.lin_beta = tlx.layers.Linear(3 * n_heads * out_features, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def message(self,  query, key, value):
        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = segment_softmax(alpha)
        alpha = tlx.layers.Dropout(alpha)
        out = value
        out = out * alpha.view(-1, self.heads, 1)
        return out
    
    def forward(self, x, edge_index):
        H, C = self.heads, self.out_channels
        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        out = self.propagate(edge_index, query=query, key=key, value=value)
        out = out.view(-1, self.heads * self.out_channels)
        if self.beta:
            x_r = self.lin_skip(x[1])
            beta = self.lin_beta(tlx.ops.concat([out, x_r, out - x_r], aixs=-1))
            beta = beta.sigmoid()
            out = beta * x_r + (1 - beta) * out
        return out