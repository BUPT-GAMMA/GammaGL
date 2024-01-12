from gammagl.transforms import BaseTransform
from scipy.stats import bernoulli
from gammagl.data import Graph
import tensorlayerx as tlx
import numpy as np

class DropEdge(BaseTransform):
    r"""Randomly drop edges, as described in
        `DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
        <https://arxiv.org/abs/1907.10903>`__ paper.

        Parameters
        ----------
        p : float, optional
            Probability of an edge to be dropped.

        Example
        -------
        >>> import numpy as np
        >>> from gammagl.data import graph
        >>> from gammagl.transforms import DropEdge
        >>> import tensorlayerx as tlx
        >>> transform = DropEdge()
        >>> g = graph.Graph(x=np.random.randn(5, 16), edge_index=tlx.convert_to_tensor([[0, 0, 0], [1, 2, 3]]), edge_attr=tlx.convert_to_tensor([[0,0,0],[1,1,1],[2,2,2]]),num_nodes=5,)
        >>> new_g = transform(g)
        >>> print(g)
        Graph(edge_index=[2, 3], edge_attr=[3, 3], x=[5, 16], num_nodes=5)
        >>> print(new_g)
        Graph(edge_index=[2, 2], edge_attr=[2, 3], x=[5, 16], num_nodes=5)
        >>> from gammagl.data import HeteroGraph
        >>> data = HeteroGraph()
        >>> num_papers=5
        >>> num_paper_features=6
        >>> num_authors=7
        >>> num_authors_features=8
        >>> data['paper'].x = tlx.convert_to_tensor(np.random.uniform((num_papers, num_paper_features)))
        >>> data['author'].x = tlx.convert_to_tensor(np.random.uniform((num_authors, num_authors_features)))
        >>> data['author', 'writes', 'paper'].edge_index = tlx.convert_to_tensor([[0, 0, 0], [1, 2, 3]])  # [2, num_edges]
        >>> data['author', 'writes', 'paper'].edge_attr = tlx.convert_to_tensor([[0, 0, 0], [1, 1, 1],[2, 2, 2]])
        >>> data['author', 'writes', 'author'].edge_index = tlx.convert_to_tensor([[1, 2, 3], [2, 3, 1]])
        >>> print(data)
        HeteroGraph(
          paper={ x=[5, 6] },
          author={ x=[7, 8] },
          (author, writes, paper)={
            edge_index=[2, 3],
            edge_attr=[3, 3]
          },
          (author, writes, author)={ edge_index=[2, 3] }
        )
        >>> transform = DropEdge()
        >>> new_data = transform(data)
        >>> print(new_data)
        HeteroGraph(
          paper={ x=[5, 6] },
          author={ x=[7, 8] },
          (author, writes, paper)={
            edge_index=[2, 2],
            edge_attr=[2, 3]
          },
          (author, writes, author)={ edge_index=[2, 2] }
        )
        """
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, g):
        if self.p == 0:
            return g
        if isinstance(g, Graph):
            samples = tlx.convert_to_tensor(bernoulli.rvs(1-self.p, size=g.num_edges), dtype=tlx.bool)

            if not tlx.is_tensor(g.edge_index) and not isinstance(g.edge_index, np.ndarray):
                return g
            elif not tlx.is_tensor(g.edge_attr) and not isinstance(g.edge_index, np.ndarray):
                g.edge_index = tlx.gather(g.edge_index, samples, axis=1)
                return g
            else:
                g.edge_index = tlx.gather(g.edge_index, samples, axis=1)
                g.edge_attr = tlx.gather(g.edge_attr, samples)
                return g
        else:
            for e_type in g.metadata()[-1]:
                samples = tlx.convert_to_tensor(bernoulli.rvs(1-self.p, size=tlx.get_tensor_shape(g[e_type].edge_index)[-1]), dtype=tlx.bool)
                if hasattr(g[e_type],'edge_index'):
                    g[e_type].edge_index = tlx.mask_select(g[e_type].edge_index, samples, axis = 1)
                    if hasattr(g[e_type],'edge_attr'):
                        g[e_type].edge_attr = tlx.mask_select(g[e_type].edge_attr, samples)
                    else:
                        pass
                else:
                    pass
            return g
