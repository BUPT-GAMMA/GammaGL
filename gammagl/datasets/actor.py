from typing import Callable, List, Optional
import numpy as np
import tensorlayerx as tlx
from gammagl.data import InMemoryDataset, download_url, Graph
from gammagl.utils import coalesce

class Actor(InMemoryDataset):
    r"""The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of words of
    actor's Wikipedia.

    Parameters
    ----------
    root: str
        Root directory where the dataset should be saved.
    transform: callable, optional
        A function/transform that takes in a
        :obj:`gammagl.data.Graph` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform: callable, optional
        A function/transform that takes in
        an :obj:`gammagl.data.Graph` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 7,600
          - 30,019
          - 932
          - 5
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None, force_reload: bool = False) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self) -> None:
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0], 'r') as f:
            node_data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, line, _ in node_data:
                indices = [int(x) for x in line.split(',')]
                rows += [int(n_id)] * len(indices)
                cols += indices
            row = np.array(rows, dtype=np.int64)
            col = np.array(cols, dtype=np.int64)

            num_nodes = int(row.max()) + 1
            num_features = int(col.max()) + 1
            x = np.zeros((num_nodes, num_features), dtype=np.float32)
            x[row, col] = 1.0

            y = np.zeros(len(node_data), dtype=np.int64)
            for n_id, _, label in node_data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            edge_data = f.read().split('\n')[1:-1]
            edge_indices = [[int(v) for v in r.split('\t')] for r in edge_data]
            edge_index = np.array(edge_indices, dtype=np.int64).T
            edge_index = coalesce(edge_index)  # 保留self loop

        train_masks, val_masks, test_masks = [], [], []
        for path in self.raw_paths[2:]:
            tmp = np.load(path)
            train_masks.append(tmp['train_mask'].astype(np.bool_))
            val_masks.append(tmp['val_mask'].astype(np.bool_))
            test_masks.append(tmp['test_mask'].astype(np.bool_))
        train_mask = np.stack(train_masks, axis=1)
        val_mask = np.stack(val_masks, axis=1)
        test_mask = np.stack(test_masks, axis=1)

        data = Graph(x=tlx.convert_to_tensor(x), edge_index=tlx.convert_to_tensor(edge_index),
                     y=tlx.convert_to_tensor(y), train_mask=tlx.convert_to_tensor(train_mask),
                     val_mask=tlx.convert_to_tensor(val_mask), test_mask=tlx.convert_to_tensor(test_mask))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save_data(self.collate([data]), self.processed_paths[0])
