import numpy as np
import tensorlayerx as tlx
from gammagl.data import Graph


class AirportsDataset:
    def __init__(self, root='data/airports', name='brazil'):
        with open(f'{root}/{name}-airports.edgelist', 'r') as f:
            edges = f.readlines()
        edges = [[int(x) for x in e.strip().split()] for e in edges]
        edge_index = np.array(edges, dtype=np.int64).T
        num_nodes = int(edge_index.max()) + 1
        x = np.eye(num_nodes, dtype=np.float32)

        with open(f'{root}/{name}-airports.labels', 'r') as f:
            labels = f.readlines()
        labels = [int(x.strip()) for x in labels]
        y = np.array(labels[:num_nodes], dtype=np.int64)
        num_classes = int(max(labels)) + 1

        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)

        indices = np.random.permutation(num_nodes)
        train_size = int(0.5 * num_nodes)
        val_size = int(0.25 * num_nodes)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        data = Graph(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        data.train_mask = tlx.convert_to_tensor(train_mask)
        data.val_mask = tlx.convert_to_tensor(val_mask)
        data.test_mask = tlx.convert_to_tensor(test_mask)
        data.num_features = num_nodes
        data.num_classes = num_classes
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data
