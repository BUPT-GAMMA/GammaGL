import numpy as np
import tensorlayerx as tlx
import networkx as nx
from gammagl.data import Graph


class KarateClubDataset:
    def __init__(self):
        G = nx.karate_club_graph()
        edges = list(G.edges())
        edge_index = np.array(edges, dtype=np.int64).T
        edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
        num_nodes = G.number_of_nodes()
        x = np.eye(num_nodes, dtype=np.float32)
        y = np.array([G.nodes[i].get('club', '') for i in range(num_nodes)])
        unique_clubs = sorted(set(y))
        y = np.array([unique_clubs.index(c) for c in y], dtype=np.int64)
        num_classes = len(unique_clubs)

        train_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[:20] = True
        val_mask = np.zeros(num_nodes, dtype=bool)
        val_mask[20:30] = True
        test_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[30:] = True

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


class GitHubDataset:
    url = 'https://raw.githubusercontent.com/EdisonLeeeee/GraphData/master/datasets/git_web_sp.npz'

    def __init__(self, root='data/github'):
        import os
        from gammagl.data import download_url

        raw_dir = os.path.join(root, 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        raw_path = os.path.join(raw_dir, 'git_web_sp.npz')
        if not os.path.exists(raw_path):
            download_url(self.url, raw_dir)

        data_np = np.load(raw_path, allow_pickle=True)
        x = data_np['features'].astype(np.float32)
        y = data_np['target'].astype(np.int64)
        edge_index = data_np['edges'].astype(np.int64).T

        data = Graph(x=x, y=y, edge_index=edge_index)
        data.num_features = x.shape[1]
        data.num_classes = int(max(y)) + 1
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data


class AirportsDataset:
    edge_url = 'https://raw.githubusercontent.com/leoribeiro/struc2vec/master/graph/{}-airports.edgelist'
    label_url = 'https://raw.githubusercontent.com/leoribeiro/struc2vec/master/graph/labels-{}-airports.txt'

    def __init__(self, root='data/airports', name='usa'):
        import os
        from gammagl.data import download_url

        self.name = name.lower()
        assert self.name in ['usa', 'brazil', 'europe']

        raw_dir = os.path.join(root, self.name.capitalize(), 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        edge_path = os.path.join(raw_dir, f'{self.name}-airports.edgelist')
        label_path = os.path.join(raw_dir, f'labels-{self.name}-airports.txt')

        if not os.path.exists(edge_path):
            download_url(self.edge_url.format(self.name), raw_dir)
        if not os.path.exists(label_path):
            download_url(self.label_url.format(self.name), raw_dir)

        index_map, ys = {}, []
        with open(label_path, 'r') as f:
            rows = f.read().split('\n')[1:-1]
            for i, row in enumerate(rows):
                idx, label = row.split()
                index_map[int(idx)] = i
                ys.append(int(label))

        num_nodes = len(ys)
        y = np.array(ys, dtype=np.int64)
        x = np.eye(num_nodes, dtype=np.float32)

        edge_indices = []
        with open(edge_path, 'r') as f:
            rows = f.read().split('\n')[:-1]
            for row in rows:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])

        edge_index = np.array(edge_indices, dtype=np.int64).T

        data = Graph(x=x, y=y, edge_index=edge_index, num_nodes=num_nodes)
        data.num_features = num_nodes
        data.num_classes = int(max(y)) + 1
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data
