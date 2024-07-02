import os.path as osp
import tensorlayerx as tlx
from gammagl.data import download_url, InMemoryDataset, HeteroGraph
from typing import Callable, List, Optional


class ACM4DHN(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/BUPT-GAMMA/HDE/main/ds/imdb'

    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None,
                 force_reload: bool = False, test_ratio: float = 0.3):
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = self.load_data(self.processed_paths[0])
        self.test_ratio = test_ratio

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'MA.txt'
        ]

    @property
    def processed_file_names(self) -> str:
        return tlx.BACKEND + '_data.pt'

    def download(self):
        for i in range(0, len(self.raw_file_names)):
            download_url(f'{self.url}/{self.raw_file_names[i]}', self.raw_dir)

    def process(self):
        G = HeteroGraph()
        edge_index_M = []
        edge_index_A = []

        path = osp.join(self.raw_dir, 'MA.txt')

        with open(path, 'r') as f:
            for line in f.readlines():
                src, dst = line.strip().split()
                src_type, src_id = src[0], src[1:]  # Resolves the source node type and ID
                dst_type, dst_id = dst[0], dst[1:]  # Resolve the target node type and ID

                # Convert the node ID to an integer index and place it in a list
                if src[0] == 'M':
                    edge_index_M.append(int(src_id))
                elif src[0] == 'A':
                    edge_index_A.append(-int(src_id) - 1)

                if dst[0] == 'M':
                    edge_index_M.append(int(dst_id))
                elif dst[0] == 'A':
                    edge_index_A.append(-int(dst_id) - 1)

        edge_index = tlx.convert_to_tensor([edge_index_M, edge_index_A])
        G['M', 'MA', 'A'].edge_index = edge_index

        # Computed split point
        sp = 1 - self.test_ratio * 2
        num_edge = len(edge_index_M)
        sp1 = int(num_edge * sp)
        sp2 = int(num_edge * self.test_ratio)

        G_train = HeteroGraph()
        G_val = HeteroGraph()
        G_test = HeteroGraph()

        # Divide the training set, the verification set, and the test set
        G_train['M', 'MA', 'A'].edge_index = tlx.convert_to_tensor([edge_index_M[:sp1], edge_index_A[:sp1]])
        G_val['M', 'MA', 'A'].edge_index = tlx.convert_to_tensor(
            [edge_index_M[sp1:sp1 + sp2], edge_index_A[sp1:sp1 + sp2]])
        G_test['M', 'MA', 'A'].edge_index = tlx.convert_to_tensor([edge_index_M[sp1 + sp2:], edge_index_A[sp1 + sp2:]])

        print(
            f"all edge: {len(G['M', 'MA', 'A'].edge_index[0])}, train edge: {len(G_train['M', 'MA', 'A'].edge_index[0])}, val edge: {len(G_val['M', 'MA', 'A'].edge_index[0])}, test edge: {len(G_test['M', 'MA', 'A'].edge_index[0])}")

        G['train'] = G_train  # training set
        G['val'] = G_val  # valuation set
        G['test'] = G_test  # testing set

        if self.pre_transform is not None:
            G = self.pre_transform(G)
        self.save_data(self.collate([G]), self.processed_paths[0])
