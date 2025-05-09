import numpy as np
import tensorlayerx as tlx
from gammagl.data.dataset_base import DatasetBase
from gammagl.data.graph_dataset import GraphDataset
from gammagl.data.svd_encodings_dataset import SVDEncodingsGraphDataset
from gammagl.data.structural_dataset import StructuralDataset
from gammagl.data import InMemoryDataset
from typing import Optional, Callable, List
import os.path as osp
from gammagl.data import Graph


class PCQM4Mv2(InMemoryDataset):
    def __init__(self, root: str = None,
                 dataset_name: str = 'pcqm4m-v2',
                 split: str = "training",
                 transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False,
                 dataset_path: str = './cache_data', 
                 ):

        self.dataset_name = dataset_name
        self.split = split
        self.dataset_path = dataset_path

        
        self.dataset_processor = PCQM4Mv2Dataset(dataset_path=self.dataset_path, split=split)
        
        super().__init__(root, transform, pre_transform, pre_filter, force_reload) 

    @property
    def raw_dir(self) -> str:
        return osp.join(self.dataset_path,'PCQM4MV2', 'pcqm4m-v2', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.dataset_path,'PCQM4MV2','training')

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.csv.gz']

    @property
    def processed_file_names(self) -> str:
        return ['max_nodes_data.npy','records.npy','svd_encoding.npy','tokens.npy']

    def download(self):
        from ogb.lsc import PCQM4Mv2Dataset
        PCQM4Mv2Dataset(root=self.dataset_path, only_smiles=True)

    @property
    def record_tokens(self):
        if not hasattr(self, '_dataset'):
            self._dataset = PCQM4Mv2Dataset(dataset_path=self.dataset_path, split=self.split)
        
        
        tokens = self._dataset.record_tokens
        if not isinstance(tokens, (list, np.ndarray)):
            raise TypeError(f"Expected record_tokens to be list/array, got {type(tokens)}")
        return tokens


    def read_record(self, token):
        return self._dataset.read_record(token)

    def process(self):
        data_list = []
        for token in self.record_tokens:
            graph = self.read_record(token)
            if graph is not None:
                if not isinstance(graph, dict):
                    print(f"Invalid graph format for token {token}")
                    continue
                try:
                    edge_index = tlx.convert_to_tensor(graph['edges'].T, dtype=tlx.int64)
                    x = tlx.convert_to_tensor(graph['node_features'], dtype=tlx.float32)
                    y = tlx.convert_to_tensor(graph['target'], dtype=tlx.float32)
                    edge_attr = tlx.convert_to_tensor(graph['edge_features'], dtype=tlx.float32)

                    data = Graph(edge_index=edge_index, x=x, y=y, edge_attr=edge_attr)
                except KeyError as e:
                    print(f"Missing required key in graph data: {e}")
                    continue

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        data, slices = self.collate(data_list)

        self.save_data((data, slices), self.processed_paths[0])

class PCQM4Mv2Dataset(DatasetBase):
    def __init__(self, 
                 dataset_path             ,
                 dataset_name = 'PCQM4MV2',
                 **kwargs
                 ):
        super().__init__(dataset_name = dataset_name,
                         **kwargs)
        self.dataset_path    = dataset_path
    
    @property
    def dataset(self):
        try:
            return self._dataset
        except AttributeError:
            import os
            from filelock import FileLock
            from ogb.lsc import PCQM4Mv2Dataset
            from ogb.utils import smiles2graph
            os.makedirs(self.dataset_path, exist_ok=True)
            lock_path = os.path.join(self.dataset_path, "dataset.lock")
            with FileLock(lock_path):
                if not hasattr(self, '_dataset'):
                    self._smiles2graph = smiles2graph
                    self._dataset = PCQM4Mv2Dataset(root=self.dataset_path, only_smiles=True)
            
            return self._dataset

    @property
    def record_tokens(self):
        try:
            return self._record_tokens
        except AttributeError:
            split = {'training':'train', 
                     'validation':'valid', 
                     'test':'test-dev', 
                     'challenge': 'test-challenge'}[self.split]
            self._record_tokens = self.dataset.get_idx_split()[split]
            return self._record_tokens
    
    def read_record(self, token):
        smiles, target = self.dataset[token]
        graph = self._smiles2graph(smiles)
        graph['num_nodes'] = np.array(graph['num_nodes'], dtype=np.int16)
        graph['edges'] = graph.pop('edge_index').T.astype(np.int16)
        graph['edge_features'] = graph.pop('edge_feat').astype(np.int16)
        graph['node_features'] = graph.pop('node_feat').astype(np.int16)
        graph['target'] = np.array(target, np.float32)
        return graph



class PCQM4Mv2GraphDataset(GraphDataset,PCQM4Mv2Dataset):
    pass

class PCQM4Mv2SVDGraphDataset(SVDEncodingsGraphDataset,PCQM4Mv2Dataset):
    pass

class PCQM4Mv2StructuralGraphDataset(StructuralDataset,PCQM4Mv2GraphDataset):
    pass

class PCQM4Mv2StructuralSVDGraphDataset(StructuralDataset,PCQM4Mv2SVDGraphDataset):
    pass



