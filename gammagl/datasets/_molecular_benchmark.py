import gzip
import json
import math
import os
import os.path as osp
import pickle
from collections.abc import Iterable
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import tensorlayerx as tlx

from gammagl.data import Graph, InMemoryDataset, download_url
from gammagl.data.extract import extract_tar, extract_zip
from gammagl.data.makedirs import makedirs

from ._graph_tokenizer_download import materialize_paper_dataset


def validate_molecular_splits(num_samples: int, split_indices: Dict[str, List[int]]) -> None:
    required = ('train', 'val', 'test')
    if set(split_indices) != set(required):
        raise ValueError(f"Split files must define exactly {required}.")

    seen = {}
    for split in required:
        for index in split_indices[split]:
            if index < 0 or index >= num_samples:
                raise ValueError(
                    f"{split} split index {index} is outside [0, {num_samples}).")
            if index in seen:
                raise ValueError(
                    f"{split} split index {index} overlaps {seen[index]} split.")
            seen[index] = split


class PreprocessedMolecularBenchmark(InMemoryDataset):
    r"""Base class for molecular benchmark datasets stored as preprocessed
    graph pickle files plus train/validation/test split indices.

    The expected raw layout is:

    .. code-block:: text

        root/name/raw/data.pkl or data.pkl.gz
        root/name/raw/train_index.json
        root/name/raw/val_index.json
        root/name/raw/test_index.json
    """

    name: str = ""
    display_name: str = ""
    url: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    metric: str = ""
    num_tasks: int = 1
    task_type: str = ""
    label_keys: Tuple[str, ...] = ()
    allow_nan_labels: bool = False
    node_feature_columns: Dict[str, int] = {}
    edge_feature_columns: Dict[str, int] = {}

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self._download()
        self._process()
        self.data, self.slices = self.load_data(self.processed_paths[0])
        self.split_indices = self._load_processed_splits()

    @property
    def raw_file_names(self) -> List[str]:
        return ['train_index.json', 'val_index.json', 'test_index.json']

    @property
    def processed_file_names(self) -> List[str]:
        return [tlx.BACKEND + '_data.pt', 'split_indices.json']

    def _download(self):
        if self._has_raw_files() or self._has_processed_files():
            return
        makedirs(self.raw_dir)
        self.download()

    def _has_raw_files(self) -> bool:
        return self._raw_data_file() is not None and all(
            osp.exists(osp.join(self.raw_dir, name)) for name in self.raw_file_names)

    def _has_processed_files(self) -> bool:
        return all(osp.exists(path) for path in self.processed_paths)

    def _raw_data_file(self) -> Optional[str]:
        for filename in ('data.pkl', 'data.pkl.gz'):
            path = osp.join(self.raw_dir, filename)
            if osp.exists(path):
                return path
        return None

    def download(self) -> None:
        if self.url is None:
            materialize_paper_dataset(
                dataset_name=self.name,
                aliases=self.aliases,
                raw_dir=self.raw_dir,
                cache_root=self.root,
            )
            return

        path = download_url(self.url, self.raw_dir)
        if path.endswith('.zip'):
            extract_zip(path, self.raw_dir)
            os.unlink(path)
        elif path.endswith(('.tar.gz', '.tgz')):
            extract_tar(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        data_file = self._raw_data_file()
        if data_file is None:
            raise FileNotFoundError(
                f"Expected data.pkl or data.pkl.gz under {self.raw_dir}")

        raw_samples = self._read_pickle(data_file)
        split_indices = self._read_split_indices()
        validate_molecular_splits(len(raw_samples), split_indices)
        data_list = [
            self._sample_to_graph(sample, index)
            for index, sample in enumerate(raw_samples)
        ]

        if self.pre_filter is not None:
            raise ValueError(
                "pre_filter is not supported for benchmark datasets because it "
                "would invalidate the fixed official split indices.")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save_data(self.collate(data_list), self.processed_paths[0])
        with open(self.processed_paths[1], 'w', encoding='utf-8') as f:
            json.dump(split_indices, f)

    def get_idx_split(self) -> Dict[str, List[int]]:
        return {key: list(value) for key, value in self.split_indices.items()}

    def get_split(self, split: str):
        if split not in self.split_indices:
            expected = ', '.join(sorted(self.split_indices))
            raise ValueError(f"Unknown split '{split}'. Expected one of: {expected}.")
        return self.index_select(self.split_indices[split])

    def _load_processed_splits(self) -> Dict[str, List[int]]:
        with open(self.processed_paths[1], 'r', encoding='utf-8') as f:
            split_indices = {key: [int(index) for index in value]
                             for key, value in json.load(f).items()}
        validate_molecular_splits(len(self), split_indices)
        return split_indices

    def _read_pickle(self, path: str):
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rb') as f:
            return pickle.load(f)

    def _read_split_indices(self) -> Dict[str, List[int]]:
        split_indices = {}
        for split in ('train', 'val', 'test'):
            path = osp.join(self.raw_dir, f'{split}_index.json')
            if not osp.exists(path):
                raise FileNotFoundError(f"Split file not found: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                split_indices[split] = [int(index) for index in json.load(f)]
        return split_indices

    def _sample_to_graph(self, sample, index: int) -> Graph:
        if isinstance(sample, Graph):
            graph = self._graph_from_mapping({
                'edge_index': getattr(sample, 'edge_index', None),
                'x': getattr(sample, 'x', None),
                'edge_attr': getattr(sample, 'edge_attr', None),
            })
            label_data = getattr(sample, 'y', None)
        elif isinstance(sample, dict):
            graph = self._graph_from_mapping(sample)
            label_data = sample.get('properties', sample.get('y', sample.get('label', sample.get('labels'))))
        elif isinstance(sample, tuple) and len(sample) >= 2:
            graph = self._graph_from_mapping(sample[0])
            label_data = sample[1]
        else:
            raise ValueError(f"Unsupported sample format at index {index}: {type(sample)!r}")

        graph.y = tlx.convert_to_tensor([self._extract_label(label_data)], dtype=tlx.float32)
        return graph

    def _graph_from_mapping(self, graph_data) -> Graph:
        if isinstance(graph_data, Graph):
            return graph_data
        if not isinstance(graph_data, dict) and hasattr(graph_data, 'edges'):
            src, dst = graph_data.edges()
            node_data = getattr(graph_data, 'ndata', {})
            edge_data = getattr(graph_data, 'edata', {})
            node_name, node_values = self._first_present(
                node_data, ('node_token_ids', 'x', 'attr'))
            edge_name, edge_values = self._first_present(
                edge_data, ('edge_token_ids', 'edge_attr'))
            graph_data = {
                'edge_index': [self._to_list(src), self._to_list(dst)],
                node_name: node_values,
                edge_name: edge_values,
            }
            graph_data = {key: value for key, value in graph_data.items()
                          if key is not None and value is not None}
        if not isinstance(graph_data, dict):
            raise ValueError(f"Unsupported graph object: {type(graph_data)!r}")

        edge_index = graph_data.get('edge_index')
        if edge_index is None and 'edges' in graph_data:
            edge_index = graph_data['edges']

        x_name, x = self._first_present(
            graph_data,
            ('node_token_ids', 'node_type_ids', 'x', 'node_features', 'node_feat',
             'attr', 'node_labels'))
        edge_name, edge_attr = self._first_present(
            graph_data,
            ('edge_token_ids', 'edge_type_ids', 'edge_attr', 'edge_features',
             'edge_feat', 'edge_labels'))
        if x is None:
            raise ValueError("Graph is missing paper-required node features.")
        if edge_attr is None:
            raise ValueError("Graph is missing paper-required edge features.")

        return Graph(
            x=tlx.convert_to_tensor(
                self._encode_feature_ids(x_name, x, is_edge=False), dtype=tlx.int64),
            edge_index=tlx.convert_to_tensor(self._normalize_edge_index(edge_index), dtype=tlx.int64),
            edge_attr=tlx.convert_to_tensor(
                self._encode_feature_ids(edge_name, edge_attr, is_edge=True), dtype=tlx.int64),
            y=None,
        )

    def _extract_label(self, label_data) -> List[float]:
        if isinstance(label_data, dict):
            if self.label_keys == ('labels',):
                if 'labels' not in label_data:
                    raise ValueError("Label mapping is missing required 'labels' field.")
                return self._as_float_list(label_data['labels'])
            missing = [key for key in self.label_keys if key not in label_data]
            if missing:
                if self.name == 'qm9':
                    raise ValueError(
                        "QM9 sample must provide exactly 16 QM9 properties; "
                        f"missing: {', '.join(missing)}.")
                raise ValueError(f"Label mapping is missing required fields: {', '.join(missing)}.")
            return self._as_float_list([label_data[key] for key in self.label_keys])
        return self._as_float_list(label_data)

    def _as_float_list(self, value) -> List[float]:
        if value is None:
            raise ValueError(f"{self.display_name} sample is missing its label.")
        values = self._to_list(value)
        while self._is_sequence(values) and len(values) == 1 and self._is_sequence(values[0]):
            values = values[0]
        if not self._is_sequence(values):
            values = [values]
        if len(values) != self.num_tasks:
            raise ValueError(
                f"{self.display_name} labels must have exactly {self.num_tasks} values; "
                f"received {len(values)}.")
        result = [float(item) for item in values]
        if not self.allow_nan_labels and any(math.isnan(item) for item in result):
            raise ValueError(f"{self.display_name} labels cannot contain NaN values.")
        return result

    @staticmethod
    def _first_present(mapping, names: Sequence[str]):
        for name in names:
            if name in mapping:
                return name, mapping[name]
        return None, None

    @classmethod
    def _normalize_edge_index(cls, edge_index) -> List[List[int]]:
        if edge_index is None:
            return [[], []]
        values = cls._to_list(edge_index)
        if len(values) == 2 and cls._is_sequence(values[0]) and cls._is_sequence(values[1]):
            return [[int(value) for value in values[0]], [int(value) for value in values[1]]]
        src = [int(pair[0]) for pair in values]
        dst = [int(pair[1]) for pair in values]
        return [src, dst]

    def _encode_feature_ids(self, field_name: str, values, is_edge: bool) -> List[int]:
        values = self._to_list(values)
        if not self._is_sequence(values):
            raise ValueError(f"{field_name} must be a sequence of feature values.")

        if field_name.endswith('_token_ids'):
            return [self._scalar_token_id(field_name, item) for item in values]

        encoded = []
        for position, item in enumerate(values):
            raw_type = self._extract_feature_type(field_name, item, is_edge)
            if raw_type < 0:
                raise ValueError(
                    f"{field_name}[{position}] must be non-negative; received {raw_type}.")
            encoded.append(2 * raw_type if is_edge else 2 * raw_type + 1)
        return encoded

    def _scalar_token_id(self, field_name: str, item) -> int:
        item = self._to_list(item)
        if self._is_sequence(item):
            if len(item) != 1 or self._is_sequence(item[0]):
                raise ValueError(
                    f"{field_name} must contain one scalar token per node or edge.")
            item = item[0]
        token_id = int(item)
        if token_id < 0:
            raise ValueError(f"{field_name} token IDs must be non-negative.")
        return token_id

    def _extract_feature_type(self, field_name: str, item, is_edge: bool) -> int:
        item = self._to_list(item)
        if not self._is_sequence(item):
            return int(item)
        if not item:
            raise ValueError(f"{field_name} contains an empty feature row.")

        if is_edge and self._is_one_hot(item):
            return int(max(range(len(item)), key=lambda index: item[index])) + 1

        columns = self.edge_feature_columns if is_edge else self.node_feature_columns
        column = columns.get(field_name, 0)
        if column >= len(item):
            raise ValueError(
                f"{field_name} requires feature column {column}, but row has {len(item)} values.")
        return int(item[column])

    @staticmethod
    def _is_one_hot(values) -> bool:
        try:
            numeric = [float(value) for value in values]
        except (TypeError, ValueError):
            return False
        return all(value in (0.0, 1.0) for value in numeric) and sum(numeric) == 1.0

    @staticmethod
    def _to_list(value):
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        if hasattr(value, 'tolist'):
            return value.tolist()
        return value

    @staticmethod
    def _is_sequence(value) -> bool:
        return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))

    def __repr__(self) -> str:
        return f'{self.display_name}({len(self)})'
