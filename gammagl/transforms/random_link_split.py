import warnings
from copy import copy
from typing import List, Optional, Union

import tensorlayerx as tlx
import numpy as np

from gammagl.data import Graph, HeteroGraph
from gammagl.transforms import BaseTransform
from gammagl.typing import EdgeType
from gammagl.utils import negative_sampling


class RandomLinkSplit(BaseTransform):
    def __init__(
        self,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.2,
        is_undirected: bool = False,
        key: str = 'edge_label',
        split_labels: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
        disjoint_train_ratio: Union[int, float] = 0.0,
        edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
        rev_edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
    ):
        if isinstance(edge_types, list):
            if rev_edge_types is None:
                rev_edge_types = [None] * len(edge_types)

            assert isinstance(rev_edge_types, list)
            assert len(edge_types) == len(rev_edge_types)

        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types

    def __call__(
        self,
        data: Union[Graph, HeteroGraph],
    ) -> Union[Graph, HeteroGraph]:
        edge_types = self.edge_types
        rev_edge_types = self.rev_edge_types

        train_data, val_data, test_data = copy(data), copy(data), copy(data)

        if isinstance(data, HeteroGraph):
            if edge_types is None:
                raise ValueError(
                    "The 'RandomLinkSplit' transform expects 'edge_types' to"
                    "be specified when operating on 'HeteroGraph' objects")

            if not isinstance(edge_types, list):
                edge_types = [edge_types]
                rev_edge_types = [rev_edge_types]

            stores = [data[edge_type] for edge_type in edge_types]
            train_stores = [train_data[edge_type] for edge_type in edge_types]
            val_stores = [val_data[edge_type] for edge_type in edge_types]
            test_stores = [test_data[edge_type] for edge_type in edge_types]
        else:
            rev_edge_types = [None]
            stores = [data._store]
            train_stores = [train_data._store]
            val_stores = [val_data._store]
            test_stores = [test_data._store]

        for item in zip(stores, train_stores, val_stores, test_stores,
                        rev_edge_types):
            store, train_store, val_store, test_store, rev_edge_type = item

            is_undirected = self.is_undirected
            is_undirected &= not store.is_bipartite()
            is_undirected &= (rev_edge_type is None
                              or store._key == data[rev_edge_type]._key)

            edge_index = store.edge_index
            if is_undirected:
                mask = edge_index[0] <= edge_index[1]
                perm = tlx.mask_select(tlx.arange(0, edge_index.shape[1]), mask)
                numel = perm.shape[0]
                perm = tlx.gather(perm, tlx.convert_to_tensor(np.random.permutation(numel)))
            else:
                numel = edge_index.shape[1]
                perm = tlx.convert_to_tensor(np.random.permutation(numel))

            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * numel)
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * numel)

            num_train = numel - num_val - num_test

            if num_train <= 0:
                raise ValueError("Insufficient number of edges for training")

            train_edges = perm[:num_train]
            val_edges = perm[num_train:num_train + num_val]
            test_edges = perm[num_train + num_val:]
            train_val_edges = perm[:num_train + num_val]

            num_disjoint = self.disjoint_train_ratio
            if isinstance(num_disjoint, float):
                num_disjoint = int(num_disjoint * num_train)
            if num_train - num_disjoint <= 0:
                raise ValueError("Insufficient number of edges for training")

            # Create data splits:
            self._split(train_store, train_edges[num_disjoint:], is_undirected,
                        rev_edge_type)
            self._split(val_store, train_edges, is_undirected, rev_edge_type)
            self._split(test_store, train_val_edges, is_undirected,
                        rev_edge_type)

            # Create negative samples:
            num_neg_train = 0
            if self.add_negative_train_samples:
                if num_disjoint > 0:
                    num_neg_train = int(num_disjoint * self.neg_sampling_ratio)
                else:
                    num_neg_train = int(num_train * self.neg_sampling_ratio)
            num_neg_val = int(num_val * self.neg_sampling_ratio)
            num_neg_test = int(num_test * self.neg_sampling_ratio)

            num_neg = num_neg_train + num_neg_val + num_neg_test

            size = store.size()
            if store._key is None or store._key[0] == store._key[-1]:
                size = size[0]
            neg_edge_index = negative_sampling(edge_index, size,
                                               num_neg_samples=num_neg,
                                               method='sparse')

            # Adjust ratio if not enough negative edges exist
            if neg_edge_index.shape[1] < num_neg:
                num_neg_found = neg_edge_index.shape[1]
                ratio = num_neg_found / num_neg
                warnings.warn(
                    f"There are not enough negative edges to satisfy "
                    "the provided sampling ratio. The ratio will be "
                    f"adjusted to {ratio:.2f}.")
                num_neg_train = int((num_neg_train / num_neg) * num_neg_found)
                num_neg_val = int((num_neg_val / num_neg) * num_neg_found)
                num_neg_test = num_neg_found - num_neg_train - num_neg_val

            # Create labels:
            if num_disjoint > 0:
                train_edges = train_edges[:num_disjoint]
            self._create_label(
                store,
                train_edges,
                neg_edge_index[:, num_neg_val + num_neg_test:],
                out=train_store,
            )
            self._create_label(
                store,
                val_edges,
                neg_edge_index[:, :num_neg_val],
                out=val_store,
            )
            self._create_label(
                store,
                test_edges,
                neg_edge_index[:, num_neg_val:num_neg_val + num_neg_test],
                out=test_store,
            )

        return train_data, val_data, test_data

    def _split(self, store, index, is_undirected, rev_edge_type):

        for key, value in store.items():
            if key == 'edge_index':
                continue

            if store.is_edge_attr(key):
                value = tlx.gather(value, index)
                if is_undirected:
                    value = tlx.concat([value, value], axis=0)
                store[key] = value

        edge_index = tlx.gather(store.edge_index, index, axis=1)
        if is_undirected:
            edge_index = tlx.concat([edge_index, tlx.gather(edge_index, tlx.convert_to_tensor([1,0]), axis=0)], axis=-1)
        store.edge_index = edge_index

        if rev_edge_type is not None:
            rev_store = store._parent()[rev_edge_type]
            for key in rev_store.keys():
                if key not in store:
                    del rev_store[key]  # We delete all outdated attributes.
                elif key == 'edge_index':
                    rev_store.edge_index = tlx.gather(store.edge_index, [1,0], axis=0)
                else:
                    rev_store[key] = store[key]

        return store

    def _create_label(self, store, index, neg_edge_index, out):

        edge_index = tlx.gather(store.edge_index, index, axis=1)

        if hasattr(store, self.key):
            edge_label = store[self.key]
            edge_label = edge_label[index]
            if neg_edge_index.shape[1] > 0:
                assert edge_label.shape[0] == store.edge_index.shape[1]
                edge_label += 1
            if hasattr(out, self.key):
                delattr(out, self.key)
        else:
            edge_label = tlx.ones((index.shape[0],))

        if neg_edge_index.shape[1] > 0:
            neg_edge_label = tlx.zeros((neg_edge_index.shape[1],) + tuple(edge_label.shape[1:]), dtype=edge_label.dtype)

        if self.split_labels:
            out[f'pos_{self.key}'] = edge_label
            out[f'pos_{self.key}_index'] = edge_index
            if neg_edge_index.shape[1] > 0:
                out[f'neg_{self.key}'] = neg_edge_label
                out[f'neg_{self.key}_index'] = neg_edge_index

        else:
            if neg_edge_index.shape[1] > 0:
                edge_label = tlx.concat([edge_label, neg_edge_label], axis=0)
                edge_index = tlx.concat([edge_index, neg_edge_index], axis=-1)
            out[self.key] = edge_label
            out[f'{self.key}_index'] = edge_index

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')