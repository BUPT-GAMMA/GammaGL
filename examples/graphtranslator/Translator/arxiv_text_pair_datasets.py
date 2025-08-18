"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
import random
from typing import Iterable
import pandas as pd
from tensorlayerx.dataflow import Dataset, ConcatDataset, IterableDataset

class BatchIterableDataset(IterableDataset):
    def __init__(self, cfg, mode):
        super(BatchIterableDataset, self).__init__()
        self._cfg = cfg
        self._mode = mode

        self.summary_embeddings = pd.read_csv(cfg['datasets_dir'])
        self.row_count = self.summary_embeddings.shape[0]
        self.start_pos = 0
        self.end_pos = self.summary_embeddings.shape[0]

        self._parser_dict = {
            "train": self._train_data_parser,
            "eval": self._eval_data_parser,
            "infer": self._infer_data_parser
        }

    def _train_data_parser(self, data):
        raise NotImplementedError

    def _eval_data_parser(self, data):
        raise NotImplementedError

    def _infer_data_parser(self, data):
        raise NotImplementedError

    def data_iterator(self):
        for _, row in self.summary_embeddings.iterrows():
            try:
                data = [tuple(row)]
            except Exception:
                break
            yield self._parser_dict[self._mode](data)

    def __iter__(self):
        return self.data_iterator()


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
    
class ArxivTextPairDataset(BatchIterableDataset):
    def __init__(self, cfg, mode):
        super(ArxivTextPairDataset, self).__init__(cfg, mode)
        self.max_length = cfg.arxiv_processor.train.max_length
        self.vocab_size = cfg.arxiv_processor.train.vocab_size

    def _train_data_parser(self, data):
        # 训练阶段使用
        user_id = data[0][0]
        embedding = np.array(data[0][1].split(','), dtype=np.float32)
        node_input = data[0][2]
        if len(node_input)>=1000:
            node_input=node_input[:1000]
        neighbour_input = data[0][3]
        title = data[0][4]

        text_input = 'The summary of this article is as follows:' + node_input+ '\nThere are some papers that cite this paper.' + neighbour_input

        return user_id, embedding, text_input, title

    def __len__(self):
        return self.row_count

class MultiIterLoader:
    """
    A simple wrapper for iterating over multiple iterators.

    Args:
        loaders (List[Loader]): List of Iterator loaders.
        ratios (List[float]): List of ratios to sample from each loader. If None, all loaders are sampled uniformly.
    """

    def __init__(self, loaders, ratios=None):
        # assert all loaders has __next__ method
        for loader in loaders:
            assert hasattr(
                loader, "__next__"
            ), "Loader {} has no __next__ method.".format(loader)

        if ratios is None:
            ratios = [1.0] * len(loaders)
        else:
            assert len(ratios) == len(loaders)
            ratios = [float(ratio) / sum(ratios) for ratio in ratios]

        self.loaders = loaders
        self.ratios = ratios

    def __next__(self):
        # random sample from each loader by ratio
        loader_idx = random.choices(range(len(self.loaders)), self.ratios, k=1)[0]
        return next(self.loaders[loader_idx])

import torch
import sys

MAX_INT = sys.maxsize


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    return samples