import math
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from gammagl.loader import DataLoader


class GINDataLoader:
    def __init__(self, dataset, batch_size, collate_fn=None, seed=0, shuffle=True, split_name='fold10', fold_idx=0,
                 split_ratio=0.7):
        self.shuffle = shuffle
        self.seed = seed

        labels = dataset.data.y

        if split_name == 'fold10':
            train_idx, valid_idx = self._split_fold10(labels=labels, fold_idx=fold_idx, seed=seed, shuffle=shuffle)
        elif split_name == 'rand':
            train_idx, valid_idx = self._split_rand(labels=labels, split_ratio=split_ratio, seed=seed, shuffle=shuffle)
        else:
            raise NotImplementedError()

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size=batch_size,
                                       collate_fn=collate_fn)
        self.valid_loader = DataLoader(dataset=dataset, sampler=valid_sampler, batch_size=batch_size,
                                       collate_fn=collate_fn)

    def train_valid_loader(self):
        return self.train_loader, self.valid_loader

    def _split_fold10(self, labels, fold_idx=0, seed=0, shuffle=True):
        """ 10 fold """
        assert 0 <= fold_idx < 10, print("fold_idx must be from 0 to 9.")

        skf = StratifiedKFold(n_splits=10, shuffle=shuffle, random_state=seed)
        idx_list = []
        for idx in skf.split(X=tlx.zeros(len(labels)), y=labels):
            idx_list.append(idx)
        train_idx, valid_idx = idx_list[fold_idx]

        print("train_set : test_set = %d : %d", len(train_idx), len(valid_idx))

        return train_idx, valid_idx

    def _split_rand(self, labels, split_ratio=0.7, seed=0, shuffle=True):
        num_labels = len(labels)
        indices = list(range(num_labels))
        np.random.seed(seed=seed)
        np.random.shuffle(x=indices)
        split = int(math.floor(split_ratio * num_labels))
        train_idx, valid_idx = indices[:split], indices[split:]

        print("train_set : test_set = %d : %d", len(train_idx), len(valid_idx))

        return train_idx, valid_idx
