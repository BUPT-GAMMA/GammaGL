import os
import numpy as np
import tensorlayerx as tlx
import scipy.sparse as sp
from typing import Callable, List, Optional
from gammagl.datasets import Planetoid, Flickr
# from ..blogcatalog import BlogCatalog
import gammagl.transforms as T


class DataSet(object):
    def __init__(self, path: str, name: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super(DataSet, self).__init__()
        self.path = path
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform

    def load_dataset(self):
        if self.name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=self.path, name=self.name, transform=self.transform,
                                pre_transform=self.pre_transform)
        elif self.name == 'Flickr':
            dataset = Flickr(root=self.path, transform=self.transform, pre_transform=self.pre_transform)
        elif self.name == 'blog':
            dataset = Blogcatalog(root=self.path, transform=self.transform, pre_transform=self.pre_transform)

        graph = dataset[0]

        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask

        num_class = dataset.num_classes
        feat = graph.x
        labels = graph.y
        edge_index = graph.edge_index
        return edge_index, feat, labels, num_class, train_mask, val_mask, test_mask
