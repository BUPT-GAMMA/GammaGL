"""Helpers for applying Geom-GCN 10-split evaluation to GammaGL Planetoid."""

import os
import os.path as osp

import numpy as np
import tensorlayerx as tlx

from gammagl.data import download_url
from gammagl.datasets import Planetoid


GEOM_GCN_URL = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits"


def _geom_raw_dir(root, name):
    return osp.join(root, name.lower(), "geom-gcn", "raw")


def _split_file(name, split_id):
    return "{}_split_0.6_0.2_{}.npz".format(name.lower(), split_id)


def ensure_geom_gcn_splits(root, name, num_splits=10):
    """Ensure Geom-GCN split files exist under the GammaGL dataset directory."""
    raw_dir = _geom_raw_dir(root, name)
    os.makedirs(raw_dir, exist_ok=True)
    for split_id in range(num_splits):
        filename = _split_file(name, split_id)
        path = osp.join(raw_dir, filename)
        if not osp.exists(path):
            download_url("{}/{}".format(GEOM_GCN_URL, filename), raw_dir)
    return raw_dir


def load_planetoid_with_geom_splits(root, name, num_splits=10):
    """Load Planetoid data and replace masks with Geom-GCN fixed splits."""
    dataset = Planetoid(root=root, name=name)
    graph = dataset[0]
    raw_dir = ensure_geom_gcn_splits(root, name, num_splits=num_splits)

    train_masks, val_masks, test_masks = [], [], []
    for split_id in range(num_splits):
        split_path = osp.join(raw_dir, _split_file(name, split_id))
        split_data = np.load(split_path)
        train_masks.append(split_data["train_mask"])
        val_masks.append(split_data["val_mask"])
        test_masks.append(split_data["test_mask"])

    graph.train_mask = tlx.convert_to_tensor(np.stack(train_masks, axis=1), dtype=tlx.bool)
    graph.val_mask = tlx.convert_to_tensor(np.stack(val_masks, axis=1), dtype=tlx.bool)
    graph.test_mask = tlx.convert_to_tensor(np.stack(test_masks, axis=1), dtype=tlx.bool)
    return dataset, graph
