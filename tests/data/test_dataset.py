# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/5/4

import os

import pytest
import tensorlayerx as tlx

from gammagl.data import Graph
from gammagl.datasets.ppi import PPI
from gammagl.data.dataset import Dataset


# dataset record to avoid downloading repeatedly


@pytest.mark.skipif(
    os.environ.get("GAMMAGL_RUN_DATASET_DOWNLOADS") != "1",
    reason="PPI dataset test downloads external data",
)
def test_dataset():
    dataset1 = PPI()
    dataset2 = PPI('./data')

    assert len(dataset1) == 20
    assert len(dataset2) == 20


def test_torch_dataset_loader_restores_saved_graph(tmp_path):
    import torch

    path = tmp_path / 'graph.pt'
    graph = Graph(
        x=tlx.convert_to_tensor([[1.0], [2.0]]),
        edge_index=tlx.convert_to_tensor([[0], [1]]),
    )
    torch.save((graph, None), path)
    dataset = object.__new__(Dataset)

    restored, slices = dataset.load_data(path)

    assert isinstance(restored, Graph)
    assert slices is None
    assert tlx.convert_to_numpy(restored.x).tolist() == [[1.0], [2.0]]
