# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/5/4

import os

import pytest

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


def test_torch_dataset_loader_restores_graph_objects(monkeypatch):
    torch = __import__('torch')
    observed = {}

    def fake_load(path, **kwargs):
        observed['path'] = path
        observed.update(kwargs)
        return ('graph', None)

    monkeypatch.setattr(torch, 'load', fake_load)
    dataset = object.__new__(Dataset)

    assert dataset.load_data('trusted_graphs.pt') == ('graph', None)
    assert observed['path'] == 'trusted_graphs.pt'
    assert observed['weights_only'] is False
