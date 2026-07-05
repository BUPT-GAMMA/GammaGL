# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/5/4

import os

import pytest

from gammagl.datasets.ppi import PPI


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
