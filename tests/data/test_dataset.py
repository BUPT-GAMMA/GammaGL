# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/5/4

from gammagl.datasets.ppi import PPI


# dataset record to avoid downloading repeatedly


def test_dataset():
    dataset1 = PPI()
    dataset2 = PPI('./data')

    assert len(dataset1) == 20
    assert len(dataset2) == 20

