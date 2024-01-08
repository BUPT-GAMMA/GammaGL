# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/3/2


from abc import ABC, abstractmethod


class BaseSampler:

    @abstractmethod
    def sample_from_nodes(self, index, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sample_from_edges(self, index, **kwargs):
        raise NotImplementedError

    @property
    def edge_permutation(self):
        return None
