.. gammagl documentation master file, created by
   sphinx-quickstart on Thu Mar 31 13:35:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gammagl's documentation!
===================================

.. toctree::
   :maxdepth: 4
   :caption: NOTES
   :hidden:
   :glob:

   notes/installation
   notes/introduction
   notes/create_dataset
   notes/create_gnn
   notes/batching

.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:
   :glob:
   
   api/gammagl.data
   api/gammagl.datasets
   api/gammagl.layers
   api/gammagl.loader
   api/gammagl.transforms
   api/gammagl.utils

GammaGL is a framework-agnostic graph learning library based on `TensorLayerX <https://github.com/tensorlayer/TensorLayerX>`_, which supports TensorFlow, PyTorch, PaddlePaddle, MindSpore as the backends.
Inspired by `PyG[PyTorch Geometric] <https://github.com/pyg-team/pytorch_geometric>`_, GammaGL is Tensor-centric. If you are familiar with PyG, it will be friendly and maybe a TensorFlow Geometric, Paddle Geometric, or MindSpore Geometric to you.

It is under development, welcome join us!

We give a development tutorial in Chinese on `wiki <https://github.com/BUPT-GAMMA/GammaGL/wiki/%E5%BC%80%E5%8F%91%E8%80%85%E6%B5%81%E7%A8%8B>`_.