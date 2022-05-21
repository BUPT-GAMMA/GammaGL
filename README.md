# Gamma Graph Library(GammaGL)

GammaGL is a framework-agnostic graph learning library based on [TensorLayerX](https://github.com/tensorlayer/TensorLayerX), which supports TensorFlow, PyTorch, PaddlePaddle, MindSpore as the backends.

Inspired by [PyTorch Geometric(PyG)](https://github.com/pyg-team/pytorch_geometric), GammaGL is Tensor-centric. If you are familiar with PyG, it will be friendly and maybe a TensorFlow Geometric, Paddle Geometric, or MindSpore Geometric to you.

It is under development, welcome join us!

We give a development tutorial in Chinese on [wiki](https://github.com/BUPT-GAMMA/GammaGL/wiki/%E5%BC%80%E5%8F%91%E8%80%85%E6%B5%81%E7%A8%8B).

## Get Started

* Install Backend
  
  ```bash
  pip install tensorflow
  ```
  
  For other backend with specific version, [please check whether TLX supports](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-backend).

* [Install TensorLayerX](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-tensorlayerx)
  
  ```bash
  pip install git+https://github.com/tensorlayer/tensorlayerx.git 
  ```
  
  *Note*: use `pip install git+https://gitee.com/clearhanhui/TensorLayerX` for network problem. But it may not be the latest. 

* Download GammaGL
  
  ```bash
  git clone https://github.com/BUPT-GAMMA/GammaGL.git
  ```

## Supported Models

|                                   | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| --------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| [GCN [ICLR 2017]](./examples/gcn) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|                                   |                    |                    |                    |                    |
|                                   |                    |                    |                    |                    |
