# Gamma Graph Library(GammaGL)

GammaGL is a framework-agnostic graph learning library based on [TensorLayerX](https://github.com/tensorlayer/TensorLayerX), which supports TensorFlow, PyTorch, PaddlePaddle, MindSpore as the backends.

Inspired by [PyTorch Geometric(PyG)](https://github.com/pyg-team/pytorch_geometric), GammaGL is Tensor-centric. If you are familiar with PyG, it will be friendly and maybe a TensorFlow Geometric, Paddle Geometric, or MindSpore Geometric to you.

It is under development, welcome join us!

We give a Chinese development procedure on [wiki](https://github.com/BUPT-GAMMA/GammaGL/wiki/%E5%BC%80%E5%8F%91%E8%80%85%E6%B5%81%E7%A8%8B).

## Get Started

* Install Backend
  
  ```bash
  pip install tensorflow-gpu # GPU version
  pip install tensorflow # CPU version
  ```
  
  Other backend with specific version, [check whether TLX supports](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-backend).

* [Install TensorLayerX](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-tensorlayerx)
  
  ```bash
  pip install git+https://github.com/tensorlayer/tensorlayerx.git 
  ```
  
  *Note*: You may need to install `cv2` with command `pip install opencv-python`.

* Download GammaGL
  
  ```bash
  git clone https://github.com/BUPT-GAMMA/GammaGL.git
  ```

## Supported Models

|                 | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| --------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| GCN [ICLR 2017] | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| G               |                    |                    |                    |                    |
|                 |                    |                    |                    |                    |
