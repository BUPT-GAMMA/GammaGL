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

## How to Run

Take [GCN](./examples/gcn) as an example:

```bash
# cd ./examples/gcn
# set parameters if necessary
python gcn_trainer.py --dataset cora --lr 0.01
```

If you want to use specific `backend` or `GPU`, just set environment variable like:

```bash
CUDA_VISIBLE_DEVICES="1" TL_BACKEND="paddle" python gcn_trainer.py
```

> Note
> 
> The DEFAULT backend is  `tensorflow` and GPU is `0`. The backend TensorFlow will take up all GPU left memory by default.
> 
> The CANDIDATE backends are `tensorflow`, `paddle`, `torch` and `mindspore`.
> 
> Set `CUDA_VISIBLE_DEVICES=""` if you want to run it in CPU.

## Supported Models

|                                                | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| [GCN [ICLR 2017]](./examples/gcn)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GAT [ICLR 2018]](./examples/gat)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GraphSAGE [NeurIPS 2017]](./examples/sage)    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [SGC [ICML 2019]](./examples/sgc)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GATv2 [ICLR 2021]](./examples/gatv2)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [APPNP [ICLR 2019]](./examples/appnp)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GCNII [ICLR 2017]](./examples/gcnii)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [AGNN [arxiv]](./examples/agnn)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [DGI [ICLR 2019]](./examples/dgi)              | :heavy_check_mark: |                    |                    |                    |
| [SIGN [ICML 2020 Workshop]](./examples/sign)   | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| [GRACE [ICML 2020 Workshop]](./examples/grace) | :heavy_check_mark: |                    |                    |                    |
|                                                |                    |                    |                    |                    |
