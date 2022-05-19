# Gamma Graph Library(GammaGL)

[![Documentation Status](https://readthedocs.org/projects/gammagl/badge/?version=latest)](https://gammagl.readthedocs.io/en/latest/?badge=latest)

**[Documentation](https://gammagl.readthedocs.io/en/latest/)** |

GammaGL is a multi-backend graph learning library based on [TensorLayerX](https://github.com/tensorlayer/TensorLayerX), which supports TensorFlow, PyTorch, PaddlePaddle, MindSpore as the backends.

It is under development, welcome join us!

We give a development tutorial in Chinese on [wiki](https://github.com/BUPT-GAMMA/GammaGL/wiki/%E5%BC%80%E5%8F%91%E8%80%85%E6%B5%81%E7%A8%8B).

## Highlighted Features

### Multi-backend

GammaGL supports multiple deep learning backends, such as TensorFlow, PyTorch, Paddle and MindSpore. Different from DGL, the GammaGL's examples are implemented with **the same code** on different backend. It allows users to run the same code on different hardwares like Nvidia-GPU and Huawei-Ascend. Besides, users could use a particular framework API based on preferences for different frameworks.

### PyG-Like

Following [PyTorch Geometric(PyG)](https://github.com/pyg-team/pytorch_geometric), GammaGL utilizes a tensor-centric API. If you are familiar with PyG, it will be friendly and maybe a TensorFlow Geometric, Paddle Geometric, or MindSpore Geometric to you.

## Quick Tour for New Users

In this quick tour, we highlight the ease of creating and training a GNN model with only a few lines of code.

### Train your own GNN model

In the first glimpse of GammaGL, we implement the training of a GNN for classifying papers in a citation graph.
For this, we load the [Cora](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid) dataset, and create a simple 2-layer GCN model using the pre-defined [`GCNConv`](https://github.com/BUPT-GAMMA/GammaGL/blob/main/gammagl/layers/conv/gcn_conv.py):

```python
import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv
from gammagl.datasets import Planetoid

dataset = Planetoid(root='.', name='Cora')

class GCN(tlx.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = tlx.ReLU()

    def forward(self, x, edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 16, dataset.num_classes)
```

<details>
<summary>We can now optimize the model in a training loop, similar to the <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation">standard PyTorch training procedure</a>.</summary>

```python
import torch.nn.functional as F

data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

</details>

More information about evaluating final model performance can be found in the corresponding [example]((https://github.com/BUPT-GAMMA/GammaGL/tree/main/examples/gcn).

### Create your own GNN layer



## Get Started

1. **Python environment** (Optional): We recommend using Conda package manager
   
   ```bash
   # python=3.7.5 or 3.9.0 is suitable for mindspore.
   conda create -n openhgnn python=3.7.5
   source activate openhgnn
   ```

2. **Install Backend**
   
   ```bash
   # tensorflow
   pip install tensorflow-gpu # GPU version
   pip install tensorflow # CPU version
   # torch
   # paddle
   # mindspore
   ```
   
   For other backend with specific version, [please check whether TLX supports](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-backend).
   
   [Install TensorLayerX](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-tensorlayerx)
   
   ```bash
   pip install git+https://github.com/tensorlayer/tensorlayerx.git 
   ```
   
   *Note*: use `pip install git+https://gitee.com/clearhanhui/TensorLayerX` for network problem. But it may not be the latest.

3. **Download GammaGL**
   
   ```bash
   git clone https://github.com/BUPT-GAMMA/GammaGL.git
   python setup.py install
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
> Set `CUDA_VISIBLE_DEVICES=" "` if you want to run it in CPU.

## Supported Models

|                                                  | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| ------------------------------------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| [GCN [ICLR 2017]](./examples/gcn)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GAT [ICLR 2018]](./examples/gat)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GraphSAGE [NeurIPS 2017]](./examples/graphsage) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [SGC [ICML 2019]](./examples/sgc)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GATv2 [ICLR 2021]](./examples/gatv2)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [APPNP [ICLR 2019]](./examples/appnp)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GCNII [ICLR 2017]](./examples/gcnii)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [AGNN [arxiv]](./examples/agnn)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [DGI [ICLR 2019]](./examples/dgi)                | :heavy_check_mark: |                    |                    |                    |
| [SIGN [ICML 2020 Workshop]](./examples/sign)     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [GRACE [ICML 2020 Workshop]](./examples/grace)   | :heavy_check_mark: |                    |                    |                    |
|                                                  |                    |                    |                    |                    |
