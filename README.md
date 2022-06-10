# Gamma Graph Library(GammaGL)

[![Documentation Status](https://readthedocs.org/projects/gammagl/badge/?version=latest)](https://gammagl.readthedocs.io/en/latest/?badge=latest)
[![visitors](https://visitor-badge.glitch.me/badge?page_id=BUPT-GAMMA.GammaGL)](https://github.com/BUPT-GAMMA/GammaGL)
[![Total lines](https://img.shields.io/tokei/lines/github/BUPT-GAMMA/GammaGL?color=red)](https://github.com/BUPT-GAMMA/GammaGL)

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

<details>
<summary>We can now optimize the model in a training loop, similar to the <a href="https://tensorflow.google.cn/tutorials/quickstart/advanced">standard TensorFlow training procedure</a>.</summary>

```python
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
for epoch in range(200):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

</details>

<details>
<summary>We can now optimize the model in a training loop, similar to the <a href="https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/train_eval_predict_cn.html#api">standard PaddlePaddle training procedure</a>.</summary>

```python
import paddle

data = dataset[0]
optim = paddle.optimizer.Adam(parameters=model.parameters())
loss_fn = paddle.nn.CrossEntropyLoss()

model.train()
for epoch in range(200):
    predicts = model(data.x, data.edge_index)
    loss = loss_fn(predicts, y_data)

    # Backpropagation
    loss.backward()
    optim.step()
    optim.clear_grad()
```

</details>

<details>
<summary>We can now optimize the model in a training loop, similar to the <a href="https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/train/train_eval.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E8%AE%AD%E7%BB%83%E5%92%8C%E8%AF%84%E4%BC%B0">standard MindSpore training procedure</a>.</summary>

```python
# 1. Generate training dataset
train_dataset = create_dataset(num_data=160, batch_size=16)

# 2.Build a model and define the loss function
net = LinearNet()
loss = nn.MSELoss()

# 3.Connect the network with loss function, and define the optimizer
net_with_loss = nn.WithLossCell(net, loss)
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

# 4.Define the training network
train_net = nn.TrainOneStepCell(net_with_loss, opt)

# 5.Set the model as training mode
train_net.set_train()

# 6.Training procedure
for epoch in range(200):
    for d in train_dataset.create_dict_iterator():
        result = train_net(d['data'], d['label'])
        print(f"Epoch: [{epoch} / {epochs}], "
              f"step: [{step} / {steps}], "
              f"loss: {result}")
        step = step + 1
```

</details>

More information about evaluating final model performance can be found in the corresponding [example](https://github.com/BUPT-GAMMA/GammaGL/tree/main/examples/gcn).

### Create your own GNN layer

## Get Started

1. **Python environment** (Optional): We recommend using Conda package manager
   
   ```bash
   # python=3.7.5 or 3.9.0 is suitable for mindspore.
   conda create -n ggl python=3.7.5
   source activate ggl
   ```

2. **Install Backend**
   
   ```bash
   # For tensorflow
   pip install tensorflow-gpu # GPU version
   pip install tensorflow # CPU version
   
   # For torch, version 1.10
   # https://pytorch.org/get-started/locally/
   pip3 install torch==1.10.2
   
   # For paddle, any latest stable version
   # https://www.paddlepaddle.org.cn/
   python -m pip install paddlepaddle-gpu
   
   # For mindspore, GammaGL only supports version1.6.1, GPU-CUDA 11.1 and python 3.7.5
   # https://www.mindspore.cn/install
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.6.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
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

|                                                  | TensorFlow         | PyTorch            | Paddle             | MindSpore |
| ------------------------------------------------ | ------------------ | ------------------ | ------------------ | --------- |
| [GCN [ICLR 2017]](./examples/gcn)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [GAT [ICLR 2018]](./examples/gat)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [GraphSAGE [NeurIPS 2017]](./examples/graphsage) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [ChebNet [NeurIPS 2016]](./examples/chebnet)     | :heavy_check_mark: | :heavy_check_mark: |                    |           |
| [GCNII [ICLR 2017]](./examples/gcnii)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [JKNet [ICML 2018]](./examples/jknet)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [DiffPool [NeurIPS 2018]](./examples/diffpool)   |                    |                    |                    |           |
| [SGC [ICML 2019]](./examples/sgc)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [GIN [ICLR 2019]](./examples/gin)                |                    |                    |                    |           |
| [APPNP [ICLR 2019]](./examples/appnp)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [AGNN [arxiv]](./examples/agnn)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [SIGN [ICML 2020 Workshop]](./examples/sign)     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [GATv2 [ICLR 2021]](./examples/gatv2)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [GPRGNN [ICLR 2021]](./examples/gprgnn)          | :heavy_check_mark: |                    |                    |           |
| [FAGCN [AAAI 2021]](./examples/fagcn)            | :heavy_check_mark: | :heavy_check_mark: |                    |           |

| Contrastive Learning                           | TensorFlow         | PyTorch            | Paddle             | MindSpore |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------ | --------- |
| [DGI [ICLR 2019]](./examples/dgi)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [GRACE [ICML 2020 Workshop]](./examples/grace) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [MVGRL [ICML 2020]](./examples/mvgrl)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [InfoGraph [ICLR 2020]](./examples/infograph)  | :heavy_check_mark: | :heavy_check_mark: |                    |           |
| [MERIT [IJCAI 2021]](./examples/merit)         | :heavy_check_mark: |                    |                    |           |

| Heterogeneous Graph Learning                 | TensorFlow         | PyTorch            | Paddle             | MindSpore |
| -------------------------------------------- | ------------------ | ------------------ | ------------------ | --------- |
| [RGCN [ESWC2018]](./examples/rgcn)           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [HAN [WWW 2019]](./examples/han)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| HGT [WWW 2020]                               |                    |                    |                    |           |
| [SimpleHGN [KDD 2021]](./examples/simplehgn) | :heavy_check_mark: |                    |                    |           |

## Contributors

GammaGL Team[GAMMA LAB] and Peng Cheng Laboratory.

See more in [CONTRIBUTING](./CONTRIBUTING.md).
