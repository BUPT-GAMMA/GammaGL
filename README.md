# Gamma Graph Library(GammaGL)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BUPT-GAMMA/GammaGL)
[![Documentation Status](https://readthedocs.org/projects/gammagl/badge/?version=latest)](https://gammagl.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/BUPT-GAMMA/GammaGL)
![visitors](https://visitor-badge.glitch.me/badge?page_id=BUPT-GAMMA.GammaGL)
![GitHub all releases](https://img.shields.io/github/downloads/BUPT-GAMMA/GammaGL/total)
![Total lines](https://img.shields.io/tokei/lines/github/BUPT-GAMMA/GammaGL?color=red)

**[Documentation](https://gammagl.readthedocs.io/en/latest/)** |**[启智社区](https://git.openi.org.cn/GAMMALab/GammaGL)**

GammaGL is a multi-backend graph learning library based on [TensorLayerX](https://github.com/tensorlayer/TensorLayerX), which supports TensorFlow, PyTorch, PaddlePaddle, MindSpore as the backends.

We release the version 0.1.0 on 20th June. 

We give a development tutorial in Chinese on [wiki](https://github.com/BUPT-GAMMA/GammaGL/wiki/%E5%BC%80%E5%8F%91%E8%80%85%E6%B5%81%E7%A8%8B).

## Highlighted Features

### Multi-backend

GammaGL supports multiple deep learning backends, such as TensorFlow, PyTorch, Paddle and MindSpore. Different from DGL, the GammaGL's examples are implemented with **the same code** on different backend. It allows users to run the same code on different hardwares like Nvidia-GPU and Huawei-Ascend. Besides, users could use a particular framework API based on preferences for different frameworks.

### PyG-Like

Following [PyTorch Geometric(PyG)](https://github.com/pyg-team/pytorch_geometric), GammaGL utilizes a tensor-centric API. If you are familiar with PyG, it will be friendly and maybe a TensorFlow Geometric, Paddle Geometric, or MindSpore Geometric to you.

## News
<details>

<summary>
2023-02-24 优秀孵化奖
</summary>
<br/>

GammaGL荣获启智社区优秀孵化项⽬奖！详细链接：https://mp.weixin.qq.com/s/PpbwEdP0-8wG9dsvRvRDaA

</details>

<details>

<summary>
2023-02-21 中国电子学会科技进步一等奖
</summary>
<br/>

算法库支撑了北邮牵头，蚂蚁、中移动、海致科技等参与的“大规模复杂异质图数据智能分析技术与规模化应用”项目。该项目获得了2022年电子学会科技进步一等奖。

</details>

<details>
<summary>2023-01-17 release v0.2
</summary>
</br>
We release the latest version v0.2.

- 40 GNN models
- 20 datasets
- Efficient message passing operators and fused operator
- GPU sampling and heterogeneous graphs samplers.

</details>

<details>
<summary>2022-06-20 release v0.1
</summary>
</br>
We release the latest version v0.1.

- Framework-agnostic design
- PyG-like
- Graph data structures, message passing module and sampling module
- 20+ GNN models

</details>

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
<summary>
We can now optimize the model in a training loop, similar to the <a href="https://tensorlayerx.readthedocs.io/en/latest/modules/model.html#trainonestep">standard TensorLayerX training procedure</a>.</summary>

```python
import tensorlayerx as tlx
data = dataset[0]
loss_fn = tlx.losses.softmax_cross_entropy_with_logits
optimizer = tlx.optimizers.Adam(learning_rate=1e-3)
net_with_loss = tlx.model.WithLoss(model, loss_fn)
train_one_step = tlx.model.TrainOneStep(net_with_loss, optimizer, train_weights)

for epoch in range(200):
    loss = train_one_step(data.x, data.y)
```

</details>

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

In addition to the easy application of existing GNNs, GammaGL makes it simple to implement custom Graph Neural Networks (see [here](https://gammagl.readthedocs.io/en/latest/notes/create_gnn.html) for the accompanying tutorial).
For example, this is all it takes to implement the [edge convolutional layer](https://arxiv.org/abs/1801.07829) from Wang *et al.*:

$$x_i^{\prime} ~ = ~ \max_{j \in \mathcal{N}(i)} ~ \textrm{MLP}_{\theta} \left( [ ~ x_i, ~ x_j - x_i ~ ] \right)$$

```python
import tensorlayerx as tlx
from tensorlayerx.nn import Sequential as Seq, Linear, ReLU
from gammagl.layers import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(x=x, edge_index,aggr_type='max')

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = tlx.concat([x_i, x_j - x_i], axis=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
```

## Get Started

1. **Python environment** (Optional): We recommend using Conda package manager
   
   ```bash
   conda create -n ggl python=3.7
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
   
   # For mindspore, GammaGL only supports version 1.8.1, GPU-CUDA 11.1
   # https://www.mindspore.cn/install
   pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.1/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-1.8.1-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   
   For other backend with specific version, [please check whether TLX supports](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-backend).
   
   [Install TensorLayerX](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html#install-tensorlayerx)
   
   ```bash
   pip install git+https://github.com/tensorlayer/tensorlayerx.git 
   ```
   
   > 大陆用户如果遇到网络问题，推荐从启智社区安装
   > 
   > Try to git clone from OpenI
   > 
   > `pip install git+https://git.openi.org.cn/OpenI/TensorLayerX.git`

3. **Download GammaGL**
   
   ```bash
   git clone https://github.com/BUPT-GAMMA/GammaGL.git
   python setup.py install
   ```
   
   > 大陆用户如果遇到网络问题，推荐从启智社区安装
   > 
   > Try to git clone from OpenI
   > 
   > `git clone https://git.openi.org.cn/GAMMALab/GammaGL.git`

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
| [ChebNet [NeurIPS 2016]](./examples/chebnet)     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [GCNII [ICLR 2017]](./examples/gcnii)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [JKNet [ICML 2018]](./examples/jknet)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [DiffPool [NeurIPS 2018]](./examples/diffpool)   |                    |                    |                    |                    |
| [SGC [ICML 2019]](./examples/sgc)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GIN [ICLR 2019]](./examples/gin)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [APPNP [ICLR 2019]](./examples/appnp)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [AGNN [arxiv]](./examples/agnn)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [SIGN [ICML 2020 Workshop]](./examples/sign)     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [DropEdge [ICLR 2020]](./examples/dropedge)      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GPRGNN [ICLR 2021]](./examples/gprgnn)          | :heavy_check_mark: |                    |                    |                    |
| [GNN-FiLM [ICML 2020]](./examples/film)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [GraphGAN [AAAI 2018]](./examples/graphgan)      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [HardGAT [KDD 2019]](./examples/hardgat)         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [MixHop [ICML 2019]](./examples/mixhop)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [PNA [NeurIPS 2020]](./examples/pna)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [FAGCN [AAAI 2021]](./examples/fagcn)            | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |
| [GATv2 [ICLR 2021]](./examples/gatv2)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [GEN [WWW 2021]](./examples/gen)                 | :heavy_check_mark: | :heavy_check_mark: |                    |                    |
| [GAE [NeurIPS 2016]](./examples/vgae)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [VGAE [NeurIPS 2016]](./examples/vgae)           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [HCHA [PR 2021]](./examples/hcha)                |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Node2Vec [KDD 2016]](./examples/node2vec)       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [DGCNN [ACM T GRAPHIC 2019]](./examples/dgcnn)   | :heavy_check_mark: | :heavy_check_mark: |                    |                    |


| Contrastive Learning                           | TensorFlow         | PyTorch            | Paddle             | MindSpore |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------ | --------- |
| [DGI [ICLR 2019]](./examples/dgi)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [GRACE [ICML 2020 Workshop]](./examples/grace) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [MVGRL [ICML 2020]](./examples/mvgrl)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [InfoGraph [ICLR 2020]](./examples/infograph)  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [MERIT [IJCAI 2021]](./examples/merit)         | :heavy_check_mark: |                    | :heavy_check_mark: |           |

| Heterogeneous Graph Learning                 | TensorFlow         | PyTorch            | Paddle             | MindSpore |
| -------------------------------------------- | ------------------ | ------------------ | ------------------ | --------- |
| [RGCN [ESWC 2018]](./examples/rgcn)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [HAN [WWW 2019]](./examples/han)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HGT [WWW 2020]](./examples/hgt/)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |           |
| [SimpleHGN [KDD 2021]](./examples/simplehgn) | :heavy_check_mark: |                    |                    |           |
| [CompGCN [ICLR 2020]](./examples/compgcn)    |                    | :heavy_check_mark: | :heavy_check_mark: |           |
| [HPN [TKDE 2021]](./examples/hpn)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

> Note
> 
> The models can be run in mindspore backend. Howerver, the results of experiments are not satisfying due to training component issue,
> which will be fixed in future.

## Contributors

GammaGL Team[GAMMA LAB] and Peng Cheng Laboratory.

See more in [CONTRIBUTING](./CONTRIBUTING.md).

Contribution is always welcomed. Please feel free to open an issue or email to tyzhao@bupt.edu.cn.
