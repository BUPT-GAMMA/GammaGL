# Gamma Graph Library(GammaGL)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/BUPT-GAMMA/GammaGL)
[![Documentation Status](https://readthedocs.org/projects/gammagl/badge/?version=latest)](https://gammagl.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/BUPT-GAMMA/GammaGL)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=BUPT-GAMMA.GammaGL)
![GitHub all releases](https://img.shields.io/github/downloads/BUPT-GAMMA/GammaGL/total)
![Code size](https://img.shields.io/github/languages/code-size/BUPT-GAMMA/GammaGL?color=red)

**[Documentation](https://gammagl.readthedocs.io/en/latest/)** |
**[Get Started](#get-started)** |
**[Quick Tour](#quick-tour-for-new-users)** |
**[Supported Models](#supported-models)** |
**[Examples](./examples)** |
**[Contributing](./CONTRIBUTING.md)** |
**[启智社区](https://git.openi.org.cn/GAMMALab/GammaGL)**

GammaGL is a multi-backend graph learning library based on [TensorLayerX](https://github.com/tensorlayer/TensorLayerX), which supports TensorFlow, PyTorch, PaddlePaddle, MindSpore as the backends.

We give a development tutorial in Chinese on [wiki](https://github.com/BUPT-GAMMA/GammaGL/wiki/%E5%BC%80%E5%8F%91%E8%80%85%E6%B5%81%E7%A8%8B).

## Highlighted Features

### Multi-backend

GammaGL supports multiple deep learning backends, such as TensorFlow, PyTorch, Paddle and MindSpore. Different from DGL, the GammaGL's examples are implemented with **the same code** on different backend. It allows users to run the same code on different hardwares like Nvidia-GPU and Huawei-Ascend. Besides, users could use a particular framework API based on preferences for different frameworks.

### PyG-Like

Following [PyTorch Geometric(PyG)](https://github.com/pyg-team/pytorch_geometric), GammaGL utilizes a tensor-centric API. If you are familiar with PyG, it will be friendly and maybe a TensorFlow Geometric, Paddle Geometric, or MindSpore Geometric to you.

## News
<details open>
<summary>2026-07-05 release v0.6.0
</summary>
</br>
We release GammaGL v0.6.0.

- Use one `gammagl` package for CPU and GPU environments, with source builds
  selected by `GAMMAGL_WITH_CUDA=0/1/auto`.
- Use the GAMMA Lab maintained TensorLayerX nightly branch for source builds.
- Keep LLM and graph foundation model dependencies optional through `llm`,
  `gfm`, and `llm-gfm` extras.
- Improve public API exports for common layers, datasets, transforms, loaders,
  models and utilities.
- Update installation guidance and release package metadata for Python 3.9+
  Linux environments.

</details>

<details>
<summary>2024-07-29 release v0.5
</summary>
</br>
We release version v0.5.

- 70 GNN models
- More fused operators
- Support GPU sample
- Support GraphStore and FeatureStore

</details>

<details>
<summary>2024-01-24 release v0.4
</summary>
</br>
We release version v0.4.

- 60 GNN models
- More fused operators and users can truly use these operators
- Support the latest version of PyTorch and MindSpore
- Support for graph database like neo4j

</details>

<details>
<summary>2023-07-12 release v0.3
</summary>
</br>
We release version v0.3.

- 50 GNN models
- Efficient message passing operators and fused operator
- Rebuild sampling architecture.

</details>

<details>

<summary>
2023-04-01 paper accepted
</summary>
<br/>

Our paper <i>GammaGL: A Multi-Backend Library for Graph Neural Networks</i> is accpeted at SIGIR 2023 resource paper track.

</details>

<details>

<summary>
2023-02-24 启智社区优秀孵化项目奖
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
We release version v0.2.

- 40 GNN models
- 20 datasets
- Efficient message passing operators and fused operator
- GPU sampling and heterogeneous graphs samplers.

</details>

<details>
<summary>2022-06-20 release v0.1
</summary>
</br>
We release version v0.1.

- Framework-agnostic design
- PyG-like
- Graph data structures, message passing module and sampling module
- 20+ GNN models

</details>

## Get Started

GammaGL 0.6.0 requires **Python >= 3.9** and is supported on **Linux**. Use the
same `gammagl` package for CPU and GPU; choose the backend wheel and extension
build mode during installation.

### Install from pip

For the released package, install a backend first, then install GammaGL:

```bash
pip install torch torchvision torchaudio
pip install gammagl
```

For CPU-only PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gammagl
```

For the latest source build, use the GAMMA Lab maintained TensorLayerX branch
and the source installation commands below.

### CPU Quick Start

```bash
conda create -n gammagl-cpu python=3.10
conda activate gammagl-cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
cd GammaGL
pip install pybind11 ninja
GAMMAGL_WITH_CUDA=0 TL_BACKEND=torch pip install -e ".[build]" --no-build-isolation
TL_BACKEND=torch python examples/gcn/gcn_trainer.py --dataset cora --n_epoch 1 --gpu -1
```

### GPU Quick Start

Choose the PyTorch CUDA wheel that matches your driver and CUDA runtime. For
example, with CUDA 12.1 wheels:

```bash
conda create -n gammagl-cu python=3.10
conda activate gammagl-cu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
git clone --recursive https://github.com/BUPT-GAMMA/GammaGL.git
cd GammaGL
pip install pybind11 ninja
GAMMAGL_WITH_CUDA=auto TL_BACKEND=torch pip install -e ".[build]" --no-build-isolation
TL_BACKEND=torch python examples/gcn/gcn_trainer.py --dataset cora --n_epoch 1 --gpu 0
```

`GAMMAGL_WITH_CUDA` accepts `0`, `1`, or `auto`. CPU-only installs should use
`0`; CUDA builds can use `auto` or `1` when CUDA headers and `nvcc` are
available.

For other backends, install the backend first, then install the GAMMA Lab
TensorLayerX branch:

```bash
pip install git+https://github.com/dddg617/tensorlayerx.git@nightly
```

PyTorch must be installed before installing this TensorLayerX branch. This
TensorLayerX is maintained by the **BUPT GAMMA Lab Team**.

### Optional LLM/GFM Extension

GraphGPT, LLaGA, LLMRec, WalkLM, NLGraph and related LLM/GFM utilities require
additional heavy dependencies. Install them only when using those features:

For the released package:

```bash
pip install "gammagl[llm-gfm]"
```

For a source checkout:

```bash
pip install pybind11 ninja
GAMMAGL_WITH_CUDA=auto TL_BACKEND=torch pip install -e ".[build,llm-gfm]" --no-build-isolation
```

Core GammaGL installation does not require `transformers`, `torch_geometric`,
`openai`, or `sentence_transformers`.

> 大陆用户如果遇到网络问题，推荐从启智社区安装：
>
> `git clone --recursive https://git.openi.org.cn/GAMMALab/GammaGL.git`
>
> If `--recursive` was omitted, run `git submodule update --init` in the
> GammaGL root directory.

## Quick Tour for New Users

In this quick tour, we highlight the ease of creating and training a GNN model with only a few lines of code.

### Train your own GNN model

In the first glimpse of GammaGL, we implement the training of a GNN for classifying papers in a citation graph.
For this, we load the [Cora](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid) dataset and train a 2-layer GCN with TensorLayerX's backend-neutral training API. The full version is available in [`examples/gcn/gcn_trainer.py`](./examples/gcn/gcn_trainer.py).

```python
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models import GCNModel
from gammagl.utils import add_self_loops, mask_to_index

class SemiSpvzLoss(WithLoss):
    def forward(self, data, y):
        logits = self.backbone_network(
            data["x"], data["edge_index"], None, data["num_nodes"]
        )
        train_logits = tlx.gather(logits, data["train_idx"])
        train_y = tlx.gather(data["y"], data["train_idx"])
        return self._loss_fn(train_logits, train_y)

dataset = Planetoid(root="./data", name="cora")
graph = dataset[0]
edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)

model = GCNModel(
    feature_dim=dataset.num_node_features,
    hidden_dim=16,
    num_class=dataset.num_classes,
    drop_rate=0.5,
    num_layers=2,
)
optimizer = tlx.optimizers.Adam(lr=0.01, weight_decay=5e-4)
train_one_step = TrainOneStep(
    SemiSpvzLoss(model, tlx.losses.softmax_cross_entropy_with_logits),
    optimizer,
    model.trainable_weights,
)
data = {
    "x": graph.x,
    "y": graph.y,
    "edge_index": edge_index,
    "train_idx": mask_to_index(graph.train_mask),
    "num_nodes": graph.num_nodes,
}

for epoch in range(200):
    model.set_train()
    loss = train_one_step(data, graph.y)
```

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


## How to Run

Take [GCN](./examples/gcn) as an example:

```bash
cd examples/gcn
TL_BACKEND=torch python gcn_trainer.py --dataset cora --lr 0.01 --n_epoch 200 --gpu 0
```

For CPU:

```bash
TL_BACKEND=torch python gcn_trainer.py --dataset cora --n_epoch 200 --gpu -1
```

For a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=1 TL_BACKEND=torch python gcn_trainer.py --dataset cora --gpu 0
```

For another backend, install that backend first and set `TL_BACKEND` explicitly:

```bash
TL_BACKEND=paddle python gcn_trainer.py --dataset cora --gpu 0
```

> Note
> 
> When `TL_BACKEND` is not set, GammaGL uses `torch` by default.
>
> Use `--gpu -1` for CPU execution.
> 
> The CANDIDATE backends are `tensorflow`, `paddle`, `torch` and `mindspore`.

## Supported Models

Now, GammaGL supports about 70 models, we welcome everyone to use or contribute models.

|                                                    | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| -------------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| [GCN [ICLR 2017]](./examples/gcn)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GAT [ICLR 2018]](./examples/gat)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GraphSAGE [NeurIPS 2017]](./examples/graphsage)   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [ChebNet [NeurIPS 2016]](./examples/chebnet)       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GCNII [ICLR 2017]](./examples/gcnii)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

<details>
<summary>You may see the other models here.</summary>

|                                                    | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| -------------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| [JKNet [ICML 2018]](./examples/jknet)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [SGC [ICML 2019]](./examples/sgc)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GIN [ICLR 2019]](./examples/gin)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [APPNP [ICLR 2019]](./examples/appnp)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [AGNN [arxiv]](./examples/agnn)                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [SIGN [ICML 2020 Workshop]](./examples/sign)       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [DropEdge [ICLR 2020]](./examples/dropedge)        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GPRGNN [ICLR 2021]](./examples/gprgnn)            | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |
| [GNN-FiLM [ICML 2020]](./examples/film)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GraphGAN [AAAI 2018]](./examples/graphgan)        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HardGAT [KDD 2019]](./examples/hardgat)           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [MixHop [ICML 2019]](./examples/mixhop)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [PNA [NeurIPS 2020]](./examples/pna)               | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [FAGCN [AAAI 2021]](./examples/fagcn)              | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |
| [GATv2 [ICLR 2021]](./examples/gatv2)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GEN [WWW 2021]](./examples/gen)                   | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |
| [GAE [NeurIPS 2016]](./examples/vgae)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [VGAE [NeurIPS 2016]](./examples/vgae)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HCHA [PR 2021]](./examples/hcha)                  |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Node2Vec [KDD 2016]](./examples/node2vec)         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [DeepWalk [KDD 2014]](./examples/deepwalk)         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [DGCNN [ACM T GRAPHIC 2019]](./examples/dgcnn)     | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |
| [GaAN [UAI 2018]](./examples/gaan)                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GMM [CVPR 2017]](./examples/gmm)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [TADW [IJCAI 2015]](./examples/tadw)               | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [MGNNI [NeurIPS 2022]](./examples/mgnni)           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [CAGCN [NeurIPS 2021]](./examples/cagcn)           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [DR-GST [WWW 2022]](./examples/drgst)              | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |
| [Specformer [ICLR 2023]](./examples/specformer)    |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [CoGSL [WWW 2022]](./examples/cogsl)               |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [AM-GCN [KDD 2020]](./examples/amgcn)              |                    | :heavy_check_mark: |                    | :heavy_check_mark: |
| [GGD [NeurIPS 2022]](./examples/ggd)               |                    | :heavy_check_mark: |                    | :heavy_check_mark: |
| [LTD [WSDM 2022]](./examples/ltd)                  |                    | :heavy_check_mark: |                    | :heavy_check_mark: |
| [Graphormer [NeurIPS 2021]](./examples/graphormer) |                    | :heavy_check_mark: |                    | :heavy_check_mark: |
| [HiD-Net [AAAI 2023]](./examples/hid_net)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [FusedGAT [MLSys 2022]](./examples/fusedgat)       |                    | :heavy_check_mark: |                    |                    |
| [GLNN [ICLR 2022]](./examples/glnn)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [DFAD-GNN [IJCAI 2022]](./examples/dfad_gnn)       |                    | :heavy_check_mark: |                    |                    |
| [GNN-LF-HF [WWW 2021]](./examples/gnnlfhf)         |                    | :heavy_check_mark: |                    |                    |
| [DNA [ICLR 2019]](./examples/dna)                  |                    | :heavy_check_mark: |                    |                    |


| Contrastive Learning                           | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| [DGI [ICLR 2019]](./examples/dgi)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GRACE [ICML 2020 Workshop]](./examples/grace) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [GRADE [NeurIPS 2022]](./examples/grade)       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [MVGRL [ICML 2020]](./examples/mvgrl)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [InfoGraph [ICLR 2020]](./examples/infograph)  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [MERIT [IJCAI 2021]](./examples/merit)         | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |
| [GNN-POT [NeurIPS 2023]](./examples/grace_pot) |                    | :heavy_check_mark: |                    |                    |
| [MAGCL [AAAI 2023]](./examples/magcl)          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Sp2GCL [NeurIPS 2023]](./examples/sp2gcl)     |                    | :heavy_check_mark: |                    |                    |

| Heterogeneous Graph Learning                       | TensorFlow         | PyTorch            | Paddle             | MindSpore          |
| -------------------------------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| [RGCN [ESWC 2018]](./examples/rgcn)                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HAN [WWW 2019]](./examples/han)                   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HGT [WWW 2020]](./examples/hgt/)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [SimpleHGN [KDD 2021]](./examples/simplehgn)       | :heavy_check_mark: |                    |                    | :heavy_check_mark: |
| [CompGCN [ICLR 2020]](./examples/compgcn)          |                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HPN [TKDE 2021]](./examples/hpn)                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [ieHGCN [TKDE 2021]](./examples/iehgcn)            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [MetaPath2Vec [KDD 2017]](./examples/metapath2vec) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HERec [TKDE 2018]](./examples/herec)              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [HeCo [KDD 2021]](./examples/heco)                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [DHN [TKDE 2023]](./examples/dhn)                  |                    | :heavy_check_mark: |                    |                    |
| [HEAT [T-ITS 2023]](./examples/heat)               |                    | :heavy_check_mark: |                    |                    |

> Note
> 
> The models can be run in mindspore backend. Howerver, the results of experiments are not satisfying due to training component issue,
> which will be fixed in future.
</details>

## Contributors

GammaGL Team[GAMMA LAB] and Peng Cheng Laboratory.

See more in [CONTRIBUTING](./CONTRIBUTING.md).

Contribution is always welcomed. Please feel free to open an issue or email to cuishanyuanai@bupt.edu.cn.

## Cite GammaGL
If you use GammaGL in a scientific publication, we would appreciate citations to the following paper:

```
@inproceedings{10.1145/3539618.3591891,
author = {Liu, Yaoqi and Yang, Cheng and Zhao, Tianyu and Han, Hui and Zhang, Siyuan and Wu, Jing and Zhou, Guangyu and Huang, Hai and Wang, Hui and Shi, Chuan},
title = {GammaGL: A Multi-Backend Library for Graph Neural Networks},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591891},
doi = {10.1145/3539618.3591891},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2861–2870},
numpages = {10},
keywords = {graph neural networks, frameworks, deep learning},
location = {, Taipei, Taiwan, },
series = {SIGIR '23}
}

```
